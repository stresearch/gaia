from collections import OrderedDict
from datetime import date
import json
import os
from cv2 import log

from gaia.callbacks import WriteGraph
from gaia.evaluate import process_results
from gaia.export import export
from gaia.models import ComputeStats, TrainingModel
from gaia.data import (
    make_dummy_dataset,
    FastTensorDataset,
    get_dataset,
    unflatten_tensor,
)
import glob
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from gaia.plot import levels26, levels
from gaia.config import get_levels
import yaml

from gaia import get_logger

# from gaia.config import Config
from gaia.config import Config

logger = get_logger(__name__)


def get_checkpoint_file(file):
    if file.endswith(".ckpt"):
        return file
    else:
        # assume its a directory
        pattern = os.path.join(file, "checkpoints", "*.ckpt")
        files = glob.glob(pattern)
        if len(files) > 0:
            return files[0]
        else:
            raise FileNotFoundError(f"no ckpt files found in {pattern}")


def update_model_params_from_dataset(
    dataset_dict, model_params, mean_thres=1e-13, levels=None
):

    model_params["model_config"].update(
        {
            "input_size": list(dataset_dict["input_index"].values())[-1][
                -1
            ],  # ordered dict of {"var":(s,e)}
            "output_size": list(dataset_dict["output_index"].values())[-1][-1],
        }
    )

    model_params.update(
        dict(
            input_index=dataset_dict["input_index"],
            output_index=dataset_dict["output_index"],
            data_stats=dataset_dict["stats"],
        )
    )

    if mean_thres > 0:
        ### ignore outputs with very small numbers
        ignore_outputs = (
            dataset_dict["stats"]["output_stats"]["mean"].abs() < mean_thres
        )
        loss_output_weights = torch.ones(ignore_outputs.shape[0])
        loss_output_weights[ignore_outputs] = 0.0
        logger.info(f"ignoring {ignore_outputs.sum()} outputs with mean < {mean_thres}")

        if levels:

            if model_params.get("upweigh_low_levels", False):
                logger.info("upweighing low levels")
                level_weights = torch.linspace(0.1, 1.0, len(levels))
            else:
                level_weights = torch.tensor([0] + levels).diff()

            temp = []
            for o, (s, e) in dataset_dict["output_index"].items():
                if e - s > 1:
                    temp.append(level_weights)
                else:
                    temp.append(torch.ones(1) * level_weights.mean())

            level_weights = torch.cat(temp)

            loss_output_weights = level_weights * loss_output_weights

        model_params["loss_output_weights"] = loss_output_weights.tolist()


def load_hparams_file(model_dir):
    yaml_file = os.path.join(model_dir, "hparams.yaml")
    if os.path.exists(yaml_file):
        params = yaml.unsafe_load(open(yaml_file))
        return params


def make_pretty_for_log(d, max_char=100):
    return "\n".join(
        [
            f"{k}: {str(v)[:max_char] if not isinstance(v,dict) else make_pretty_for_log(v)}"
            for k, v in d.items()
        ]
    )


def main(
    mode="train",
    trainer_params=Config.set_trainer_params(),
    dataset_params=Config.set_dataset_params(),
    model_params=Config.set_model_params(),
    seed=345,
    interpolation_params=None,
):
    if seed is not None:
        logger.info("seeding everything")
        pl.seed_everything(seed)

    model_dir = None

    logger.info("starting a run with:")
    # logger.info(f"trainer_params: \n{make_pretty_for_log(trainer_params)}")
    # logger.info(f"dataset_params: \n{make_pretty_for_log(dataset_params)}")
    # logger.info(f"model_params: \n{make_pretty_for_log(model_params)}")

    model_dir = None
    trainer = None
    val_dataloader = None

    if "train" in mode:

        logger.info("**** TRAINING ******")

        model_grid = model_params.get(
            "model_grid", dataset_params["train"]["data_grid"]
        )

        train_dataset, train_dataloader = get_dataset(
            **dataset_params["train"], model_grid=model_grid
        )
        val_dataset, val_dataloader = get_dataset(
            **dataset_params["val"], model_grid=model_grid
        )

        mean_thres = dataset_params["mean_thres"]

        update_model_params_from_dataset(
            train_dataset,
            model_params,
            mean_thres=mean_thres,
            levels=model_grid,
        )

        model = TrainingModel(dataset_params=dataset_params, seed = seed, **model_params)

        checkpoint_callback = ModelCheckpoint(monitor="val_mse", mode="min")

        callbacks=[checkpoint_callback, LearningRateMonitor()]

        swa_lrs =  model_params.get("swa_lrs",None)
        if swa_lrs is not None:
            logger.info("using SWA")
            swa = StochasticWeightAveraging(swa_lrs=swa_lrs)
            callbacks.append(swa)

        # write_graph = WriteGraph()


        from pytorch_lightning.profilers import SimpleProfiler

        profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

        trainer = pl.Trainer(
            callbacks=callbacks,
            log_every_n_steps=max(1, len(train_dataloader) // 100),
            profiler = profiler,
            **trainer_params,
        )



        if model_params.get("ckpt", None) is not None:
            logger.info(f"loading existing ckpt {model_params}")
            ckpt = get_checkpoint_file(model_params["ckpt"])
        else:
            ckpt = None

        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)
        model_dir = trainer.log_dir

    if "finetune" in mode:

        logger.info("**** FINE TUNING ******")

        if model_dir is None:
            assert "ckpt" in model_params
            model_dir = model_params["ckpt"]

        model_params_to_update = {k: model_params[k] for k in ["lr", "lr_schedule"]}
        model_params_to_update["dataset_params"] = dataset_params
        model_params_to_update["is_finetuned"] = True

        model = TrainingModel.load_from_checkpoint(
            get_checkpoint_file(model_dir),
            fine_tuning=False,
            map_location="cpu",
            **model_params_to_update,
        )

        # disable everything but last later
        # model.requires_grad_(False)
        # model.model.model[-1].requires_grad_(True)

        model_grid = model.hparams.get("model_grid", None)

        assert model_grid

        train_dataset, train_dataloader = get_dataset(
            **dataset_params["train"], model_grid=model_grid
        )
        val_dataset, val_dataloader = get_dataset(
            **dataset_params["val"], model_grid=model_grid
        )

        assert train_dataset["input_index"] == model.hparams.input_index
        assert train_dataset["output_index"] == model.hparams.output_index

        mean_thres = dataset_params["mean_thres"]

        checkpoint_callback = ModelCheckpoint(monitor="val_mse", mode="min")

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, LearningRateMonitor()],
            log_every_n_steps=max(1, len(train_dataloader) // 100),
            **trainer_params,
        )

        trainer.fit(model, train_dataloader, val_dataloader)
        model_dir = trainer.log_dir

    if "val" in mode:

        logger.info("**** VALIDATING ******")

        if model_dir is None:
            assert "ckpt" in model_params
            model_dir = model_params["ckpt"]

        # compute validation on best model
        model = TrainingModel.load_from_checkpoint(get_checkpoint_file(model_dir))

        if val_dataloader is None:
            if dataset_params is None:
                logger.info(
                    "no dataset_params provided, using saved ones in checkpoint"
                )
                dataset_params = model.hparams.dataset_params

                # if "var_index_file" not in model.hparams.dataset_params["val"]:
                #     logger.info("adding var index file")
                #     model.hparams.dataset_params["val"][
                #         "var_index_file"
                #     ] = model.hparams.dataset_params["val"]["dataset_file"].replace(
                #         "_val.pt", "_var_index.pt"
                #     )

            val_dataset, val_dataloader = get_dataset(
                **dataset_params["val"],
                model_grid=model.hparams.get("model_grid", None),
            )

        if trainer is None:
            trainer = pl.Trainer(
                checkpoint_callback=False,
                logger=False,
                **trainer_params,
            )

        validation_score = trainer.validate(model, val_dataloader)
        json.dump(
            validation_score,
            open(os.path.join(model_dir, "validation_score.json"), "w"),
        )

    if ("test" in mode) or ("predict" in mode):

        if model_dir is None:
            assert "ckpt" in model_params
            model_dir = model_params["ckpt"]

        model = TrainingModel.load_from_checkpoint(get_checkpoint_file(model_dir))

        # model.model.scale = 16

        if dataset_params is None:
            logger.info("no dataset params provided, using saved ones in checkpoint")

            # # bug
            # if "var_index_file" not in model.hparams.dataset_params["test"]:
            #     logger.info("adding var index file")
            #     model.hparams.dataset_params["test"][
            #         "var_index_file"
            #     ] = model.hparams.dataset_params["test"]["dataset_file"].replace(
            #         "_test.pt", "_var_index.pt"
            #     )

            dataset_params = model.hparams.dataset_params

        # model.hparams.dataset_params["test"]["batch_size"] = 96*144

        model_grid = model.hparams.get("model_grid", None)
        if model_grid is None:
            logger.info("model grid is not found... trying to infer from dataset")
            dataset = model.hparams.dataset_params.get("dataset", None)
            if dataset:
                model_grid = get_levels(dataset)

        split = "test"

        if split != "test":
            logger.warning("split for testing is not set to test? is that right")

        test_dataset, test_dataloader = get_dataset(
            **dataset_params[split], model_grid=model_grid
        )

        if "test" in mode:
            logger.info("**** TESTING ******")

            trainer = pl.Trainer(
                checkpoint_callback=False,
                logger=False,
                **trainer_params,
            )

            test_results = trainer.test(model, dataloaders=test_dataloader)
            # run_dir = os.path.split(os.path.split(model_params["ckpt"])[0])[0]
            dataset = dataset_params.get("dataset", "")
            path_to_save = os.path.join(model_dir, f"{split}_results_{dataset}.json")
            json.dump(test_results, open(path_to_save, "w"))

        if "predict" in mode:

            logger.info("**** PREDICTING ******")

            dataset = dataset_params.get("dataset", "")
            prediction_file_name = f"predictions_{dataset}.pt"

            trainer = pl.Trainer(
                log_every_n_steps=max(1, len(test_dataloader) // 100),
                checkpoint_callback=False,
                logger=False,
                **trainer_params,
            )

            yhat = trainer.predict(model, dataloaders=test_dataloader)
            # return yhat
            yhat = torch.cat(yhat)

            if len(yhat.shape) in [2, 3]:
                yhat = unflatten_tensor(yhat)

            # run_dir = os.path.split(os.path.split(model_params["ckpt"])[0])[0]
            path_to_save = os.path.join(model_dir, prediction_file_name)

            torch.save(yhat, path_to_save)

    if "results" in mode:

        if model_dir is None:
            assert "ckpt" in model_params
            model_dir = model_params["ckpt"]

        logger.info("processing results")

        other_predictions = None
        # if "cam4" in model_dir:
        #     other_predictions = "lightning_logs_compare_models/spcam_nn/predictions_on_cam4.pt"
        # else:
        #     other_predictions = "lightning_logs_compare_models/cam4_nn/predictions_on_spcam.pt"

        process_results(model_dir, levels=None, other_predictions=other_predictions)

    if "export" in mode:
        timestamp = date.today().strftime("%m-%d-%y")
        export_name = f"export_model_{timestamp}.pt"
        export(model_dir, export_name, inputs=None, outputs=None)


    if "plot" in mode:
        from gaia.plot import save_diagnostic_plot, save_gradient_plots
        save_gradient_plots(model_dir, device=f"cuda:{trainer_params['gpus'][0]}")
        save_diagnostic_plot(model_dir)

    return model_dir


def get_dataset_from_model(model, split="test"):

    model_grid = model.hparams.get("model_grid", None)
    if model_grid is None:
        logger.info("model grid is not found... trying to infer from dataset")
        dataset = model.hparams.dataset_params.get("dataset", None)
        if dataset:
            model_grid = get_levels(dataset)

    dataset_params = model.hparams.dataset_params

    test_dataset, test_dataloader = get_dataset(
        **dataset_params[split], model_grid=model_grid
    )

    return test_dataset, test_dataloader
