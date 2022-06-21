from collections import OrderedDict
import json
import os
from cv2 import log

from sympy import interpolate
from gaia.callbacks import WriteGraph
from gaia.evaluate import process_results
from gaia.models import ComputeStats, TrainingModel
from gaia.data import (
    NcDatasetMem,
    make_dummy_dataset,
    NcIterableDataset,
    FastTensorDataset,
    get_dataset,
    unflatten_tensor,
)
import glob
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from gaia.plot import levels26, levels, get_levels
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


def update_model_params_from_dataset(dataset_dict, model_params, mean_thres=1e-13, levels = None):

    model_params["model_config"].update(
        {
            "input_size": dataset_dict["x"].shape[-1],
            "output_size": dataset_dict["y"].shape[-1],
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
            level_weights = torch.tensor([0] + levels).diff()
            temp = []
            for o,(s,e) in dataset_dict["output_index"].items():
                if e-s > 1:
                    temp.append(level_weights)
                else:
                    temp.append(torch.ones(1)*level_weights.mean())

            level_weights = torch.cat(temp)

            loss_output_weights = level_weights*loss_output_weights


        model_params["loss_output_weights"] = loss_output_weights.tolist()

    


def default_dataset_params(
    batch_size=24 * 96 * 144,
    base="/ssddg1/gaia/spcam/spcamclbm-nx-16-20m-timestep_4",
    mean_thres=1e-13,
):
    logger.warn("depreciated")
    var_index_file = base + "_var_index.pt"

    return dict(
        train=dict(
            dataset_file=base + "_train.pt",
            batch_size=batch_size,
            shuffle=True,
            flatten=False,  # already flattened
            var_index_file=var_index_file,
        ),
        val=dict(
            dataset_file=base + "_val.pt",
            batch_size=batch_size,
            shuffle=False,
            flatten=False,  # already flattened
            var_index_file=var_index_file,
        ),
        test=dict(
            dataset_file=base + "_test.pt",
            batch_size=batch_size,
            shuffle=False,
            flatten=True,  # already flattened
            var_index_file=var_index_file,
        ),
        mean_thres=mean_thres,
    )

def default_model_params(**kwargs):
    logger.exception(DeprecationWarning)
    d = dict(
        lr=1e-3,
        optimizer="adam",
        model_config={
            "model_type": "fcn",
            "num_layers": 7,
        },
    )

    d.update(kwargs)

    return d


def load_hparams_file(model_dir):
    yaml_file = os.path.join(model_dir, "hparams.yaml")
    if os.path.exists(yaml_file):
        params = yaml.unsafe_load(open(yaml_file))
        return params


def default_trainer_params(**kwargs):
    logger.exception(DeprecationWarning)
    d = dict(precision=16,max_epochs=200)
    d.update(kwargs)
    return d


def make_pretty_for_log(d, max_char=100):
    return "\n".join(
        [
            f"{k}: {str(v)[:max_char] if not isinstance(v,dict) else make_pretty_for_log(v)}"
            for k, v in d.items()
        ]
    )



def main(
    mode="train",
    trainer_params=Config().trainer_params,
    dataset_params=Config().dataset_params,
    model_params=Config().model_params,
    seed=Config().seed,
    interpolation_params = Config().interpolation_params
):
    if seed:
        logger.info("seeding everything")
        pl.seed_everything(345)

    logger.info("starting a run with:")
    logger.info(f"trainer_params: \n{make_pretty_for_log(trainer_params)}")
    logger.info(f"dataset_params: \n{make_pretty_for_log(dataset_params)}")
    logger.info(f"model_params: \n{make_pretty_for_log(model_params)}")

    model_dir = None
    trainer = None
    val_dataloader = None

    if "train" in mode:
        train_dataset, train_dataloader = get_dataset(**dataset_params["train"])
        val_dataset, val_dataloader = get_dataset(**dataset_params["val"])

        mean_thres = dataset_params["mean_thres"]

        #TODO fix
        dataset_name = "cam4" if "cam4" in dataset_params["train"]["dataset_file"] else "spcam"

        update_model_params_from_dataset(
            train_dataset, model_params, mean_thres=mean_thres, levels = get_levels(dataset_name)
        )

       

        model = TrainingModel(dataset_params=dataset_params, **model_params)

        checkpoint_callback = ModelCheckpoint(monitor="val_mse", mode="min")

        # write_graph = WriteGraph()

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, LearningRateMonitor()],
            log_every_n_steps=max(1, len(train_dataloader) // 100),
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
        #assuming we'll fine-tune on a different dataset
        if model_params.get("pretrained", None) is not None:
            logger.info(f"loading pretrained {model_params}")
            pretrained = get_checkpoint_file(model_params["pretrained"])
        else:
            raise ValueError("need to specify pretrained to fine-tune")

        mean_thres = dataset_params["mean_thres"]

        train_dataset, train_dataloader = get_dataset(**dataset_params["train"])
        val_dataset, val_dataloader = get_dataset(**dataset_params["val"])

        ## update interpolation params
        interpolation_params = dict()
        interpolation_params["input_grid"] = levels
        interpolation_params["output_grid"] = levels26
        interpolation_params["optimize"]=False
        interpolation_params["model_config"] = dict()

        update_model_params_from_dataset(
            train_dataset, interpolation_params, mean_thres=mean_thres
        )

        model = TrainingModel.load_from_checkpoint(pretrained, strict = False,  **{"interpolate": interpolation_params})

        # update output normalizaton
        model.add_module("output_normalize", model.get_normalization(interpolation_params["data_stats"]["output_stats"], zero_mean=True))

        # update output weights
        model.make_output_weights(interpolation_params["loss_output_weights"])
        model.hparams.loss_output_weights = interpolation_params["loss_output_weights"]

        checkpoint_callback = ModelCheckpoint(monitor="val_mse", mode="min")

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            log_every_n_steps=max(1, len(train_dataloader) // 100),
            **trainer_params,
        )

        if model_params.get("ckpt", None) is not None:
            logger.info(f"loading existing ckpt {model_params}")
            ckpt = get_checkpoint_file(model_params["ckpt"])
        else:
            ckpt = None

        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)
        model_dir = trainer.log_dir



    if "val" in mode:
        if model_dir is None:
            assert "ckpt" in model_params
            model_dir = model_params["ckpt"]

        # compute validation on best model
        model = TrainingModel.load_from_checkpoint(get_checkpoint_file(model_dir))

        if val_dataloader is None:
            dataset_params = model.hparams.dataset_params

            if "var_index_file" not in model.hparams.dataset_params["val"]:
                logger.info("adding var index file")
                model.hparams.dataset_params["val"]["var_index_file"] = model.hparams.dataset_params["val"]["dataset_file"].replace("_val.pt", "_var_index.pt")


            val_dataset, val_dataloader = get_dataset(**dataset_params["val"])

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

    if "test" in mode:

        if model_dir is None:
            assert "ckpt" in model_params
            model_dir = model_params["ckpt"]

        model = TrainingModel.load_from_checkpoint(get_checkpoint_file(model_dir))

        # model.model.scale = 16

        #bug
        if "var_index_file" not in model.hparams.dataset_params["test"]:
            logger.info("adding var index file")
            model.hparams.dataset_params["test"]["var_index_file"] = model.hparams.dataset_params["test"]["dataset_file"].replace("_test.pt", "_var_index.pt")


        # model.hparams.dataset_params["test"]["batch_size"] = 96*144

        test_dataset, test_dataloader = get_dataset(**model.hparams.dataset_params["test"])

        trainer = pl.Trainer(
            checkpoint_callback=False,
            logger=False,
            **trainer_params,
        )

        test_results = trainer.test(model, dataloaders=test_dataloader)
        # run_dir = os.path.split(os.path.split(model_params["ckpt"])[0])[0]
        path_to_save = os.path.join(model_dir, f"test_results.json")
        json.dump(test_results, open(path_to_save, "w"))

    if "predict" in mode:

        if model_dir is None:
            assert "ckpt" in model_params
            model_dir = model_params["ckpt"]


        model = TrainingModel.load_from_checkpoint(
            get_checkpoint_file(model_dir),
            strict=False,
            **{"interpolate": interpolation_params, "predict_hidden_states" : False},
        )

        ### loading a different dataset
        if interpolation_params:
            logger.info("running interpolation")
            test_dataset, test_dataloader = get_dataset(**dataset_params["test"])
            prediction_file_name = interpolation_params["prediction_file_name"]
        else:
            test_dataset, test_dataloader = get_dataset(**model.hparams.dataset_params["test"])
            prediction_file_name = "predictions.pt"


       

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
