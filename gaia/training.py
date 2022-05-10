from collections import OrderedDict
import json
import os

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
from pytorch_lightning.callbacks import ModelCheckpoint
from gaia.plot import levels26, levels
import yaml

from gaia import get_logger

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


def update_model_params_from_dataset(dataset_dict, model_params, mean_thres=1e-13):

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
        ignore_outputs = (
            dataset_dict["stats"]["output_stats"]["mean"].abs() < mean_thres
        )
        loss_output_weights = torch.ones(ignore_outputs.shape[0])
        loss_output_weights[ignore_outputs] = 0.0
        logger.info(f"ignoring {ignore_outputs.sum()} outputs with mean < {mean_thres}")

        model_params["loss_output_weights"] = loss_output_weights.tolist()


def get_train_val_test_split(files, interleave=False, seperate_val_set=False):
    raise DeprecationWarning
    if not interleave:
        # use last 10% for test
        assert seperate_val_set
        N = int(len(files) * 0.1)
        train_files = files[: -N * 2]
        val_files = files[-N * 2 : -N]
        test_files = files[-N:]
        return train_files, val_files, test_files
    else:
        train_files = []
        test_files = []
        if seperate_val_set:
            val_files = []
            for i, f in enumerate(files):
                day_mod_30 = i % 30
                if day_mod_30 >= 27:
                    test_files.append(f)
                # elif (day_mod_30 < 13) & (day_mod_30 >= 10):
                #     val_files.append(f)
                else:
                    train_files.append(f)

            N = len(test_files)

            from random import shuffle

            shuffle(train_files)

            val_files = train_files[:N]
            train_files = train_files[N:]

            logger.warning("overwriting with predefined split")

            temp = json.load(open("/ssddg1/gaia/cache/files_split.json"))
            train_files = temp["train"]
            val_files = temp["val"]

            return train_files, val_files, test_files
        else:
            # last 3 days of every 30 days
            for i, f in enumerate(files):
                if i % 30 < 27:
                    train_files.append(f)
                else:
                    test_files.append(f)

            return train_files, test_files, test_files


def default_dataset_params(
    batch_size=24 * 96 * 144,
    base="/ssddg1/gaia/spcam/spcamclbm-nx-16-20m-timestep_4",
    mean_thres=1e-13,
):

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


def default_dataset_params_v1(
    batch_size=24 * 96 * 144,
    subsample_factor=12,
    flatten=True,
    interleave=True,
    seperate_val_set=False,
    inputs=["T", "Q", "RELHUM", "U", "V"],
    outputs=["PTEQ", "PTTEND", "PRECT"],
):
    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    train_files, val_files, test_files = get_train_val_test_split(
        files, interleave=interleave, seperate_val_set=seperate_val_set
    )

    return dict(
        train=dict(
            files=train_files,
            subsample_factor=subsample_factor,
            batch_size=batch_size,
            shuffle=True,
            in_memory=True,
            flatten=flatten,
            compute_stats=True,
            inputs=inputs,
            outputs=outputs,
        ),
        val=dict(
            files=val_files,
            subsample_factor=subsample_factor,
            batch_size=batch_size,
            shuffle=True,
            in_memory=True,
            flatten=flatten,
            compute_stats=False,
            inputs=inputs,
            outputs=outputs,
        ),
        test=dict(
            files=test_files,
            subsample_factor=1,
            batch_size=batch_size,
            shuffle=False,
            in_memory=True,
            flatten=False,
            compute_stats=False,
            inputs=inputs,
            outputs=outputs,
        ),
    )


def default_model_params(**kwargs):
    d = dict(
        lr=1e-3,
        optimizer="adam",
        model_config={
            "model_type": "fcn_history",
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
    trainer_params=default_trainer_params(),
    dataset_params=default_dataset_params(),
    model_params=default_model_params(),
    seed=None,
    interpolation_params = None
):
    if seed:
        logger.info("seeding everything")
        pl.seed_everything(345)

    logger.info("starting a run with:")
    logger.info(f"trainer_params: \n{make_pretty_for_log(trainer_params)}")
    logger.info(f"dataset_params: \n{make_pretty_for_log(dataset_params)}")
    logger.info(f"model_params: \n{make_pretty_for_log(model_params)}")

    mode = mode.split(",")

    model_dir = None
    trainer = None
    val_dataloader = None

    if "train" in mode:
        train_dataset, train_dataloader = get_dataset(**dataset_params["train"])
        val_dataset, val_dataloader = get_dataset(**dataset_params["val"])

        # val_dataset, val_dataloader = get_dataset(**dataset_params["val"])
        # test_dataset, test_dataloader =     get_dataset(dataset_params["test"])

        mean_thres = dataset_params["mean_thres"]

        update_model_params_from_dataset(
            train_dataset, model_params, mean_thres=mean_thres
        )

       

        model = TrainingModel(dataset_params=dataset_params, **model_params)

        checkpoint_callback = ModelCheckpoint(monitor="val_mse", mode="min")

        # write_graph = WriteGraph()

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

        #bug
        if "var_index_file" not in model.hparams.dataset_params["test"]:
            logger.info("adding var index file")
            model.hparams.dataset_params["test"]["var_index_file"] = model.hparams.dataset_params["test"]["dataset_file"].replace("_test.pt", "_var_index.pt")


        test_dataset, test_dataloader = get_dataset(**model.hparams.dataset_params["test"])

        trainer = pl.Trainer(
            checkpoint_callback=False,
            logger=False,
            **trainer_params,
        )

        test_results = trainer.test(model, dataloaders=test_dataloader)
        # run_dir = os.path.split(os.path.split(model_params["ckpt"])[0])[0]
        path_to_save = os.path.join(model_dir, "test_results.json")
        json.dump(test_results, open(path_to_save, "w"))

    if "predict" in mode:

        if model_dir is None:
            assert "ckpt" in model_params
            model_dir = model_params["ckpt"]


        model = TrainingModel.load_from_checkpoint(
            get_checkpoint_file(model_dir),
            strict=False,
            **{"interpolate": interpolation_params},
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

        process_results(model_dir, levels=None)
