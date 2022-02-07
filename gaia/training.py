from asyncio.log import logger
from collections import OrderedDict
import json
import os
from gaia.models import ComputeStats, TrainingModel
from gaia.data import (
    NcDatasetMem,
    make_dummy_dataset,
    NcIterableDataset,
    FastTensorDataset,
    get_dataset,
    unflatten,
)
import glob
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

from gaia import get_logger

logger = get_logger(__name__)





def update_model_params_from_dataset(dataset_dict, model_params):

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


def get_train_val_test_split(files, interleave=False):
    if not interleave:
        # use last 10% for test
        N = int(len(files) * 0.1)
        train_files = files[: -N * 2]
        val_files = files[-N * 2 : -N]
        test_files = files[-N:]
        return train_files, val_files, test_files
    else:
        train_files = []
        test_files = []
        # last 3 days of every 30 days
        for i, f in enumerate(files):
            if i % 30 < 27:
                train_files.append(f)
            else:
                test_files.append(f)

        return train_files, test_files, test_files


def default_dataset_params(
    batch_size=24 * 96 * 144,
    subsample_factor=12,
    flatten=True,
    interleave=True,
    inputs=["T", "Q", "RELHUM", "U", "V"],
    outputs=["PTEQ", "PTTEND", "PRECT"],
):
    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    train_files, val_files, test_files = get_train_val_test_split(
        files, interleave=interleave
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
            shuffle=False,
            in_memory=True,
            flatten=False,
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
            "model_type": "fcn",
            "num_layers": 7,
        },
    )

    d.update(kwargs)

    return d


def default_trainer_params(gpus=None):
    return dict(gpus=gpus, precision=16)


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
):
    pl.seed_everything(345)

    logger.info("starting a run with:")
    logger.info(f"trainer_params: \n{make_pretty_for_log(trainer_params)}")
    logger.info(f"dataset_params: \n{make_pretty_for_log(dataset_params)}")
    logger.info(f"model_params: \n{make_pretty_for_log(model_params)}")

    if mode == "train":
        train_dataset, train_dataloader = get_dataset(**dataset_params["train"])
        val_dataset, val_dataloader = get_dataset(
            flatten_anyway=True, **dataset_params["val"]
        )
        # test_dataset, test_dataloader =     get_dataset(dataset_params["test"])

        update_model_params_from_dataset(train_dataset, model_params)

        model = TrainingModel(dataset_params=dataset_params, **model_params)
        trainer = pl.Trainer(
            log_every_n_steps=max(1, len(train_dataloader) // 100), **trainer_params
        )
        trainer.fit(model, train_dataloader, val_dataloader)

    elif mode == "test":
        assert "ckpt" in model_params

        model = TrainingModel.load_from_checkpoint(model_params["ckpt"])
        test_dataset, test_dataloader = get_dataset(
            flatten_anyway=True, **dataset_params["test"]
        )

        trainer = pl.Trainer(
            log_every_n_steps=max(1, len(test_dataloader) // 100),
            checkpoint_callback=False,
            logger=False,
            **trainer_params,
        )
        test_results = trainer.test(model, dataloaders=test_dataloader)
        run_dir = os.path.split(os.path.split(model_params["ckpt"])[0])[0]
        path_to_save = os.path.join(run_dir, "test_results.json")
        json.dump(test_results, open(path_to_save, "w"))

    elif mode == "predict":
        assert "ckpt" in model_params

        model = TrainingModel.load_from_checkpoint(model_params["ckpt"])
        test_dataset, test_dataloader = get_dataset(
            flatten_anyway=True, **dataset_params["test"]
        )

        trainer = pl.Trainer(
            log_every_n_steps=max(1, len(test_dataloader) // 100),
            checkpoint_callback=False,
            logger=False,
            **trainer_params,
        )
        yhat = trainer.predict(model, dataloaders=test_dataloader)
        # return yhat
        yhat = torch.cat(yhat)

        if len(yhat.shape) == 2:
            yhat = unflatten(yhat)

        save_dir = os.path.join(
            os.path.split(model_params["ckpt"])[0], "predictions.pt"
        )
        torch.save(yhat, save_dir)

