from asyncio.log import logger
from collections import OrderedDict
from gaia.models import ComputeStats, TrainingModel
from gaia.data import (
    NcDatasetMem,
    make_dummy_dataset,
    NcIterableDataset,
    FastTensorDataset,
    get_dataset,
)
import glob
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

from gaia import get_logger

logger = get_logger(__name__)

inputs = ["T", "Q", "RELHUM", "U", "V"]
outputs = ["PTEQ", "PTTEND", "PRECT", "TTEND_TOT"]

def compute_stats():
    model = ComputeStats()
    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))[:-34]
    train_dataset = NcIterableDataset(
        files,
        max_files_in_memory=10,
        batch_size=10 * 48,
        shuffle=False,
        flatten=False,
        inputs=inputs,
        outputs=outputs,
    )
    train_data_loader = DataLoader(train_dataset, batch_size=None, pin_memory=True)
    trainer = pl.Trainer(
        gpus=[2],
        checkpoint_callback=False,
        logger=False,
        precision=16,
    )

    outs = trainer.predict(model, train_data_loader)
    outs = model.process_predictions(outs)

    out_dict = OrderedDict()
    out_dict["input_stats"] = outs[0]
    out_dict["output_stats"] = outs[1]
    out_dict["inputs"] = inputs
    out_dict["outputs"] = outputs
    out_dict["files"] = files

    torch.save(out_dict, "stats.pt")


def construct_data():
    pl.seed_everything(345)

    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    N = 34
    train_files = files[: -34 * 2]
    val_files = files[-34 * 2 : -34]
    test_files = files[-34:]

    datasets = []

    for f_list in [train_files]:  # , val_files]:

        dataset = NcIterableDataset(
            f_list,
            max_files_in_memory=1,
            batch_size=24,
            shuffle=False,
            flatten=True,  # False  -> use "globe" images
            inputs=inputs,
            outputs=outputs,
            subsample_factor=12,
        )

        datasets.append(dataset.get_tensors(cache_dir="/ssddg1/gaia/cache"))

    (train_dataset,) = datasets

    if "stats" not in train_dataset:
        train_dataset["stats"] = dict(
            input_stats=dataset.compute_stats(train_dataset["x"]),
            output_stats=dataset.compute_stats(train_dataset["y"]),
        )

        dataset.save(train_dataset, "/ssddg1/gaia/cache")


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


def default_dataset_params(batch_size=24 * 96 * 144, subsample_factor=12, flatten=True):
    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    N = 34
    train_files = files[: -N * 2]
    val_files = files[-N * 2 : -N]
    test_files = files[-N:]

    return dict(
        train=dict(
            files=train_files,
            subsample_factor=subsample_factor,
            batch_size=batch_size,
            shuffle=True,
            in_memory=True,
            flatten=flatten,
        ),
        val=dict(
            files=val_files,
            subsample_factor=subsample_factor,
            batch_size=batch_size,
            shuffle=False,
            in_memory=True,
            flatten=flatten,
        ),
        test=dict(
            files=test_files,
            subsample_factor=1,
            batch_size=batch_size,
            shuffle=False,
            in_memory=True,
            flatten=flatten,
        ),
    )


def default_model_params():
    return dict(
        model_config={
            "model_type": "fcn",
            "num_layers": 7,
        }
    )


def default_trainer_params(gpus=None):
    return dict(gpus=gpus, precision=16)


def make_pretty_for_log(d,max_char = 100):
    return "\n".join([f"{k}: {str(v)[:max_char] if not isinstance(v,dict) else make_pretty_for_log(v)}" for k,v in d.items()])

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
        val_dataset, val_dataloader = get_dataset(**dataset_params["val"])
        # test_dataset, test_dataloader =     get_dataset(dataset_params["test"])

        update_model_params_from_dataset(train_dataset, model_params)

        model = TrainingModel(dataset_params=dataset_params, **model_params)
        trainer = pl.Trainer(
            log_every_n_steps=max(1, len(train_dataloader) // 100), **trainer_params
        )
        trainer.fit(model, train_dataloader, val_dataloader)

    elif mode == "test":
        assert "ckpt" in model_params

        test_dataset, test_dataloader = get_dataset(**dataset_params["test"])
        model = TrainingModel.load_from_checkpoint(model_params["ckpt"])
        trainer = pl.Trainer(
            log_every_n_steps=max(1, len(test_dataloader) // 100),
            checkpoint_callback=False,
            logger=False,
            **trainer_params,
        )
        trainer.validate(model, dataloaders=test_dataloader)

    elif mode == "predict":
        assert "ckpt" in model_params

        test_dataset, test_dataloader = get_dataset(**dataset_params["test"])
        model = TrainingModel.load_from_checkpoint(model_params["ckpt"])
        trainer = pl.Trainer(
            log_every_n_steps=max(1, len(test_dataloader) // 100),
            checkpoint_callback=False,
            logger=False,
            **trainer_params,
        )
        yhat = trainer.predict(model, dataloaders=test_dataloader)
        return yhat


def run(
    gpus=[3],
    lr=0.0002,
    num_layers=7,
    batch_size=1024,
    optimizer="adam",
    subsample_factor=12,
):
    pl.seed_everything(345)

    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    N = 34
    train_files = files[: -N * 2]
    val_files = files[-N * 2 : -N]
    test_files = files[-N:]

    datasets = []

    for f_list in [train_files, val_files]:

        dataset = NcIterableDataset(
            f_list,
            max_files_in_memory=1,
            batch_size=24,
            shuffle=False,
            flatten=True,  # False  -> use "globe" images
            inputs=inputs,
            outputs=outputs,
            subsample_factor=subsample_factor,
            compute_stats=True,
        )

        datasets.append(dataset.get_tensors(cache_dir="/ssddg1/gaia/cache"))

    train_dataset, val_dataset = datasets

    # train_dataset = make_dummy_dataset()
    # val_dataset = NcDatasetMem(test_files)

    model_config = {
        "model_type": "fcn",
        "input_size": train_dataset["x"].shape[-1],
        "output_size": train_dataset["y"].shape[-1],
        "num_layers": num_layers,
    }

    model = TrainingModel(
        lr=lr,
        optimizer=optimizer,
        model_config=model_config,
        input_index=train_dataset["input_index"],
        output_index=train_dataset["output_index"],
        data_stats=train_dataset["stats"],
        dataset_params=None,
    )

    train_data_loader = DataLoader(
        FastTensorDataset(
            train_dataset["x"], train_dataset["y"], batch_size=batch_size, shuffle=True
        ),
        batch_size=None,
        pin_memory=True,
    )
    val_data_loader = DataLoader(
        FastTensorDataset(
            val_dataset["x"], val_dataset["y"], batch_size=batch_size, shuffle=False
        ),
        batch_size=None,
        pin_memory=True,
    )

    trainer = pl.Trainer(
        gpus=gpus,
        # checkpoint_callback=False,
        # logger=False,
        precision=16,
        log_every_n_steps=max(1, len(train_data_loader) // 100),
    )

    trainer.fit(model, train_data_loader, val_data_loader)


def test(gpus=[1]):
    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    N = 34
    # train_files = files[: -N * 2]
    # val_files = files[-N * 2 : -N]
    test_files = files[-N:]

    dataset = NcIterableDataset(
        test_files,
        max_files_in_memory=1,
        batch_size=24,
        shuffle=False,
        flatten=False,  # False  -> use "globe" images
        inputs=inputs,
        outputs=outputs,
        subsample_factor=1,
        compute_stats=False,
    ).get_tensors(cache_dir="/ssddg1/gaia/cache")

    ##TODO auto flatten

    for v in ["x", "y"]:
        dataset[v] = dataset[v].permute([0, 2, 3, 1]).reshape(-1, dataset[v].shape[1])

    dataloader = DataLoader(
        FastTensorDataset(
            dataset["x"], dataset["y"], batch_size=24 * 96 * 144, shuffle=False
        ),
        batch_size=None,
        pin_memory=True,
    )

    model_config = {
        "model_type": "fcn",
        "input_size": 130,
        "output_size": 79,
        "num_layers": 7,
    }

    # path = "lightning_logs/version_2/checkpoints/epoch=224-step=141299.ckpt"
    path = "lightning_logs/version_6/checkpoints/epoch=999-step=1889999.ckpt"
    model = TrainingModel.load_from_checkpoint(
        path, model_config=model_config, data_stats=None
    )

    trainer = trainer = pl.Trainer(
        gpus=gpus,
        checkpoint_callback=False,
        logger=False,
        precision=16,
        log_every_n_steps=max(1, len(dataloader) // 100),
    )

    trainer.validate(model, dataloaders=dataloader)


def predict(gpus=[1]):
    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    N = 34
    # train_files = files[: -N * 2]
    # val_files = files[-N * 2 : -N]
    test_files = files[-N:]

    dataset = NcIterableDataset(
        test_files,
        max_files_in_memory=1,
        batch_size=24,
        shuffle=False,
        flatten=False,  # False  -> use "globe" images
        inputs=inputs,
        outputs=outputs,
        subsample_factor=1,
        compute_stats=False,
    ).get_tensors(cache_dir="/ssddg1/gaia/cache")

    dataloader = DataLoader(
        FastTensorDataset(dataset["x"], dataset["y"], batch_size=24, shuffle=False),
        batch_size=None,
        pin_memory=True,
    )

    model_config = {
        "model_type": "conv",
        "input_size": 130,
        "output_size": 79,
        "num_layers": 7,
    }

    path = "/proj/gaia-climate/team/kirill/gaia-surrogate/lightning_logs/version_2/checkpoints/epoch=224-step=141299.ckpt"
    model = TrainingModel.load_from_checkpoint(path, model_config=model_config)

    trainer = trainer = pl.Trainer(
        gpus=gpus,
        checkpoint_callback=False,
        logger=False,
        precision=16,
        log_every_n_steps=max(1, len(dataloader) // 100),
    )

    # yhat = trainer.predict(model,dataloaders=dataloader)
    # yhat = torch.cat(yhat)

    # torch.save(yhat, "predictions.pt")

    trainer.validate(model, dataloaders=dataloader)


def run_old(flatten=False, gpus=[3], lr=0.0002, num_layers=7):
    raise DeprecationWarning

    pl.seed_everything(345)

    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    train_files = files[:-34]
    val_files = files[-34:]

    train_dataset = NcIterableDataset(
        train_files,
        max_files_in_memory=50,
        batch_size=24 if not flatten else 2048,
        shuffle=True,
        flatten=flatten,  # False  -> use "globe" images
        inputs=inputs,
        outputs=outputs,
    )

    val_dataset = NcIterableDataset(
        val_files,
        max_files_in_memory=len(val_files),
        batch_size=2 * 48 if not flatten else 2 * 48 * 96 * 144,
        shuffle=False,
        flatten=flatten,  # False  -> use "globe" images
        inputs=inputs,
        outputs=outputs,
        keep_in_memory=True,
    )

    # train_dataset = make_dummy_dataset()
    # val_dataset = NcDatasetMem(test_files)

    model_config = {
        "model_type": "conv" if not flatten else "fcn",
        "input_size": train_dataset.input_size,
        "output_size": train_dataset.output_size,
        "num_layers": num_layers,
    }

    model = TrainingModel(
        lr=lr,
        model_config=model_config,
        input_index=train_dataset.input_index,
        output_index=train_dataset.output_index,
        data_stats="/proj/gaia-climate/team/kirill/gaia-surrogate/stats.pt",
    )

    train_data_loader = DataLoader(train_dataset, batch_size=None, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=None, pin_memory=True)

    trainer = pl.Trainer(
        gpus=gpus,
        # checkpoint_callback=False,
        # logger=False,
        precision=16,
        log_every_n_steps=len(train_dataset) // 100,
    )

    trainer.fit(model, train_data_loader, val_data_loader)
    # trainer.validate(model,val_data_loader)
