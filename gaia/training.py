from collections import OrderedDict
from gaia.models import ComputeStats, TrainingModel
from gaia.data import NcDatasetMem, make_dummy_dataset, NcIterableDataset
import glob
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch


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


def run():

    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))[:-34]
    # Ntrain = int(len(files) * 0.9)
    # train_files = files[:Ntrain]
    # test_files = files[Ntrain:]

    flatten = False

    train_dataset = NcIterableDataset(
        files,
        max_files_in_memory=50,
        batch_size=24,
        shuffle=True,
        flatten=flatten, #False  -> use "globe" images 
        inputs=inputs,
        outputs=outputs,
    )
    # train_dataset = make_dummy_dataset()
    # val_dataset = NcDatasetMem(test_files)

    model_config = {"model_type": "conv",
                     "input_size" : list(train_dataset.input_index.values())[-1][-1],
                     "output_size": list(train_dataset.output_index.values())[-1][-1]}

    model = TrainingModel(
        model_config=model_config,
        input_index=train_dataset.input_index,
        output_index=train_dataset.output_index,
        data_stats="/proj/gaia-climate/team/kirill/gaia-surrogate/stats.pt",
    )

    train_data_loader = DataLoader(train_dataset, batch_size=None, pin_memory=True)
    # test_data_loader = DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    # )

    trainer = pl.Trainer(
        gpus=[2],
        # checkpoint_callback=False,
        # logger=False,
        precision=16,
    )

    trainer.fit(model, train_data_loader)
