from gaia.models import Baseline
from gaia.data import  NcDatasetMem, make_dummy_dataset, NcIterableDataset
import glob
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def run():

    model = Baseline(input_size=26 * 2, output_size=26 * 2)

    files = sorted(glob.glob("/ssddg1/gaia/cesm106_cam4/*.nc"))
    # Ntrain = int(len(files) * 0.9)
    # train_files = files[:Ntrain]
    # test_files = files[Ntrain:]

    train_dataset = NcIterableDataset(files, max_files_in_memory=50, batch_size=512)
    # train_dataset = make_dummy_dataset()
    # val_dataset = NcDatasetMem(test_files)

    batch_size = 10000

    train_data_loader = DataLoader(
        train_dataset, batch_size=None,  pin_memory=True
    )
    # test_data_loader = DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    # )

    trainer = pl.Trainer(
        gpus=[2],
        checkpoint_callback=False,
        logger=False,
        precision=16,
    )

    trainer.fit(model, train_data_loader)
