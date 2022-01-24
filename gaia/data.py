from asyncio import base_tasks
from math import prod
from random import shuffle
from typing import Union
from collections import OrderedDict
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
    Dataset,
    IterableDataset,
)
import glob
from gaia import get_logger
from netCDF4 import Dataset as netCDF4_Dataset
from tqdm import auto as tqdm
import numpy as np

logger = get_logger(__name__)


class NcIterableDataset(IterableDataset):
    def __init__(
        self,
        files_or_pattern: Union[list, str] = None,
        inputs: list = ["T", "Q"],
        outputs: list = ["PTEQ", "PTTEND"],
        max_files_in_memory=50,
        shuffle=True,
        batch_size=10,
        var_dim=1,
    ):

        self.max_files_in_memory = max_files_in_memory
        self.shuffle = shuffle
        self.var_dim = var_dim
        self.batch_size = batch_size

        if isinstance(files_or_pattern, str):
            self.files = sorted(glob.glob(files_or_pattern))
        elif isinstance(files_or_pattern, list):
            self.files = files_or_pattern
        else:
            raise ValueError("unsupported files_or_pattern")

        self.inputs = inputs
        self.outputs = outputs

        assert len(self.files) > 0
        assert len(inputs) > 0
        assert len(outputs) > 0

        self.dimensions = self.get_dimension()

        self.size = len(self.files)*prod(self.dimensions)//self.dimensions[var_dim]//self.batch_size

    def __len__(self):
        return self.size

    def get_dimension(self):
        return netCDF4_Dataset(self.files[0], "r", format="NETCDF4")[self.inputs[0]].shape

    def load_file(self, file):
        dataset = netCDF4_Dataset(file, "r", format="NETCDF4")
        # TODO dont hard code
        x = torch.cat(
            [
                torch.from_numpy(np.asarray(dataset[n]))
                .permute([0, 2, 3, 1])
                .reshape(-1, 26)
                for n in self.inputs
            ],
            dim=1,
        )

        y = torch.cat(
            [
                torch.from_numpy(np.asarray(dataset[n]))
                .permute([0, 2, 3, 1])
                .reshape(-1, 26)
                for n in self.outputs
            ],
            dim=1,
        )

        return x, y

    def load_files(self, files):
        x = []
        y = []
        logger.info(f"loading {len(files)} files")
        for file in tqdm.tqdm(files, leave=False):
            xi, yi = self.load_file(file)
            x.append(xi)
            y.append(yi)

        return torch.cat(x), torch.cat(y)

    def get_epoch_data(self):
        files_shuffled = self.files.copy()
        if self.shuffle:
            shuffle(files_shuffled)

        for start_file_index in range(0, len(files_shuffled), self.max_files_in_memory):
            files_to_load = files_shuffled[
                start_file_index : start_file_index + self.max_files_in_memory
            ]
            x, y = self.load_files(files_to_load)
            
            if self.shuffle:
                shuffled_index = torch.randperm(x.shape[0])
            else:
                shuffled_index = torch.arange(x.shape[0])

            for i in range(0,x.shape[0],self.batch_size):
                start = i
                end = start + self.batch_size
                indeces = shuffled_index[start:end]
                yield x[indeces, ...], y[indeces, ...]

    def __iter__(self):
        return self.get_epoch_data()


class NcDatasetMem(Dataset):
    def __init__(
        self,
        files_or_pattern: Union[list, str] = None,
        inputs: list = ["T", "Q"],
        outputs: list = ["PTEQ", "PTTEND"],
        var_dim=1,
    ):
        if isinstance(files_or_pattern, str):
            self.files = sorted(glob.glob(files_or_pattern))
        elif isinstance(files_or_pattern, list):
            self.files = files_or_pattern
        else:
            raise ValueError("unsupported files_or_pattern")

        assert len(self.files) > 0
        assert len(inputs) > 0
        assert len(outputs) > 0

        logger.info(f"reading dataset from {len(self.files)} files")

        ## compute variable shapes

        self.inputs = inputs
        self.outputs = outputs

        self.x = []
        self.y = []

        for f in tqdm.tqdm(self.files):
            dataset = netCDF4_Dataset(f, "r", format="NETCDF4")
            # TODO dont hard code
            ins = torch.cat(
                [
                    torch.from_numpy(np.asarray(dataset[n]))
                    .permute([0, 2, 3, 1])
                    .reshape(-1, 26)
                    for n in inputs
                ],
                dim=1,
            )
            self.x.append(ins)
            outs = torch.cat(
                [
                    torch.from_numpy(np.asarray(dataset[n]))
                    .permute([0, 2, 3, 1])
                    .reshape(-1, 26)
                    for n in outputs
                ],
                dim=1,
            )
            self.y.append(outs)

        self.x = torch.cat(self.x)
        self.x = self.norm(self.x)

        self.y = torch.cat(self.y)
        self.y = self.norm(self.y)

    def norm(self, x):
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_dummy_dataset():
    return TensorDataset(torch.randn(10000, 26 * 2), torch.randn(10000, 26 * 2))
