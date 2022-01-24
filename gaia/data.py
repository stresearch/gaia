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
        channel_dim=1,
        space_dims=[2, 3],
        time_dim=0,
        flatten=True,
    ):

        self.flatten = flatten
        self.max_files_in_memory = max_files_in_memory
        self.shuffle = shuffle

        self.channel_dim = channel_dim
        self.time_dim = time_dim
        self.space_dims = space_dims

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

        self.dimensions = self.get_dimension(inputs[0])

        if self.flatten:
            self.size = (
                len(self.files)
                * self.dimensions[self.time_dim]
                * self.dimensions[self.space_dims[0]]
                * self.dimensions[self.space_dims[1]]
                // self.batch_size
            )
        else:
            self.size = (
                len(self.files) * self.dimensions[self.time_dim] // self.batch_size
            )


        self.input_index = self.get_variable_index(inputs)
        self.output_index = self.get_variable_index(outputs)


    def __len__(self):
        return self.size

    def get_dimension(self,name):
        return netCDF4_Dataset(self.files[0], "r", format="NETCDF4")[name].shape

    def get_variable_index(self,variable_names):
        out = OrderedDict()
        i = 0
        for n in variable_names:
            shape = self.get_dimension(n)
            if len(shape)<4:
                num_channels = 1
            elif len(shape) == 4:
                num_channels = shape[self.channel_dim]
            else:
                raise ValueError("all variables must have at least 3 dims")
            j = i + num_channels
            out[n] = [i,j]
            i  = j

        return out


            
    def load_variable(self, name, dataset):
        v = torch.from_numpy(np.asarray(dataset[name]))
        if len(v.shape) < 3:
            raise ValueError("variables must have at least 3 dimensions")

        if len(v.shape) == 3:  # scalar
            v = v[:, None, :, :]  # adding singleton dimension

        num_channels = v.shape[self.channel_dim]

        if self.flatten:
            v = v.permute([0, 2, 3, 1]).reshape(-1, num_channels)
        return v

    def load_file(self, file):
        dataset = netCDF4_Dataset(file, "r", format="NETCDF4")
        # TODO dont hard code
        x = torch.cat(
            [self.load_variable(n, dataset) for n in self.inputs],
            dim=1,
        )

        y = torch.cat(
            [self.load_variable(n, dataset) for n in self.outputs],
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

            for i in range(0, x.shape[0], self.batch_size):
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
        raise DeprecationWarning
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
