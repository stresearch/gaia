from asyncio import base_tasks
from math import prod
from random import shuffle
from re import L
from typing import Union
from collections import OrderedDict
from sklearn.neighbors import VALID_METRICS
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
    Dataset,
    IterableDataset,
)
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
import glob
from gaia import get_logger
from netCDF4 import Dataset as netCDF4_Dataset
from tqdm import auto as tqdm
import numpy as np
import os
import json
import hashlib

logger = get_logger(__name__)


inputs = ["T", "Q", "RELHUM", "U", "V"]
outputs = ["PTEQ", "PTTEND", "PRECT", "TTEND_TOT"]

def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


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
        keep_in_memory=False,
        subsample_factor=1,
        compute_stats=True,
    ):

        self.flatten = flatten
        self.max_files_in_memory = max_files_in_memory
        self.shuffle = shuffle
        self.compute_stats = compute_stats

        self.channel_dim = channel_dim
        self.time_dim = time_dim
        self.space_dims = space_dims
        self.subsample_factor = subsample_factor

        self.batch_size = batch_size

        if isinstance(files_or_pattern, str):
            self.files = sorted(glob.glob(files_or_pattern))
        elif isinstance(files_or_pattern, list):
            self.files = sorted(files_or_pattern)
        else:
            raise ValueError("unsupported files_or_pattern")

        self.keep_in_memory = (
            keep_in_memory & (len(self.files) <= max_files_in_memory) & (not shuffle)
        )
        if self.keep_in_memory:
            logger.info(f"keeping entire dataset of {len(self.files)} files in memory")
        self.cache = dict()

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
        self.input_size = list(self.input_index.values())[-1][-1]
        self.output_size = list(self.output_index.values())[-1][-1]

    def __len__(self):
        return self.size

    def get_dimension(self, name):
        return netCDF4_Dataset(self.files[0], "r", format="NETCDF4")[name].shape

    def get_variable_index(self, variable_names):
        out = OrderedDict()
        i = 0
        for n in variable_names:
            shape = self.get_dimension(n)
            if len(shape) < 4:
                num_channels = 1
            elif len(shape) == 4:
                num_channels = shape[self.channel_dim]
            else:
                raise ValueError("all variables must have at least 3 dims")
            j = i + num_channels
            out[n] = [i, j]
            i = j

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

    def store_in_cache(self, files, data):
        key = " ".join(files)
        self.cache[key] = data

    def get_from_cache(self, files):
        key = " ".join(files)
        return self.cache.get(key)

    def load_files(self, files):

        if self.keep_in_memory:
            out = self.get_from_cache(files)
            if out is not None:
                x, y = out
                return x, y

        x = []
        y = []
        logger.info(f"loading {len(files)} files")
        for file in tqdm.tqdm(files, leave=False):
            xi, yi = self.load_file(file)
            x.append(xi)
            y.append(yi)

        out = torch.cat(x), torch.cat(y)

        if self.keep_in_memory:
            self.store_in_cache(files, out)

        return out

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

    def data_params(self):
        return dict(
            files=self.files,
            subsample_factor=self.subsample_factor,
            inputs=self.inputs,
            outputs=self.outputs,
        )

    def get_hash(self):
        return dict_hash(self.data_params()) + ".pt"

    def save(self, data, cache_dir):
        file_name = os.path.join(cache_dir, self.get_hash())
        logger.info(f"saving cached data {file_name}")
        torch.save(data, file_name)

    def load(self, cache_dir):
        file_name = os.path.join(cache_dir, self.get_hash())
        if os.path.exists(file_name):
            logger.info(f"loading cached data {file_name}")
            return torch.load(file_name)

    def get_tensors(self, cache_dir=None):
        if cache_dir is not None:
            data = self.load(cache_dir)
            if data is not None:
                return data
            data = self._get_tensors()
            self.save(data, cache_dir)
            return data
        else:
            return self._get_tensors()

    def _get_tensors(self):
        subsample_factor = self.subsample_factor
        x = []
        y = []
        index = []
        for file in tqdm.tqdm(self.files):
            xi, yi = self.load_file(file)
            if subsample_factor > 1:
                size = xi.shape[0]
                new_size = size // subsample_factor
                shuffled_index = torch.randperm(size)[:new_size]
                xi = xi[shuffled_index, ...]
                yi = yi[shuffled_index, ...]
                index.append(shuffled_index[None, :])
            x.append(xi)
            y.append(yi)

        x = torch.cat(x)
        y = torch.cat(y)
        if len(index) > 0:
            index = torch.cat(index)
        else:
            index = None

        out = dict(
            x=x,
            y=y,
            files=self.files,
            index=index,
            subsample_factor=subsample_factor,
            input_index=self.input_index,
            output_index=self.output_index,
        )

        if self.compute_stats:
            out["stats"] = dict(
                input_stats=self.compute_stats(x), output_stats=self.compute_stats(y)
            )

        return out

    @staticmethod
    def compute_stats(x):
        logger.info(f"computing stats for tensor of shape {x.shape}")
        outs = dict()
        if len(x.shape) == 4:
            reduce_dims = [0, 2, 3]
        elif len(x.shape) == 2:
            reduce_dims = [0]
        else:
            raise ValueError("only 2D or 4D shapes supported")

        outs["mean"] = x.mean(dim=reduce_dims)
        outs["std"] = x.std(dim=reduce_dims)
        outs["min"] = x.amin(dim=reduce_dims)
        outs["max"] = x.amax(dim=reduce_dims)
        return outs


def get_dataset(
    files=None,
    subsample_factor=12,
    batch_size=1024,
    shuffle=False,
    in_memory=True,
    flatten=True,
):

    

    if not in_memory:
        raise ValueError

    dataset_dict = NcIterableDataset(
        files,
        max_files_in_memory=1,
        batch_size=24,
        shuffle=False,
        flatten=flatten,  # False  -> use "globe" images
        inputs=inputs,
        outputs=outputs,
        subsample_factor=subsample_factor,
        compute_stats=True,
    ).get_tensors(cache_dir="/ssddg1/gaia/cache")

    data_loader = DataLoader(
        FastTensorDataset(
            dataset_dict["x"], dataset_dict["y"], batch_size=batch_size, shuffle=shuffle
        ),
        batch_size=None,
        pin_memory=True,
    )

    return dataset_dict, data_loader

class FastTensorDataset(IterableDataset):
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class FastTensorDataset2(IterableDataset):
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        self.r = torch.arange(self.dataset_len)

    def __iter__(self):
        if self.shuffle:
            self.r = torch.randperm(self.dataset_len)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch_index = self.r[self.i : self.i + self.batch_size]
        batch = tuple(t[batch_index] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


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
