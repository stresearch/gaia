from math import prod
from random import shuffle
import shutil
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
import boto3
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
import glob
from gaia import get_logger
from netCDF4 import Dataset as netCDF4_Dataset
from tqdm import auto as tqdm
import numpy as np
import os
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = get_logger(__name__)


# inputs = ["T", "Q", "RELHUM", "U", "V"]
# outputs = ["PTEQ", "PTTEND", "PRECT", "TTEND_TOT"]
# outputs = ["PTEQ", "PTTEND", "PRECT"]


# from here https://arxiv.org/pdf/2010.12996.pdf
SCALING_FACTORS = {"PRECT": 1728000.0, "PTTEND": 1.00464e3, "PTEQ": 2.501e6 + 3.337e5}


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
            ## lets check if data exists with no subsampling

            data = self.load(cache_dir)
            if data is not None:
                return data
            else:
                if self.subsample_factor != 1:
                    logger.info(
                        f"subsample factor = {self.subsample_factor}, checking to see if data exists with subsample = 1"
                    )
                    og_subsample_factor = self.subsample_factor
                    self.subsample_factor = 1
                    data = self.load(cache_dir)
                    if data is not None:
                        logger.info(
                            f"cache exists, subsampling to og factor {og_subsample_factor}"
                        )
                        x, y, index = self.subsample_data(
                            data["x"], data["y"], og_subsample_factor
                        )
                        data["x"] = x
                        data["y"] = y
                        data["index"] = index
                        return data
                    else:
                        self.subsample_factor = og_subsample_factor
                        logger.info(
                            f"nope... need to load data and subsample by {self.subsample_factor}"
                        )

            data = self._get_tensors()
            self.save(data, cache_dir)
            return data
        else:
            return self._get_tensors()

    def subsample_data(self, xi, yi, subsample_factor):
        size = xi.shape[0]
        new_size = size // subsample_factor
        shuffled_index = torch.randperm(size)[:new_size]
        xi = xi[shuffled_index, ...]
        yi = yi[shuffled_index, ...]
        return xi, yi, shuffled_index

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
                input_stats=self.get_stats(x), output_stats=self.get_stats(y)
            )

        return out

    @staticmethod
    def get_stats(x):
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


def flatten_tensor(v):
    # return x.permute([0, 2, 3, 1]).reshape(-1, x.shape[1])

    if len(v.shape) == 4:
        num_samples, num_channels, num_lons, num_lats = v.shape
        time_steps = 1
    elif len(v.shape) == 5:
        num_samples, time_steps, num_channels, num_lons, num_lats = v.shape

    else:
        raise ValueError(f"shape {v.shape} not supported")

    if time_steps > 1:
        v = v.permute([0, 3, 4, 1, 2]).reshape(
                -1, time_steps, num_channels
            )
    else:
        v = v.permute([0, 2, 3, 1]).reshape(-1, num_channels)

    return v


def unflatten_tensor(v):
    if len(v.shape) == 2:
        num_samples, num_channels = v.shape
        time_steps = 1
    elif len(v.shape) == 3:
        num_samples, time_steps, num_channels = v.shape
    else:
        raise ValueError(f"shape {v.shape} not supported")

    if time_steps > 1:
        v = v.reshape(-1, 96, 144, time_steps, num_channels).permute([0, 3, 4,  1, 2])
    else:

         v = v.reshape(-1, 96, 144, num_channels).permute([0, 3, 1, 2])

    return v


def get_variable_index(dataset, variable_names, channel_dim = 1):
    out = OrderedDict()
    i = 0
    
    for n in variable_names:
        shape = dataset[n].shape
        if len(shape) < 4:
            num_channels = 1
        elif len(shape) == 4:
            num_channels = shape[channel_dim]
        else:
            raise ValueError("all variables must have at least 3 dims")
        j = i + num_channels
        out[n] = [i, j]
        i = j

    return out

class NCDataConstructor:
    def __init__(
        self,
        inputs: list = ["T", "Q"],
        outputs: list = ["PTEQ", "PTTEND"],
        shuffle=True,
        channel_dim=1,
        space_dims=[2, 3],
        time_dim=0,
        flatten=True,
        subsample_factor=1,
        compute_stats=True,
        cache=".",
        s3_client_kwargs=None,
        time_steps=1,
    ):

        self.flatten = flatten
        self.shuffle = shuffle
        self.compute_stats = compute_stats

        self.channel_dim = channel_dim
        self.time_dim = time_dim
        self.space_dims = space_dims
        self.subsample_factor = subsample_factor
        self.cache = cache
        self.s3_client_kwargs = s3_client_kwargs
        self.time_steps = time_steps

        if self.s3_client_kwargs is not None:
            self.file_location = "s3"

        self.inputs = inputs
        self.outputs = outputs

        assert len(inputs) > 0
        assert len(outputs) > 0

        self.input_index = None
        self.output_index = None

    @classmethod
    def default_data(
        cls,
        split="train",
        bucket_name="ff350d3a-89fc-11ec-a398-ac1f6baca408",
        prefix="spcamclbm-nx-16-20m-timestep",
        save_location="/ssddg1/gaia/spcam",
        train_years = 7,
        subsample_factor = 4,
        cache = ".",
        workers = 1, 
        inputs = "Q,T,U,V,OMEGA,PSL,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,Z3".split(
                ","
            ),
        outputs = "PRECT,PRECC,PTEQ,PTTEND".split(","),
        time_steps = 0
    ):
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

        s3 = boto3.resource(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        bucket = s3.Bucket(bucket_name)
        files = sorted(
            [f"{bucket_name}/{f.key}" for f in bucket.objects.iterator(Prefix=prefix)]
        )

        logger.info(f"found {len(files)} files")

        if split == "test":
            start_index = train_years*365
            end_index = (1+train_years)*365
            files = files[start_index:end_index]
        else:
            start_index = 0
            end_index = train_years*365
            files = files[start_index:end_index]

        s3_client_kwargs = dict(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        data_constructor = cls(
            inputs=inputs,
            outputs=outputs,
            flatten = split == "train",
            shuffle = split == "train",
            subsample_factor=subsample_factor,
            compute_stats=True,
            cache = os.path.join(cache,split),
            s3_client_kwargs=s3_client_kwargs,
            time_steps=time_steps,
        )


        dataset_name = files[0].split("/")[-2]

        # out = data_constructor.load_files(files, save_file=None)


        out = data_constructor.load_files_parallel(
            files, num_workers=workers, save_file=None
        )

        if split == "train":

            # lets make dedicated train and val so that we dont have to worry about it anymore
            x = out.pop("x")
            y = out.pop("y")
            index = out.pop("index")

            mask = torch.rand(x.shape[0]) > 0.1  # .9 train

            xtrain = x[mask, ...]
            ytrain = y[mask, ...]
            index_train = index[mask]

            out["x"] = xtrain
            out["y"] = ytrain
            out["index"] = index_train

            torch.save(
                out,
                os.path.join(
                    save_location,
                    f"{dataset_name}_{data_constructor.subsample_factor}_train.pt",
                ),
            )

            xval = x[~mask, ...]
            yval = y[~mask, ...]
            index_val = index[~mask]

            out["x"] = xval
            out["y"] = yval
            out["index"] = index_val

            torch.save(
                out,
                os.path.join(
                    save_location,
                    f"{dataset_name}_{data_constructor.subsample_factor}_val.pt",
                ),
            )

        else:
            torch.save(
                out,
                os.path.join(
                    save_location,
                    f"{dataset_name}_{data_constructor.subsample_factor}_test.pt",
                ),
            )

    def get_input_index(self, dataset):
        if self.input_index is None:
            self.input_index = self.get_variable_index(dataset, self.inputs)

    def get_output_index(self, dataset):
        if self.output_index is None:
            self.output_index = self.get_variable_index(dataset, self.outputs)

    def get_variable_index(self, dataset, variable_names):
        out = OrderedDict()
        i = 0
        for n in variable_names:
            shape = dataset[n].shape
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

        # num_channels = v.shape[self.channel_dim]

        # if self.flatten:
        #     v = v.permute([0, 2, 3, 1]).reshape(-1, num_channels)
        return v

    def load_variables(self, names, dataset):
        v = torch.cat(
            [self.load_variable(n, dataset) for n in names],
            dim=self.channel_dim,
        )

        num_samples, num_channels, num_lons, num_lats = v.shape

        if self.time_steps > 1:
            v = v.reshape(-1, self.time_steps, num_channels, num_lons, num_lats)

            if self.flatten:
                v = v.permute([0, 3, 4, 1, 2]).reshape(
                    -1, self.time_steps, num_channels
                )
        else:
            if self.flatten:
                v = v.permute([0, 2, 3, 1]).reshape(-1, num_channels)

        return v

    def read_data(self, file):
        if self.file_location == "s3":
            return self.read_s3_file(file)

        else:
            return self.read_disk_file(file)

    def read_s3_file(self, file):
        temp = file.split("/")

        bucket_name = temp[0]
        file_name = temp[-1]
        object_key = "/".join(temp[1:])
        local_file = os.path.join(self.cache, file_name)
        if os.path.exists(local_file):
            # logger.info(f"local file {local_file} exists")
            return self.read_disk_file(local_file)
        else:
            # logger.info(f"downloading {local_file}")
            s3_client = boto3.client("s3", **self.s3_client_kwargs)
            s3_client.download_file(
                Bucket=bucket_name, Key=object_key, Filename=local_file
            )
            return self.read_disk_file(local_file)

    def read_disk_file(self, file):
        return netCDF4_Dataset(file, "r", format="NETCDF4")

    def load_file(self, file, cache_file = None):

        try:
            dataset = self.read_data(file)
            x = self.load_variables(self.inputs, dataset)
            y = self.load_variables(self.outputs, dataset)

            # will only
            self.get_input_index(dataset)
            self.get_output_index(dataset)

            if self.subsample_factor > 1:
                x, y, new_index = self.subsample_data(
                    x, y, subsample_factor=self.subsample_factor
                )

            #
            self.clean_up_file(dataset)

            if cache_file is not None:
                torch.save([x, y, new_index], cache_file)
                return cache_file

            return x, y, new_index


        except Exception as e:
            logger.exception(e)
            logger.warning(f"failed {file}")
            return

    def clean_up_file(self, dataset):
        temp_file = dataset.filepath()
        dataset.close()
        # logger.info(f"removing temp file {temp_file}")
        os.remove(temp_file)

    def subsample_data(self, xi, yi, subsample_factor):
        size = xi.shape[0]
        new_size = size // subsample_factor
        if self.shuffle:
            shuffled_index = torch.randperm(size)[:new_size]
        else:
            shuffled_index = torch.arange(0, size, subsample_factor)

        xi = xi[shuffled_index, ...]
        yi = yi[shuffled_index, ...]
        return xi, yi, shuffled_index

    def load_files(self, files, save_file=None):

        

        x = []
        y = []
        index = []
        for file in tqdm.tqdm(files):
            try:
                xi, yi, indexi = self.load_file(file)
            except Exception as e:
                logger.exception(e)
                logger.warning(f"failed {file}")

            x.append(xi)
            y.append(yi)
            index.append(indexi)

        x = torch.cat(x)
        y = torch.cat(y)
        index = torch.cat(index)

        out = dict(
            x=x,
            y=y,
            files=files,
            index=index,
            subsample_factor=self.subsample_factor,
            input_index=self.input_index,
            output_index=self.output_index,
        )

        if self.compute_stats:
            out["stats"] = dict(
                input_stats=self.get_stats(x), output_stats=self.get_stats(y)
            )

        if save_file is not None:
            torch.save(out, save_file)

        return out
        

    def load_files_parallel(self, files, num_workers=8, save_file=None):
        x = []
        y = [] 
        index = []

        # def load_file_wrapper(*args,**kwargs):
        #     try:
        #         return self.load_file(*args,**kwargs)
        #     except Exception as e:
        #         logger.exception(e)
        #         logger.warning(f"failed {args}, {kwargs}")
        #         return

        logger.info("delete cache files if any")

        os.makedirs(self.cache, exist_ok=True)

        for f in tqdm.tqdm(glob.glob(os.path.join(self.cache,"*"))):
            os.remove(f)

        logger.info("downloading files")

        with ProcessPoolExecutor(max_workers=num_workers) as exec:

           

            cache_files = [os.path.join(self.cache, f"{i:06}_cache.pt") for i in range(len(files))]

            futs  = []

            for f,cf in zip(files,cache_files):
                if os.path.exists(cf):
                    #skip and continue
                    continue
                fut = exec.submit(self.load_file,f,cf)
                futs.append(fut)

            for fut in tqdm.tqdm(as_completed(futs),total = len(files)):
                try:
                    fut.result()
                except Exception as e:
                    logger.exception(e)

            
        logger.info("merging files")

        for f in tqdm.tqdm(sorted(cache_files)):
            if os.path.exists(f):
                try:
                    xi,yi,indexi = torch.load(f)
                except Exception as e:
                    logger.exception(e)
                    logger.warning(f"failed {f}")
                    continue
            else:
                logger.warning(f"no file {f}")
                continue
            x.append(xi)
            y.append(yi)
            index.append(indexi)
            

        x = torch.cat(x)
        y = torch.cat(y)
        index = torch.cat(index)

        out = dict(
            x=x,
            y=y,
            files=files,
            index=index,
            subsample_factor=self.subsample_factor,
            input_index=self.input_index,
            output_index=self.output_index,
        )

        if self.compute_stats:
            out["stats"] = dict(
                input_stats=self.get_stats(x), output_stats=self.get_stats(y)
            )

        if save_file is not None:
            torch.save(out, save_file)

        logger.info("erasing temp files")
        for f in tqdm.tqdm(sorted(cache_files)):
            if os.path.exists(f):
                os.remove(f)
                
        return out

    def get_stats(self,x):
        logger.info(f"computing stats for tensor of shape {x.shape}")
        outs = dict()

        channel_dim = 1

        if self.time_steps > 1:
            channel_dim = 2

        reduce_dims = [i for i in range(len(x.shape)) if i != channel_dim]
        

        outs["mean"] = x.mean(dim=reduce_dims)
        outs["std"] = x.std(dim=reduce_dims)
        outs["min"] = x.amin(dim=reduce_dims)
        outs["max"] = x.amax(dim=reduce_dims)
        return outs


def get_dataset_v1(
    files=None,
    subsample_factor=12,
    batch_size=1024,
    shuffle=False,
    in_memory=True,
    flatten=True,
    compute_stats=True,
    flatten_anyway=False,
    inputs=None,
    outputs=None,
    split_fraction=None,
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
        compute_stats=compute_stats,
    ).get_tensors(cache_dir="/ssddg1/gaia/cache")

    del dataset_dict["index"]

    if flatten_anyway:
        logger.warning("flattening dataset")
        for v in ["x", "y"]:
            dataset_dict[v] = (
                dataset_dict[v]
                .permute([0, 2, 3, 1])
                .reshape(-1, dataset_dict[v].shape[1])
            )

    if split_fraction is not None:
        logger.info("making val set by splitting train set in a truly random fashion")
        mask_train = torch.rand(dataset_dict["x"].shape[0]) >= split_fraction

        data_loader_train = DataLoader(
            FastTensorDataset(
                dataset_dict["x"][mask_train],
                dataset_dict["y"][mask_train],
                batch_size=batch_size,
                shuffle=shuffle,
            ),
            batch_size=None,
            pin_memory=True,
        )

        data_loader_test = DataLoader(
            FastTensorDataset(
                dataset_dict["x"][~mask_train],
                dataset_dict["y"][~mask_train],
                batch_size=batch_size,
                shuffle=False,
            ),
            batch_size=None,
            pin_memory=True,
        )

        return dataset_dict, data_loader_train, data_loader_test

    data_loader = DataLoader(
        FastTensorDataset(
            dataset_dict["x"], dataset_dict["y"], batch_size=batch_size, shuffle=shuffle
        ),
        batch_size=None,
        pin_memory=True,
    )

    return dataset_dict, data_loader


def get_dataset(
    dataset_file,
    batch_size=1024,
    flatten=False,
    shuffle = False,
    var_index_file = None
):

    dataset_dict = torch.load(dataset_file)

    # var_index = torch.load("/ssddg1/gaia/spcam/var_index.pt")
    var_index = torch.load(var_index_file)

    dataset_dict.update(var_index)

    del dataset_dict["index"]

    if flatten:
        logger.warning("flattening dataset")
        for v in ["x", "y"]:
            dataset_dict[v] = flatten_tensor(dataset_dict[v])

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





