import glob
import hashlib
import json
import os
import shutil
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import prod
from random import shuffle
from typing import Union

import boto3
import numpy as np
import torch
from netCDF4 import Dataset as netCDF4_Dataset
from torch.utils.data import (DataLoader, Dataset, IterableDataset,
                              TensorDataset, random_split)
from torch.utils.data.sampler import (BatchSampler, RandomSampler,
                                      SequentialSampler)
from tqdm import auto as tqdm

from gaia import get_logger
from gaia.layers import InterpolateGrid1D

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
        v = v.permute([0, 3, 4, 1, 2]).reshape(-1, time_steps, num_channels)
    else:
        v = v.permute([0, 2, 3, 1]).reshape(-1, num_channels)

    return v


def unflatten_tensor(v, num_lons = 144, num_lats = 96 ):
    if len(v.shape) == 2:
        num_samples, num_channels = v.shape
        time_steps = 1
    elif len(v.shape) == 3:
        num_samples, time_steps, num_channels = v.shape
    else:
        raise ValueError(f"shape {v.shape} not supported")

    if time_steps > 1:
        v = v.reshape(-1, num_lats, num_lons, time_steps, num_channels).permute([0, 3, 4, 1, 2])
    else:

        v = v.reshape(-1, num_lats, num_lons, num_channels).permute([0, 3, 1, 2])

    return v


def get_variable_index(dataset, variable_names, channel_dim=1, return_dict=True):
    out = []
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
        out.append((n, [i, j]))
        i = j

    if return_dict:
        out = OrderedDict(out)

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

        if len(outputs) == 0:
            no_output = True
            logger.warning("no outputs will be constructed... adding a dummy output")
            outputs = ["PRECC"]

        data_constructor = cls(
            inputs=inputs,
            outputs=outputs,
            flatten = split == "train",
            shuffle = True, #split == "train",
            subsample_factor=subsample_factor,
            compute_stats=True,
            cache=os.path.join(cache, split),
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

            if not no_output:
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

            if not no_output:
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
            if no_output:
                out.pop("y")

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

    def load_file(self, file, cache_file=None):

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

        for f in tqdm.tqdm(glob.glob(os.path.join(self.cache, "*"))):
            os.remove(f)

        logger.info("downloading files")

        with ProcessPoolExecutor(max_workers=num_workers) as exec:


            cache_files = [os.path.join(self.cache, f"{i:06}_cache.pt") for i in range(len(files))]

            futs = []

            for f, cf in zip(files, cache_files):
                if os.path.exists(cf):
                    # skip and continue
                    continue
                fut = exec.submit(self.load_file, f, cf)
                futs.append(fut)

            for fut in tqdm.tqdm(as_completed(futs), total=len(files)):
                try:
                    fut.result()
                except Exception as e:
                    logger.exception(e)

        logger.info("merging files")

        for f in tqdm.tqdm(sorted(cache_files)):
            if os.path.exists(f):
                try:
                    xi, yi, indexi = torch.load(f)
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

    def get_stats(self, x):
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


def unravel_index(flat_index, shape):
    # flat_index = operator.index(flat_index)
    res = []

    # Short-circuits on zero dim tensors
    if shape == torch.Size([]):
        return 0

    for size in shape[::-1]:
        res.append(flat_index % size)
        flat_index = flat_index // size

    # return torch.cat(res

    if len(res) == 1:
        return res[0]

    return res[::-1]


def get_dataset(
    dataset_file,
    batch_size=1024,
    flatten=False,
    shuffle=False,
    var_index_file=None,
    include_index=False,
    subsample=1,
    space_filter=None,
    inputs=None,
    outputs=None,
    data_grid = None,
    model_grid = None,
    subsample_mode = "random",
    chunk_size = 0
):

    dataset_dict = torch.load(dataset_file)

    # var_index = torch.load("/ssddg1/gaia/spcam/var_index.pt")
    var_index = torch.load(var_index_file)

    # construct dataset from specified inputs 

    if (inputs is not None) or (outputs is not None):


        logger.info(f"constructing custom inputs from datasets: inputs: {inputs} outputs: {outputs}")


        def extract_time_index(names):
            if len(names[0].split(" "))==1:
                # no time index in names
                return names,None
            names,times = zip(*[n.split(" ") for n in names])
            times = [1 if "1" in t else 0 for t in times]
            return list(names),times


        if inputs is None:
            inputs = list(var_index["input_index"].keys())
            inputs_time_index = 0
        else:
            inputs, inputs_time_index = extract_time_index(inputs)
            if inputs_time_index is None:
                logger.info("no time index info in input spec, assuming index 0")
                inputs_time_index = 0


        if outputs is None:
            outputs = list(var_index["output_index"].keys())
            outputs_time_index = 1
        else:
            outputs, outputs_time_index = extract_time_index(outputs)
            if outputs_time_index is None:
                logger.info("no time index info in input spec, assuming index 1")
                outputs_time_index = 1


        assert len(dataset_dict["x"].shape) in [3, 5]

        common_index = var_index["input_index"]
        common_stats = dataset_dict["stats"]["input_stats"]
        common_data = dataset_dict["x"]

        if "y" in dataset_dict:
            logger.info("found y... merging with x")
            channel_dim = 2
            D = common_data.shape[channel_dim]

            common_data = torch.cat([common_data, dataset_dict["y"]], dim=channel_dim)

            for k, v in var_index["output_index"].items():
                s, e = v
                common_index[k] = [s + D, e + D]

            for k, v in dataset_dict["stats"]["output_stats"].items():
                common_stats[k] = torch.cat([common_stats[k], v])

        def _make_one(names, time_index):
            stats = defaultdict(list)
            index = OrderedDict()
            data = []
            current_index = 0

            if not isinstance(time_index,list):
                logger.info(f"time_index is not a list.... assuming all variabes will be selected at the same time index {time_index}")
                time_index = [time_index]*len(names)

            for n,t in zip(names,time_index) :
                s, e = common_index[n]
                d = e - s
                data.append(common_data[:, t, s:e, ...])

                for k, v in common_stats.items():
                    stats[k].append(v[s:e, ...])

                index[n] = [current_index, current_index + d]
                current_index = current_index + d

            data = torch.cat(data, dim=1)

            for k in list(stats.keys()):
                stats[k] = torch.cat(stats[k])

            return data, dict(stats), index

        logger.info("creating input")

        d, s, i = _make_one(inputs, inputs_time_index)
        dataset_dict["x"] = d
        dataset_dict["input_index"] = i
        dataset_dict["stats"]["input_stats"] = s

        logger.info("creating output")

        d, s, i = _make_one(outputs, outputs_time_index)
        dataset_dict["y"] = d
        dataset_dict["output_index"] = i
        dataset_dict["stats"]["output_stats"] = s

    else:

        logger.info("assuming default inputs")

        dataset_dict.update(var_index)

        assert len(dataset_dict["x"].shape) in [3, 5]

        logger.warn("inputs are time step 0 and outputs are at timestep 1")

        dataset_dict["x"] = dataset_dict["x"][:, 0, ...]
        dataset_dict["y"] = dataset_dict["y"][:, 1, ...]

    if flatten:
        logger.warning("flattening dataset")
        for v in ["x", "y"]:
            dataset_dict[v] = flatten_tensor(dataset_dict[v])


    # do some grid interpolation

    if (data_grid is not None) and (model_grid is not None) and (data_grid != model_grid):

        logger.info(f"model grid is not equal to data grid.. need to interpolate from {len(data_grid)} grids to {len(model_grid)}")
        input_interpolation = InterpolateGrid1D(input_grid=data_grid, output_grid=model_grid, input_grid_index=dataset_dict["input_index"])
        dataset_dict["x"] = input_interpolation(dataset_dict["x"])

        for k in dataset_dict["stats"]["input_stats"].keys():
            dataset_dict["stats"]["input_stats"][k] = input_interpolation(dataset_dict["stats"]["input_stats"][k])

        dataset_dict["input_index"] = input_interpolation.output_grid_index

        output_interpolation = InterpolateGrid1D(input_grid=data_grid, output_grid=model_grid, input_grid_index=dataset_dict["output_index"])
        dataset_dict["y"] = output_interpolation(dataset_dict["y"])

        for k in dataset_dict["stats"]["output_stats"].keys():
            dataset_dict["stats"]["output_stats"][k] = output_interpolation(dataset_dict["stats"]["output_stats"][k])

        dataset_dict["output_index"] = output_interpolation.output_grid_index

    tensor_list = [dataset_dict["x"], dataset_dict["y"]]

    if include_index or (space_filter is not None) or subsample_mode != "random":
        # TODO dont hard code this
        logger.warning("THIS IS HARDCODED num_ts, num_lats, num_lons = 8, 96, 144")
        num_ts, num_lats, num_lons = 8, 96, 144
        logger.warning(
            f"using hardcoded expected shape for unraveling the index: {num_ts,num_lats,num_lons}"
        )

        if flatten:

            # index is flattened
            # shape = dataset_dict["x"].shape
            # if len(dataset_dict["x"].shape) == 3:
            #     samples, timesteps, channels = dataset_dict["x"].shape
            # else:
            #     samples, channels = dataset_dict["x"].shape

            num_samples = dataset_dict["index"].shape[0]

            lats = (
                torch.ones(num_samples, 1, num_lons)
                * torch.arange(num_lats)[None, :, None]
            )
            lons = (
                torch.ones(num_samples, num_lats, 1)
                * torch.arange(num_lons)[None, None, :]
            )
            index = torch.cat(
                [lats.ravel().long()[:, None], lons.ravel().long()[:, None]], dim=-1
            )
        else:
            index = unravel_index(
                dataset_dict["index"], shape=[num_ts, num_lats, num_lons]
            )
            index = torch.cat(
                [i[:, None] for i in index[1:]], dim=-1
            )  # just want lats, lons

    else:
        del dataset_dict["index"]

    if include_index:
        tensor_list += [index]

    if space_filter is not None:
        # filter out dataset

        logger.info(f"applying space filter {space_filter}")

        from gaia.plot import lats as lats_vals
        from gaia.plot import lons as lons_vals

        lats_vals = torch.tensor(lats_vals)
        lons_vals = torch.tensor(lons_vals)

        mask = torch.ones(len(tensor_list[0])).bool()

        if "lat_bounds" in space_filter:
            lat_min, lat_max = space_filter["lat_bounds"]
            temp = lats_vals[index[:, 0]]
            mask = mask & (temp <= lat_max) & (temp >= lat_min)

        if "lon_bounds" in space_filter:
            lon_min, lon_max = space_filter["lon_bounds"]
            temp = lons_vals[index[:, 1]]
            mask = mask & (temp <= lon_max) & (temp >= lon_min)

        assert mask.any()

        tensor_list = [t[mask, ...] for t in tensor_list]

    if subsample > 1:
        if subsample_mode == "random":
            tensor_list = [t[::subsample, ...] for t in tensor_list]
            logger.info(f"subsampling by factor of {subsample}")

        elif isinstance(subsample_mode, torch.Tensor):
            logger.info(f"using index to subsample")
            sample_index = subsample_mode
            tensor_list = [t[sample_index, ...] for t in tensor_list]

        else:
            logger.info(f"using weighted subsample mode from file :{subsample_mode}")
            lat_lon_weights = torch.load(subsample_mode)
            sample_weights = lat_lon_weights[index[:,0], index[:,1]]
            number_of_samples = tensor_list[0].shape[0]//subsample
            # lat_sample_index, lon_sample_index =  unravel_index(number_of_samples, shape = lat_lon_weights.shape)

            lat_lon_weights_sorted, sorted_index = sample_weights.ravel().sort(descending = True)
            lat_lon_weights_sorted /= lat_lon_weights_sorted.sum()
            lat_lon_weights_sorted_cumsum = lat_lon_weights_sorted.cumsum(0)

            sample_index = torch.searchsorted(lat_lon_weights_sorted_cumsum, torch.rand(number_of_samples))
            sample_index = sorted_index[sample_index]

            dataset_dict["sample_index"] = sample_index

            tensor_list = [t[sample_index, ...] for t in tensor_list]

            
            

    logger.info(f"data size {len(tensor_list[0])}")

    data_loader = DataLoader(
        FastTensorDataset(*tensor_list, batch_size=batch_size, shuffle=shuffle, chunk_size = chunk_size if shuffle else 0),
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

    def __init__(self, *tensors, batch_size=32, shuffle=False, chunk_size = 0):
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

        self.chunk_size = chunk_size


        if self.chunk_size > 0:
            self.slice_size = batch_size // self.chunk_size
            self.slice_size += (self.slice_size // 10)
            logger.info(f"using pseudo random shuffing with chunksize = {self.slice_size}")
            slice_index = torch.arange(0,tensors[0].shape[0], self.slice_size).long()
            self.slices = [[t[s:s+self.slice_size] for s in slice_index] for t in self.tensors]

            self.num_slices = len(self.slices[0])
            # self.slices = torch.empty(self.num_slices,2).long()
            # self.slices[:,0] =  slices
            # self.slices[:-1,1] = slices[1:]
            # self.slices[-1,1] = tensors[0].shape[0]
            # self.slice_size = slice_size
        else:
            self.slice_size = 1

 
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches


    def shuffle_tensors(self):
        if self.chunk_size  == 0:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        else:
            slice_order = torch.randperm(self.num_slices)
            self.tensors = [torch.cat([s[i] for i in slice_order]) for s in self.slices]

    # def shuffle_chunks_per_tensor(self, t, chunk_order):
    #     return torch.cat([t[s:e] for s,e in self.slices[chunk_order,:]])

    def __iter__(self):
        if self.shuffle:
            self.shuffle_tensors()
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


def make_dummy_dataset():
    return TensorDataset(torch.randn(10000, 26 * 2), torch.randn(10000, 26 * 2))
