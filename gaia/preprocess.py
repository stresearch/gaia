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
from tqdm import auto as tqdm

from gaia import get_logger

logger = get_logger(__name__)


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


class DataConstructor:
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
        else:
            self.file_location = "disk"

        self.inputs = inputs
        self.outputs = outputs

        assert len(inputs) > 0
        assert len(outputs) > 0

        self.input_index = None
        self.output_index = None

    @classmethod
    def preprocess_aws(
        cls,
        split="train",
        bucket_name="ff350d3a-89fc-11ec-a398-ac1f6baca408",
        prefix="spcamclbm-nx-16-20m-timestep",
        save_location="/ssddg1/gaia/spcam",
        train_years=7,
        subsample_factor=4,
        cache=".",
        workers=1,
        inputs="Q,T,U,V,OMEGA,PSL,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,Z3".split(","),
        outputs="PRECT,PRECC,PTEQ,PTTEND".split(","),
        time_steps=0,
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
            start_index = train_years * 365
            end_index = (1 + train_years) * 365
            files = files[start_index:end_index]
        else:
            start_index = 0
            end_index = train_years * 365
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
            flatten=split == "train",
            shuffle=True,  # split == "train",
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
            save_path_prefix = os.path.join(
                save_location,
                f"{dataset_name}_{data_constructor.subsample_factor}",
            )

            data_constructor.post_process_train(
                out=out, save_path_prefix=save_path_prefix, no_output=no_output
            )

            # lets make dedicated train and val so that we dont have to worry about it anymore

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

    @classmethod
    def preprocess_local_files(
        cls,
        split="train",
        dataset_name = None,
        files = None,
        save_location=None,
        train_years=3,
        subsample_factor=4,
        cache=".",
        workers=16,
        inputs="Q,T,U,V,OMEGA,PSL,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,Z3".split(","),
        outputs="PRECT,PRECC,PTEQ,PTTEND".split(","),
        time_steps=2,
    ):
        
        if len(outputs) == 0:
            no_output = True
            logger.warning("no outputs will be constructed... adding a dummy output")
            outputs = ["PRECC"]
        else:
            no_output = False

        
        files = sorted(files)

        logger.info(f"found {len(files)} files")

        if not os.path.exists(save_location):
            logger.error(f"save location {save_location} does not exist")
            return

        if split == "test":
            start_index = train_years * 365
            end_index = (1 + train_years) * 365
            files = files[start_index:end_index]
        else:
            start_index = 0
            end_index = train_years * 365
            files = files[start_index:end_index]


        if len(outputs) == 0:
            no_output = True
            logger.warning("no outputs will be constructed... adding a dummy output")
            outputs = ["PRECC"]

        data_constructor = cls(
            inputs=inputs,
            outputs=outputs,
            flatten=split == "train",
            shuffle=split == "train",
            subsample_factor=subsample_factor,
            compute_stats=True,
            cache=os.path.join(cache, split),
            s3_client_kwargs=None,
            time_steps=time_steps,
        )


        # out = data_constructor.load_files(files, save_file=None)

        out = data_constructor.load_files_parallel(
            files, num_workers=workers, save_file=None
        )

        if split == "train":
            save_path_prefix = os.path.join(
                save_location,
                f"{dataset_name}_{data_constructor.subsample_factor}",
            )

            data_constructor.post_process_train(
                out=out, save_path_prefix=save_path_prefix, no_output=no_output
            )

            # lets make dedicated train and val so that we dont have to worry about it anymore

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

    def post_process_train(self, out=None, save_path_prefix=None, no_output=None):
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
            f"{save_path_prefix}_train.pt",
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
            f"{save_path_prefix}_val.pt",
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

        if self.file_location == "s3":
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

    
    def set_up_cache(self):

        logger.info("delete cache files if any")

        os.makedirs(self.cache, exist_ok=True)

        for f in tqdm.tqdm(glob.glob(os.path.join(self.cache, "*"))):
            os.remove(f)


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

        self.set_up_cache()

        logger.info("loading files")


        with ProcessPoolExecutor(max_workers=num_workers) as exec:
            
            cache_files = [
                os.path.join(self.cache, f"{i:06}_cache.pt") for i in range(len(files))
            ]

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
