from gaia.data import get_variable_index
from netCDF4 import Dataset as netCDF4_Dataset
import torch

prefix= "/ssddg1/gaia/cam4_upload_230418/cam4-famip-30m-timestep-third-upload_24"

var_index_name = f"{prefix}_var_index.pt"
ref_dataset = netCDF4_Dataset(f"{prefix}_sample.nc")

inputs = open("/ssddg1/gaia/cam4_upload_230418/vars.txt").read().split("\n")

print(inputs)

var_index = dict()
var_index["input_index"] = get_variable_index(ref_dataset, inputs)
print(var_index)

torch.save(var_index,var_index_name)