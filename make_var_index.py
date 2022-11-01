from gaia.data import get_variable_index
from netCDF4 import Dataset as netCDF4_Dataset
import torch

prefix= "/ssddg1/gaia/cam4_upload4_v1/cam4-famip-30m-timestep-with-b_relhum-4rth-upload"

var_index_name = f"{prefix}_var_index.pt"
ref_dataset = netCDF4_Dataset("/proj/gaia-climate/data/cam4_upload4/rF_AMIP_CN_CAM4--torch-test.cam2.h1.1979-01-01-00000.nc")

inputs = open("/ssddg1/gaia/cam4_upload4_v1/vars.txt").read().split(",")

print(inputs)

var_index = dict()
var_index["input_index"] = get_variable_index(ref_dataset, inputs)
print(var_index)

torch.save(var_index,var_index_name)