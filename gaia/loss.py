import torch
from gaia.physics import HumidityConversion

def total_variation(img):

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().mean(dim=reduce_axes)
    res2 = pixel_dif2.abs().mean(dim=reduce_axes)

    return res1 + res2


class RelHumWeight(torch.nn.Module):
    def __init__(self, input_index, hum_name = "B_Q", temp_name = "B_T", ps_name = "B_PS", file="/proj/gaia-climate/data/cam4_v3/rF_AMIP_CN_CAM4--torch-test.cam2.h1.1979-01-01-00000.nc" ):
        super().__init__()
        self.hum_index = input_index[hum_name]
        self.temp_index = input_index[temp_name]
        self.ps_index = input_index[ps_name]
        self.hum_conversion = HumidityConversion.from_nc_file(file)


    def forward(self, x):
        hum = x[:,self.hum_index[0]:self.hum_index[1],...]
        temp_k = x[:,self.temp_index[0]:self.temp_index[1],...]
        ps = x[:,self.ps_index[0]:self.ps_index[1],...]
        rel_hum = self.hum_conversion(hum, temp_k, ps, mode = "spec2rel")

        weight = rel_hum[:,10:].mean(1).clip(min = 10)

        # weight = weight**2
        weight = weight / weight.sum() * weight.shape[0]

        return weight
        