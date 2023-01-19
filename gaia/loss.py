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




class RelHumGradientReg(torch.nn.Module):
    def __init__(self, input_index, hum_name = "B_Q", temp_name = "B_T", ps_name = "B_PS", rel_hum_threshold = 60, model = None, file="/proj/gaia-climate/data/cam4_v3/rF_AMIP_CN_CAM4--torch-test.cam2.h1.1979-01-01-00000.nc" ):
        super().__init__()
        self.hum_index = input_index[hum_name]
        self.temp_index = input_index[temp_name]
        self.ps_index = input_index[ps_name]
        self.hum_conversion = HumidityConversion.from_nc_file(file)
        self.model = model
        self.rel_hum_threshold  = rel_hum_threshold


    def forward(self, x):
        hum = x[:,self.hum_index[0]:self.hum_index[1],...]
        temp_k = x[:,self.temp_index[0]:self.temp_index[1],...]
        ps = x[:,self.ps_index[0]:self.ps_index[1],...]
        rel_hum = self.hum_conversion(hum, temp_k, ps, mode = "spec2rel")

        return rel_hum
        



class LinearConstraintResidual(torch.nn.Module):
    def __init__(self, input_index = None, output_index= None, input_signs_and_names= None, output_signs_and_names = None):
        super().__init__()

        def make_signs_and_masks(index, signs_and_names):
            input_length = list(input_index.values())[-1][-1]
            mask = torch.zeros(input_length)
            sign = torch.ones(input_length)

            for temp in signs_and_names:
                sg = 1. if temp[0] == "+" else -1.
                v = temp[1:]
                s,e = index[v]
                mask[s:e] = 1
                sign[s:e] = sg

            return sign, mask

        self.apply_to_input = True

        if input_signs_and_names is not None:
            sign,mask = make_signs_and_masks(input_index,input_signs_and_names )
            self.register_buffer("input_sign", sign)
            self.register_buffer("input_mask", mask)
            self.apply_to_input = True

        self.apply_to_output = True

        if output_signs_and_names is not None:
            sign,mask = make_signs_and_masks(output_index,output_signs_and_names )
            self.register_buffer("output_sign", sign)
            self.register_buffer("output_mask", mask)


    def forward(self, x_unnorm, y_unnorm):

        sm = 0.

        if self.apply_to_input:

            x_sum = x_unnorm * self.inpint_sign[None,:]
            x_sum = x_sum[:,self.input_mask].sum(-1)

            sm += x_sum

        if self.apply_to_output:

            y_sum = y_unnorm * self.output_sign[None, :]
            sm += y_sum[:,self.output_mask].sum(-1)


        return sm



