import torch
import xarray as xr

from metpy.constants.nounit import sat_pressure_0c, epsilon


def relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity):
    # converted to torch from mpc.relative_humidity_from_specific_humidity

    return mixing_ratio_from_specific_humidity(
        specific_humidity
    ) / saturation_mixing_ratio(pressure, temperature)


def mixing_ratio_from_specific_humidity(specific_humidity):
    return specific_humidity / (1 - specific_humidity)


def saturation_mixing_ratio(total_press, temperature):
    return mixing_ratio(saturation_vapor_pressure(temperature), total_press)


def mixing_ratio(partial_press, total_press, molecular_weight_ratio=epsilon):
    return molecular_weight_ratio * partial_press / (total_press - partial_press)


def saturation_vapor_pressure(temperature):
    return sat_pressure_0c * torch.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )


class HumidityConversion(torch.nn.Module):
    def __init__(
        self, hyam=None, hybm=None, P0=None, time_step_seconds=30 * 60
    ) -> None:
        super().__init__()

        self.register_buffer("hyam", torch.tensor(hyam))
        self.register_buffer("hybm", torch.tensor(hybm))

        self.reference_pa = P0
        self.time_step_seconds = time_step_seconds

    @classmethod
    def from_nc_file(cls, file):
        dataset = xr.load_dataset(file)
        keys = ["hyam", "hybm", "P0"]
        return cls(**{k: dataset[k].values.tolist() for k in keys})

    def calc_pressure(self, ps):
        # shape: vertical x levels
        sigma_pressure = self.hyam * self.reference_pa

        # shape: batch x levels
        orog_component = ps * self.hybm[None, :]

        press_pa = sigma_pressure[None, :] + orog_component

        # shape: batch x levels
        return press_pa

    def spec2rel(self, temp_k, specific_humidity, ps):
        press_pa = self.calc_pressure(ps)
        mr = mixing_ratio_from_specific_humidity(specific_humidity)
        smr = saturation_mixing_ratio(press_pa, temp_k)
        rh = 100 * mr / smr
        return rh

    def rel2spec(self, temp_k, rel_hum, ps):
        press_pa = self.calc_pressure(ps)
        # mr = mixing_ratio_from_specific_humidity(specific_humidity)
        smr = saturation_mixing_ratio(press_pa, temp_k)
        a = smr * rel_hum / 100
        q = a / (1 + a)
        return q

    def forward(self, hum, temp_k, ps, mode="spec2rel"):
        if mode == "rel2spec":
            return self.rel2spec(temp_k, hum, ps)
        elif mode == "spec2rel":
            return self.spec2rel(temp_k, hum, ps)
        else:
            raise ValueError(f"unknown mode {mode}")


class RelHumConstraint(torch.nn.Module):
    def __init__(
        self,
        input_index,
        output_index,
        input_normalize,
        output_normalize,
        file="/proj/gaia-climate/data/cam4_v3/rF_AMIP_CN_CAM4--torch-test.cam2.h1.1979-01-01-00000.nc",
        activation="clip",
        ub=110,
        lb=0,
    ) -> None:
        super().__init__()
        self.hum_conversion = HumidityConversion.from_nc_file(file=file)
        self.activation = activation
        self.ub = ub
        self.lb = lb
        self.input_normalize = input_normalize
        self.output_normalize = output_normalize

        input_length = list(input_index.values())[-1][-1]
        output_length = list(output_index.values())[-1][-1]
        self.index = dict()

        for v in ["B_PS", "B_Q", "B_T"]:
            mask = torch.zeros(input_length).bool()
            s,e = input_index[v]
            self.index[v] = (s,e)
            mask[s:e] = True
            self.register_buffer(v,mask)

        for v in ["A_PTTEND", "A_PTEQ"]:
            mask = torch.zeros(output_length).bool()
            s,e = output_index[v]
            self.index[v] = (s,e)
            mask[s:e] = True
            self.register_buffer(v,mask)

    def forward(self, x, y, time_step_seconds=30 * 60):

        x_denorm = self.input_normalize(x,False)
        y_denorm = self.output_normalize(y,False)

        ps, q, t = [
            x_denorm[:, self.index[v][0] : self.index[v][1]] for v in ["B_PS", "B_Q", "B_T"]
        ]
        dt = y_denorm[:, self.index["A_PTTEND"][0] : self.index["A_PTTEND"][1]]
        

        t1 = t + dt * time_step_seconds

        ub = (
            self.hum_conversion(self.ub, t1, ps, mode="rel2spec") - q
        ) / time_step_seconds

        lb = (
            self.hum_conversion(self.lb, t1, ps, mode="rel2spec") - q
        ) / time_step_seconds


        ## normalize them back

        dq_std = self.output_normalize.std[:,self.index["A_PTEQ"][0] : self.index["A_PTEQ"][1], 0, 0]
        
        dq = y[:,self.index["A_PTEQ"][0] : self.index["A_PTEQ"][1]]

        ub = ub / dq_std
        lb = lb / dq_std


        if self.activation == "sigmoid":
            dq = torch.sigmoid(dq) * (ub - lb) + lb
        elif self.activation == "clip":
            # dq =  dq.clip(min=-1, max=1)
            dq = torch.min(dq,ub)
            dq = torch.max(dq,lb)
            # dq = dq*2
        else:
            raise ValueError(f"unknown activation {self.activation}")

        out = y.clone()
        
        out[:,self.index["A_PTEQ"][0] : self.index["A_PTEQ"][1]] = dq

        return out


# # unconstrained
# y1 = model(x, other_things)

# # y2 \in [0,100+eps]
# y2 = contraint(y1, other_outputs)

# #
# y3 = convert_to_spec_hum(y2)
