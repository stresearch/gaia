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


class SpecficHumidy2RelativeHumidy(torch.nn.Module):
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
    
    def forward(self ,temp_k, specific_humidity, ps):
        press_pa = self.calc_pressure(ps)
        rh = relative_humidity_from_specific_humidity(press_pa, temp_k, specific_humidity)
        return 100*rh
 


# # unconstrained
# y1 = model(x, other_things)

# # y2 \in [0,100+eps]
# y2 = contraint(y1, other_outputs)

# # 
# y3 = convert_to_spec_hum(y2)
