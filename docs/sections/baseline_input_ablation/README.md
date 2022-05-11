# Input Variable Abalstion

Ablation of input variable groups is performed on the surrogate model trained CAM4 data. We measure over skill on a test set as we remove one variable (group) at a time

- Each line corresponds to skill per output variable (note that PTEQ and PTTEND have 26 levels each while PRECC, PRECT are scalars)
- Dashed lines correspond to skill for each output with all input variables
- We observe that OMEGA, Q, T have the greatest impact

[![](ablation_baseline.png)](ablation_baseline.png)


## Variable names for reference

### Inputs

| Name | Long Name | shape | unit |
| --- | --- | --- | --- |
| Q | Specific humidity | (T, L, 96, 144) | kg/kg|
| T | Temperature | (T, L, 96, 144) | K|
| U | Zonal wind | (T, L, 96, 144) | m/s|
| V | Meridional wind | (T, L, 96, 144) | m/s|
| OMEGA | Vertical velocity (pressure) | (T, L, 96, 144) | Pa/s|
| PSL | Sea level pressure | (T, 96, 144) | Pa|
| SOLIN | Solar insolation | (T, 96, 144) | W/m2|
| SHFLX | Surface sensible heat flux | (T, 96, 144) | W/m2|
| LHFLX | Surface latent heat flux | (T, 96, 144) | W/m2|
| FSNS | Net solar flux at surface | (T, 96, 144) | W/m2|
| FLNS | Net longwave flux at surface | (T, 96, 144) | W/m2|
| FSNT | Net solar flux at top of model | (T, 96, 144) | W/m2|
| FLNT | Net longwave flux at top of model | (T, 96, 144) | W/m2|
| Z3 | Geopotential Height (above sea level) | (T, L, 96, 144) | m|

### Outputs

| Name | Long Name | shape | unit |
| --- | --- | --- | --- |
| PRECT | Total (convective and large-scale) precipitation rate (liq + ice) | (T, 96, 144) | m/s|
| PRECC | Convective precipitation rate (liq + ice) | (T, 96, 144) | m/s|
| PTEQ | Q total physics tendency | (T, L, 96, 144) | kg/kg/s|
| PTTEND | T total physics tendency | (T, L, 96, 144) | K/s|




