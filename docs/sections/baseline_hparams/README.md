# Baseline Surrogate Model <!-- omit in toc --> 

- [Surrogate Architecture](#surrogate-architecture)
- [Hyperparameters](#hyperparameters)
- [Dataset Variables](#dataset-variables)
  - [Inputs](#inputs)
  - [Outputs](#outputs)

## Surrogate Architecture

- 7 FCN (fully connected layers)
- Each layer has a hidden dimension of 512.
- Each layer is followed by batch normalization and dropout with rate of .1

[![](surrogate_architecture.png)](surrogate_architecture.png)


## Hyperparameters

We evaluate several parameter values:
- num_layers: 3,5,7,14
- hidden_size: 128, 256, 512, 1024, 1536

[![](baseline_hparams.png)](baseline_hparams.png)


## Dataset Variables


Shapes are (timesteps (T), number of levels (L), number of lat bins, number of lon bins)

- CAM4 has L =  L levels
- SPCAM has L = L levels

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

