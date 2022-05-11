# Datasets
## GCM
- [Community Atmospheric Model (CAM4)](https://www.cesm.ucar.edu/models/ccsm4.0/cam/)
- 30 minute time-step
- 2.5-degree grid (144x96)
- 30 levels
- One year debug run (1979 SST; Time Varying) been extended to four than ten years.
- Outputs every 3 hours + additional model time-step (memory)

## CRM
- [SPCAM (super parameterized CAM)](https://ncar.github.io/CAM/doc/build/html/users_guide/atmospheric-configurations.html#super-parameterized-cam-spcam)
- 20 minute time-step
- 16 SAM (The System for Atmospheric Modeling) Columns
- 26 levels
- Year 2000 SST (Climatology)
- Three year simulations:
  -  Morrison Microphysics + Conventional parameterization for moist convection and large-scale condensation.
  - Morrison Microphysics + Higher-order turbulence closure scheme, Cloud Layers Unified By Binormals (CLUBB)
- Outputs every 3 hours + additional model time-step (memory)

## LES
- [WRF (Weather Research and Forecasting Model)](https://www2.mmm.ucar.edu/wrf/users/model_overview.html)
- 50 km x 50 km domains; periodic boundary conditions
- 100 levels
- 100 weather cases
- 1 week spin up at 2 km + 5 day simulation at 200m resolution (LES)
- 3 hourly Boundary Conditions by SPCAM runs + nudging of state variables
- History outputs at 10 minutes (horizontally averaged and mapped to same vertical grid as CAM4)


## Approach
- Develop an AI surrogate to parametric atmospheric physics models used by a GCM at sub-grid scales
- Refine the surrogate using a fine-grid (200m) LES model, focusing on environmental regimes relevant to the MJO and "super-MJO" tipping points
- Novel memory feauture helps the surrogate capture persistent local convection phenomena
- Embed into a GCM to produce a multi-scale AI hybrid model that for the first time accurately captures MJO-relevant convection aggregation at all scales > 200m
- Exploit manifold learning to characterize MJO tipping points and signatures, refining the surrogate as needed