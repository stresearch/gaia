# Comparison of Surrogate Models <!-- omit in toc -->

- [Summary](#summary)
- [CAM4 Simulation vs CAM4-trained Neural Network](#cam4-simulation-vs-cam4-trained-neural-network)
- [SPCAM Simulation vs SPCAM-trained Neural Network](#spcam-simulation-vs-spcam-trained-neural-network)
- [CAM4-trained Neural Network vs SPCAM-trained Neural Network](#cam4-trained-neural-network-vs-spcam-trained-neural-network)
- [SPCAM-trained Neural Network vs CAM4-trained Neural Network](#spcam-trained-neural-network-vs-cam4-trained-neural-network)
- [Animations of Model Performance and Comparison Over Time](#animations-of-model-performance-and-comparison-over-time)


## Summary

Predictions of two models are compared on the specified dataset.
- To compute skill the first model is used as "reference" i.e. to compute the variance
- For PTEQ* skill, high altitude levels are ignored since they are mostly zero/process noise

> comparison: cam4_sim-cam4_nn, dataset: cam4

|output|skill|mse|
|---|---|---|
|PRECT|0.965|1.51e-16|
|PRECC|0.956|7.15e-17|
|PTEQ*|0.874|4.2e-17|
|PTTEND|0.929|1.07e-10|

> comparison: cam4_nn-spcam_nn, dataset: cam4
  
|output|skill|mse|
|---|---|---|
|PRECT|0.682|1.14e-15|
|PRECC|0.0|2.29e-15|
|PTEQ*|0.219|1.1e-17|
|PTTEND|0.363|1.13e-09|

> comparison: spcam_sim-spcam_nn, dataset: spcam 

|output|skill|mse|
|---|---|---|
|PRECT|0.947|4.72e-16|
|PRECC|0.947|4.72e-16|
|PTEQ*|0.693|3.2e-16|
|PTTEND|0.85|3.28e-10|

> comparison: spcam_nn-cam4_nn, dataset: spcam 

|output|skill|mse|
|---|---|---|
|PRECT|0.902|8.41e-16|
|PRECC|0.4|5.16e-15|
|PTEQ*|0.505|2.19e-16|
|PTTEND|0.607|7.7e-10|

## CAM4 Simulation vs CAM4-trained Neural Network

[*Means for each model, MSE and Skill (click here for interactive plot).*](cam4_sim-cam4_nn.html)
[![](cam4_sim-cam4_nn.png)](cam4_sim-cam4_nn.png)

## SPCAM Simulation vs SPCAM-trained Neural Network

[*Means for each model, MSE and Skill (click here for interactive plot).*](spcam_sim-spcam_nn.html)
[![](spcam_sim-spcam_nn.png)](spcam_sim-spcam_nn.png)

## CAM4-trained Neural Network vs SPCAM-trained Neural Network

Both models are evaluated on the CAM4 input datas

[*Means for each model, MSE and Skill (click here for interactive plot).*](cam4_nn-spcam_nn.html)
[![](cam4_nn-spcam_nn.png)](cam4_nn-spcam_nn.png)

## SPCAM-trained Neural Network vs CAM4-trained Neural Network

Both models are evaluated on the SPCAM input datas

[*Means for each model, MSE and Skill (click here for interactive plot).*](spcam_nn-cam4_nn.html)
[![](spcam_nn-cam4_nn.png)](spcam_nn-cam4_nn.png)


## Animations of Model Performance and Comparison Over Time

Videos showing:
- output over time for two models
- absolute error between model outputs
- skill-like metric where variance is computed over the time component

Table columns:
- model1 - name of first model
- model2 - name of second model
- output - output variable
- evalated on - dataset model evaluated on
  
model naming convension: `{dataset}_{type}` where  
-  `type` in `sim[ulation],nn[neural network]`
-  `dataset` in `cam4,spcam` that the model is trained on `type==nn` or name of simulator



|model1|model2|output|evaluated on|video|
|---|---|---|---|---|
|cam4_nn|spcam_nn|PRECC|cam4|[cam4_nn-spcam_nn-PRECC.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_nn-spcam_nn-PRECC.mp4)|
|cam4_nn|spcam_nn|PRECT|cam4|[cam4_nn-spcam_nn-PRECT.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_nn-spcam_nn-PRECT.mp4)|
|cam4_nn|spcam_nn|PTEQ_24|cam4|[cam4_nn-spcam_nn-PTEQ_24.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_nn-spcam_nn-PTEQ_24.mp4)|
|cam4_nn|spcam_nn|PTEQ_25|cam4|[cam4_nn-spcam_nn-PTEQ_25.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_nn-spcam_nn-PTEQ_25.mp4)|
|cam4_nn|spcam_nn|PTTEND_24|cam4|[cam4_nn-spcam_nn-PTTEND_24.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_nn-spcam_nn-PTTEND_24.mp4)|
|cam4_nn|spcam_nn|PTTEND_25|cam4|[cam4_nn-spcam_nn-PTTEND_25.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_nn-spcam_nn-PTTEND_25.mp4)|
|cam4_sim|cam4_nn|PRECC|cam4|[cam4_sim-cam4_nn-PRECC.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_sim-cam4_nn-PRECC.mp4)|
|cam4_sim|cam4_nn|PRECT|cam4|[cam4_sim-cam4_nn-PRECT.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_sim-cam4_nn-PRECT.mp4)|
|cam4_sim|cam4_nn|PTEQ_24|cam4|[cam4_sim-cam4_nn-PTEQ_24.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_sim-cam4_nn-PTEQ_24.mp4)|
|cam4_sim|cam4_nn|PTEQ_25|cam4|[cam4_sim-cam4_nn-PTEQ_25.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_sim-cam4_nn-PTEQ_25.mp4)|
|cam4_sim|cam4_nn|PTTEND_24|cam4|[cam4_sim-cam4_nn-PTTEND_24.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_sim-cam4_nn-PTTEND_24.mp4)|
|cam4_sim|cam4_nn|PTTEND_25|cam4|[cam4_sim-cam4_nn-PTTEND_25.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_sim-cam4_nn-PTTEND_25.mp4)|
|spcam_nn|cam4_nn|PRECC|spcam|[spcam_nn-cam4_nn-PRECC.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_nn-cam4_nn-PRECC.mp4)|
|spcam_nn|cam4_nn|PRECT|spcam|[spcam_nn-cam4_nn-PRECT.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_nn-cam4_nn-PRECT.mp4)|
|spcam_nn|cam4_nn|PTEQ_28|spcam|[spcam_nn-cam4_nn-PTEQ_28.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_nn-cam4_nn-PTEQ_28.mp4)|
|spcam_nn|cam4_nn|PTEQ_29|spcam|[spcam_nn-cam4_nn-PTEQ_29.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_nn-cam4_nn-PTEQ_29.mp4)|
|spcam_nn|cam4_nn|PTTEND_28|spcam|[spcam_nn-cam4_nn-PTTEND_28.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_nn-cam4_nn-PTTEND_28.mp4)|
|spcam_nn|cam4_nn|PTTEND_29|spcam|[spcam_nn-cam4_nn-PTTEND_29.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_nn-cam4_nn-PTTEND_29.mp4)|
|spcam_sim|spcam_nn|PRECC|spcam|[spcam_sim-spcam_nn-PRECC.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_sim-spcam_nn-PRECC.mp4)|
|spcam_sim|spcam_nn|PRECT|spcam|[spcam_sim-spcam_nn-PRECT.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_sim-spcam_nn-PRECT.mp4)|
|spcam_sim|spcam_nn|PTEQ_28|spcam|[spcam_sim-spcam_nn-PTEQ_28.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_sim-spcam_nn-PTEQ_28.mp4)|
|spcam_sim|spcam_nn|PTEQ_29|spcam|[spcam_sim-spcam_nn-PTEQ_29.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_sim-spcam_nn-PTEQ_29.mp4)|
|spcam_sim|spcam_nn|PTTEND_28|spcam|[spcam_sim-spcam_nn-PTTEND_28.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_sim-spcam_nn-PTTEND_28.mp4)|
|spcam_sim|spcam_nn|PTTEND_29|spcam|[spcam_sim-spcam_nn-PTTEND_29.mp4](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/spcam_sim-spcam_nn-PTTEND_29.mp4)|
