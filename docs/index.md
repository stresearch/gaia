# Global AI Accelerator - GAIA <!-- omit in toc --> 

Welcome to GAIA Team Project Page, a DARPA AIE ACTM project.

Code repository and documentation for:
-  training NN surrogates: [https://github.com/stresearch/gaia](https://github.com/stresearch/gaia)
-  deploying NN surrogates in GCM (still private, will be made public soon): [https://github.com/stresearch/gaia-deploy](https://github.com/stresearch/gaia-deploy)

## Project Description <!-- omit in toc --> 

- [Sections](#sections)
- [Overview](#overview)
- [Milestone Reports](#milestone-reports)

## Team <!-- omit in toc --> 

![](sections/overview/str.png)   ![](https://www.unsw.edu.au/content/dam/images/graphics/logos/unsw/unsw_0.png)

- [STR](https://www.str.us) 
  -   Jim Gimlett
  -   Gregory Bushey
  -   Kirill Trapeznikov
  -   Eric Jones
- [University of New South Wales, Sydney](https://www.unsw.edu.au/)
  -  [Steven Sherwood](https://www.unsw.edu.au/staff/steven-sherwood)
  -  [Abhnil Prasad](https://www.unsw.edu.au/staff/abhnil-prasad)
  -  David Fuchs 

## Sections
- [Datasets](sections/datasets/README.md) - description of datasets used in surrogate training
- [Surrogate Details and Hyperparameter Sweeps](sections/baseline_hparams/README.md) - neural network architecture and hyperparameter comparison
- [Comparison of Surrogates Trained on Different Datasets](sections/surrogate_comparison/README.md) - compare surrogates to CAM4, SPCAM simulations and compare CAM4 and SPCAM4 trained surrogates to each other
- [Baseline Input Ablation](sections/baseline_input_ablation/README.md) - ablation of input variables and their effect on model performance
- [Surrogate with Memory Inputs](sections/memory/README.md) - improving surrogate performance by adding memory i.e. outputs from previous timestep
- [Integration of AI Surrogate into GCM](sections/gcm_integration/README.md) - deploying GAIA ML model in the GCM
- [MJO Analysis](sections/mjo_analysis/README.md) - analysis of MJO in the datasets
- [Surrogate Bottleneck Analysis](sections/bottleneck/README.md) - analysis of bottleneck dimension of the surrogate model
- [Linearization](sections/linearization/README.md) - analysis linearization of surrogate models to understand sensitivity to inputs
- [Surrogate Architectures](sections/architectures/README.md) - comparison of different neural network architectures and configurations
- [Fine Tuning](sections/finetuning/README.md) - comparison of fine-tuning an CAM4 trained model on SPCAM vs a model traiend from scratch
- [Intergration Journal](sections/integration/README.md) - comparison of fine-tuning an CAM4 trained model on SPCAM vs a model traiend from scratch

## Overview

*Animation of CAM4 Simulation and Neural Network Surrogate Model*
[![](sections/overview/overview_screenshot.png)](https://855da60d-505b-4eee-942c-e19fb87dcc5f.s3.amazonaws.com/gaia/videos/cam4_sim-cam4_nn-PRECT.mp4)

**Concept & Approach**
- Develop new hybrid AI tools and methods to accelerate ultra-high-resolution Global Climate Models (GCMs) models, substituting AI local surrogates to improve cloud-resolving physics
- Test climate “tipping point” hypotheses, with focus on the Madden Julian Oscillation (MJO) 

![](sections/overview/overview.png)

**What’s hard?**
- GCMs are computationally expensive and lack the resolution needed to adequately model local convection and thus clouds
- This greatly increases GCM forecast errors and impedes propagation of large-scale wave phenomena such as the MJO 


**What will GAIA accomplish?**
- Enable a GCM to accurately model local convection and predict self-organizing atmospheric wave phenomena 
- Exploit this GCM to explore possible future regimes and identify early warning signatures for MJO-related tipping points.
  

**Progress**
- Developed and validated high-skill AI surrogates for multiple GCM local models, including CAM4, SPCAM with Morrison micro-physics, SPCAM with CLUBB cloud parameterization
- Currently studying impact of added memory and ablating inputs
- Next Steps: 
  - Determine input regimes where CAM/SPCAM underperform
  - Analyze GCM sensitivity using lower order dynamical model 
  - Apply results to selectively enhance local surrogate using ultra-high-resolution local eddy simulation for training. 
  - Integrate surrogate into GCM; use GCM to test MJO hypotheses


## Milestone Reports

- [Milestone Report 1](milestone_reports/milestone_report_1.pdf)
- [Milestone Report 2](milestone_reports/milestone_report_2.pdf)
- [Milestone Report 3](milestone_reports/milestone_report_3.pdf)
- [Milestone Report 4](milestone_reports/milestone_report_4.pdf)
- [Milestone Report 5](milestone_reports/milestone_report_5.pdf)
- [Milestone Report 6](milestone_reports/milestone_report_6.pdf)


