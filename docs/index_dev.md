# Welcome to DARPA AIE ACTM: Gaia Team Project Page

## Team
- [STR](str.us)
- [University of New South Wales, Sydney](https://www.ccrc.unsw.edu.au/ccrc-team/academic-research/steven-sherwood)


## Approach
- Develop an AI surrogate to parametric atmospheric physics models used by a GCM at sub-grid scales
- Refine the surrogate using a fine-grid (200m) LES model, focusing on environmental regimes relevant to the MJO and "super-MJO" tipping points
- Novel memory feauture helps the surrogate capture persistent local convection phenomena
- Embed into a GCM to produce a multi-scale AI hybrid model that for the first time accurately captures MJO-relevant convection aggregation at all scales > 200m
- Exploit manifold learning to characterize MJO tipping points and signatures, refining the surrogate as needed
- 

## Sections
- [Datasets](sections/datasets/README.md) - description of datasets used in surrogate training
- [Surrogate Details and Hyperparameter Sweeps](sections/baseline_hparams/README.md) - neural network architecture and hyperparameter comparison
- [Comparison of Surrogates Trained on Different Datasets](sections/surrogate_comparison/README.md) - compare surrugates to CAM4, SPCAM simulations and compare CAM4 and SPCAM4 trained surrogates to each other
- [Baseline Input Ablation](sections/baseline_input_ablation/README.md) - ablation of input variables and their effect of model performance
- [Surrogate with Memory Inputs](sections/memory/README.md) - improving surrogate performance by adding memory i.e. outputs from previous timestep
- [Integration of AI Surrogate into GCM](sections/gcm_integration/README.md) - deploying GAIA ML model in the GCM
- [MJO Analysis](sections/mjo_analysis/README.md) - analysis of MJO in the datasets

## Milestone Reports

- [Milestone Report 1](milestone_report_1.pdf)
- [Milestone Report 2](milestone_report_2.pdf)


[.](sdfhj32fsfva/results.md)