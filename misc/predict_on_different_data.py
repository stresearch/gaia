from gaia.training import default_trainer_params,default_dataset_params,default_model_params,main
from gaia.plot import levels, levels26
import torch
gpus = 5

# dataset = "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4"
dataset_names = {
    "cam4": "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4",
    "spcam": "/ssddg1/gaia/spcam/spcamclbm-nx-16-20m-timestep_4",
}


dataset = "spcam"
ckpt = "lightning_logs_compare_models/spcam_nn"


dataset_params = default_dataset_params(
            base=dataset_names[dataset]
        )

var_index = torch.load(dataset_params["test"]["var_index_file"])

interpolation_params = dict()
interpolation_params["input_index"] = var_index["input_index"]
interpolation_params["output_index"] = var_index["output_index"]

if dataset == "cam4":
    interpolation_params["input_grid"] = levels26
    interpolation_params["output_grid"] = levels
else:
    interpolation_params["input_grid"] = levels
    interpolation_params["output_grid"] = levels26

interpolation_params["prediction_file_name"] = f"predictions_on_{dataset}_1.pt"



main(
        "predict",
        trainer_params=default_trainer_params(
            gpus=[gpus], precision=16
        ),
        dataset_params=dataset_params,
        model_params=default_model_params(
            ckpt=ckpt
        ),
        interpolation_params=None
    )