import os
from gaia import get_logger
logger = get_logger(__name__)

os.environ["GAIA_CAM4_CUSTOM"] = "/ssddg1/gaia/cam4_v5/cam4-famip-30m-timestep-third-upload"
logger.warning(f'change the dataset location, using {os.environ["GAIA_CAM4_CUSTOM"]}')


from default_inputs_outputs import variable_spec_2210_V1
from gaia.config import Config, get_levels
from gaia.training import main


gpu = 6

logger.warning(f"using GPU ID {gpu}")


inputs, outputs = variable_spec_2210_V1()

config = Config(
    {
        "mode": "train, test, predict,export,plot",
        "dataset_params": {
            "dataset": "cam4_custom",
            "inputs": inputs,
            "outputs": outputs,
        },
        "trainer_params": {
            "gpus": [gpu],
            "max_epochs": 100,
            "precision": 16,
            "gradient_clip_val": 0.5,
            "track_grad_norm": 2,
        },
        "model_params": {
            "model_type": "fcn",
            "model_grid": get_levels("cam4"),
            "upweigh_low_levels": True,
            "weight_decay": 1.0,
            "lr": 1e-3,
            "positive_output_pattern": "PREC,FS,FL,SOL",
            "positive_func": "rectifier",
        },
        "seed": 345,
    }
)


model_dir = main(**config.config)
