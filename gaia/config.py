from omegaconf import OmegaConf, DictConfig
from mergedeep import merge
import sys
import json
import yaml
from gaia import get_logger
import re

logger = get_logger(__name__)

class Config():
    valid_top_level = ["mode","seed","interpolation_params","dataset_params" ,"trainer_params","model_params" ]
    """
    Initialize config with default parameter
    then parse cli args and merge
    """
    def __init__(self, cli_args=dict()):
        """
        Set default model parameters
        """
        # set general params (mode, seed, etc.)

        for k in cli_args:
            if k not in self.valid_top_level:
                logger.warn(f"{k} invalid top level param category, ignoring")
                cli_args.pop(k)

        mode = 'train,val,test'
        seed = True
        interpolation_params = None
        
        # set trainer params 
        trainer_params = self.set_trainer_params(cli_args)
        
        # set dataset params
        dataset_params = self.set_dataset_params(cli_args)
        
        # set model params
        model_params = self.set_model_params(cli_args)
                
        # general config
        config = dict(
            mode = mode,
            trainer_params = trainer_params,
            dataset_params = dataset_params,
            model_params = model_params,
            seed = seed,
            interpolation_params = interpolation_params,
        )
        self.config = merge(config, cli_args)

        logger.info(f"Config: \n{self}")


    def __repr__(self) -> str:
        return yaml.dump(self.config, indent=2)

    @classmethod
    def readCLIargs(cls):
        """
        Get the CLI args to override defaults during runtime
        """       
        cli_args = OmegaConf.to_container(OmegaConf.from_cli())
        t = json.dumps(cli_args).translate(r'{}:\"\'')
        logger.info(f'CLI parameters: \n{t}')
        return cls(cli_args = cli_args)
    
    @staticmethod
    def set_trainer_params(cli_args=dict()):
        """
        Set trainer params
        """
        default_trainer_params = dict(
            gpus=[1],
            precision=16,
            max_epochs=200
        )
        return default_trainer_params
        # return merge(default_trainer_params, cli_args.get('trainer_params',{}))
    
    @staticmethod
    def set_dataset_params(cli_args=dict()):
        """
        Set the dataset params
        """
        dataset_paths = {
            "cam4": "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4",
            "spcam": "/ssddg1/gaia/spcam/spcamclbm-nx-16-20m-timestep_4",
        }
        dataset = cli_args.get('dataset_params',{}).get('dataset', 'cam4')
        base = dataset_paths[dataset]
        mean_thres_defaults = {"cam4": 1e-13, "spcam": 1e-15}
        var_index_file = base + "_var_index.pt"
        batch_size = cli_args.get('dataset_params',{}).get("batch_size",24 * 96 * 144)
        
        dataset_params = dict(
            train=dict(
                dataset_file=base + "_train.pt",
                batch_size=batch_size,
                shuffle=True,
                flatten=False,  # already flattened
                var_index_file=var_index_file
            ),
            val=dict(
                dataset_file=base + "_val.pt",
                batch_size=batch_size,
                shuffle=False,
                flatten=False,  # already flattened
                var_index_file=var_index_file
            ),
            test=dict(
                dataset_file=base+'_test.pt',
                batch_size=batch_size,
                shuffle=False,
                flatten=True,  # already flattened
                var_index_file=var_index_file
            ),
            mean_thres=mean_thres_defaults[dataset]
        )
        return dataset_params
        # return merge(dataset_params, cli_args.get('dataset_params',{}))
    
    @staticmethod
    def set_model_params(cli_args=dict()):
        model_config = model_type_lookup(cli_args.get('model_params',{}).get('model_type', 'fcn'))
        model_params = dict(
            memory_variables=None,      # can be ',' sep
            ignore_input_variables=None,# can be ',' sep
            model_config = model_config,
            lr=1e-3,
            use_output_scaling=False,
            replace_std_with_range=False,
            ckpt=None,
            lr_schedule = "cosine"
        )
        return model_params
        # return merge(model_params, cli_args.get('model_params',{}))

    
def model_type_lookup(model_type):
    """
    Define the model_configs for various model_types
    """
    # you can change th
    if model_type == "fcn":
        model_config = {
            "model_type": "fcn",
            "num_layers": 7,
            "hidden_size": 512,
            "dropout": 0.01,
            "leaky_relu": 0.15
        }
    elif model_type == "fcn_history":
        model_config = {
            "model_type": "fcn_history",
            "num_layers": 7,
            "hidden_size": 512,
            "leaky_relu": 0.15
        }

    elif model_type == "conv1d":
        model_config = {
            "model_type": "conv1d",
            "num_layers": 7,
            "hidden_size": 128
        }

    elif model_type == "conv2d":
        model_config = {
            "model_type": "conv2d",
            "num_layers": 7,
            "hidden_size": 176,
            "kernel_size": 3,
        }

    elif model_type == "resdnn":
        model_config = {
            "model_type": "resdnn",
            "num_layers": 7,
            "hidden_size": 512,
            "dropout": 0.01,
            "leaky_relu": 0.15
        }
    elif model_type == "encoderdecoder":
        model_config = {
            "model_type": "encoderdecoder",
            "num_layers": 7,
            "hidden_size": 512,
            "dropout": 0.01,
            "leaky_relu": 0.15,
            "bottleneck_dim": 32,
        }
    elif model_type == "transformer":
        model_config = {
            "model_type": "transformer",
            "num_layers": 3,
            "hidden_size": 128,
        }
    else:
        raise ValueError
    
    return model_config


    def merge_cli_args(self):
        """
        Merge cli args with default parameters
        """
        raise DeprecationWarning
        print(OmegaConf.to_yaml(DictConfig(self.config)))
        cli_args = dict(OmegaConf.from_cli())
        # model_config initialize here
        self.config = merge(self.config, cli_args)
        print(OmegaConf.to_yaml(DictConfig(self.config)))
        
    