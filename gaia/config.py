from omegaconf import OmegaConf, DictConfig
from mergedeep import merge
import sys
import json
import yaml
from gaia import get_logger
import re

logger = get_logger(__name__)


levels = {"spcam" : [
    3.64346569404006,
    7.594819646328688,
    14.356632251292467,
    24.612220004200935,
    38.26829977333546,
    54.59547974169254,
    72.01245054602623,
    87.82123029232025,
    103.31712663173676,
    121.54724076390266,
    142.99403876066208,
    168.22507977485657,
    197.9080867022276,
    232.82861895859241,
    273.9108167588711,
    322.2419023513794,
    379.10090386867523,
    445.992574095726,
    524.6871747076511,
    609.7786948084831,
    691.3894303143024,
    763.404481112957,
    820.8583686500788,
    859.5347665250301,
    887.0202489197254,
    912.644546944648,
    936.1983984708786,
    957.485479535535,
    976.325407391414,
    992.556095123291,
],
"cam4" : [
    3.5446380000000097,
    7.3888135000000075,
    13.967214000000006,
    23.944625,
    37.23029000000011,
    53.1146050000002,
    70.05915000000029,
    85.43911500000031,
    100.51469500000029,
    118.25033500000026,
    139.11539500000046,
    163.66207000000043,
    192.53993500000033,
    226.51326500000036,
    266.4811550000001,
    313.5012650000006,
    368.81798000000157,
    433.8952250000011,
    510.45525500000167,
    600.5242000000027,
    696.7962900000033,
    787.7020600000026,
    867.1607600000013,
    929.6488750000024,
    970.5548300000014,
    992.5560999999998,
]
}

def get_levels(dataset):
    if "cam4" in dataset:
        return levels["cam4"]

    elif "spcam" in dataset:
        return levels["spcam"]
    else:
        raise ValueError(f"unknown dataset {dataset}")

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

        if cli_args.get('dataset_params',None) is None:
            logger.info("no dataset provided ... you must be loading it from an existing model")
            return None

        base = cli_args.get('dataset_params',{}).get("prefix",None)

        if base is None:
            dataset_paths = {
                "cam4": "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4",
                "cam4_fixed": "/ssddg1/gaia/fixed/cam4-famip-30m-timestep_4",
                "cam4_v2": "/ssddg1/gaia/cam4_v2/cam4-famip-30m-timestep-second-upload",
                "spcam": "/ssddg1/gaia/spcam/spcamclbm-nx-16-20m-timestep_4",
                "spcam_fixed": "/ssddg1/gaia/fixed/spcamclbm-nx-16-20m-timestep_4",
                "cam4_spatial": "/ssddg1/gaia/spatial/cam4-famip-30m-timestep_4",
                "spcam_spatial": "/ssddg1/gaia/spatial/spcamclbm-nx-16-20m-timestep_4"
            }

            dataset = cli_args.get('dataset_params',{}).get('dataset', None)
            base = dataset_paths[dataset]
            
        if "cam4" in dataset:
            mean_thres = 1e-13
            data_grid = levels["cam4"]

        elif "spcam" in dataset:
            mean_thres = 1e-15
            data_grid = levels["spcam"]
        else:
            raise ValueError(f"unknown dataset {dataset}")

        var_index_file = base + "_var_index.pt"

        #possibly shared params

        batch_size = cli_args.get('dataset_params',{}).get("batch_size",24 * 96 * 144)
        include_index = cli_args.get('dataset_params',{}).get("include_index",False)
        subsample = cli_args.get('dataset_params',{}).get("subsample",1)
        space_filter = cli_args.get('dataset_params',{}).get("space_filter",None)
        inputs = cli_args.get('dataset_params',{}).get("inputs",None)
        outputs = cli_args.get('dataset_params',{}).get("outputs",None)
        data_grid = cli_args.get('dataset_params',{}).get("data_grid",data_grid)
        
        dataset_params = dict(
            train=dict(
                dataset_file=base + "_train.pt",
                batch_size=batch_size,
                shuffle=True,
                flatten=False,  # already flattened
                var_index_file=var_index_file,
                include_index = include_index,
                subsample = subsample,
                space_filter =space_filter,
                inputs = inputs,
                outputs = outputs,
                data_grid = data_grid
            ),
            val=dict(
                dataset_file=base + "_val.pt",
                batch_size=batch_size,
                shuffle=False,
                flatten=False,  # already flattened
                var_index_file=var_index_file,
                include_index = include_index,
                subsample = subsample,
                space_filter =space_filter,
                inputs = inputs,
                outputs = outputs,
                data_grid = data_grid
            ),
            test=dict(
                dataset_file=base+'_test.pt',
                batch_size=batch_size,
                shuffle=False,
                flatten=True,  # already flattened
                var_index_file=var_index_file,
                include_index = include_index,
                subsample = subsample,
                space_filter =space_filter,
                inputs = inputs,
                outputs = outputs,
                data_grid = data_grid
            ),
            mean_thres=mean_thres,
            dataset = dataset
        )
        return dataset_params
    
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
    elif model_type == "fcn_with_index":
        model_config = {
            "model_type": "fcn_with_index",
            "num_layers": 7,
            "hidden_size": 512,
            "dropout": 0.01,
            "leaky_relu": 0.15
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
        
    