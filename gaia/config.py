from omegaconf import OmegaConf, DictConfig
from mergedeep import merge

class Config():
    """
    Initialize config with default parameter
    then parse cli args and merge
    """
    def __init__(self):
        """
        Set default model parameters
        """
        
        dataset_paths = {
            "cam4": "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4",
            "spcam": "/ssddg1/gaia/spcam/spcamclbm-nx-16-20m-timestep_4",
        }
        mean_thres_defaults = {"cam4": 1e-13, "spcam": 1e-15}
        
        # general runtime parameters
        self.mode = 'train,val'
        self.dataset = 'cam4'
        self.model_type = 'baseline'
        self.seed = True
        self.interpolation_params = None
        
        # default trainer parameters
        self.trainer_params = dict(
            gpus=[1],
            precision=16,
            max_epochs=250
        )
    
        # default dataset parameters
        self.dataset_params = dict(
            train=dict(
                dataset_file=dataset_paths[self.dataset] + "_train.pt",
                batch_size=24 * 96 * 144,
                shuffle=True,
                flatten=False,  # already flattened
                var_index_file=dataset_paths[self.dataset]+'_var_index.pt',
            ),
            val=dict(
                dataset_file=dataset_paths[self.dataset] + "_val.pt",
                batch_size=24 * 96 * 144,
                shuffle=False,
                flatten=False,  # already flattened
                var_index_file=dataset_paths[self.dataset]+'_var_index.pt',
            ),
            test=dict(
                dataset_file=dataset_paths[self.dataset]+'_var_index.pt',
                batch_size=24 * 96 * 144,
                shuffle=False,
                flatten=True,  # already flattened
                var_index_file=dataset_paths[self.dataset]+'_var_index.pt',
            ),
            mean_thres=mean_thres_defaults[self.dataset]
        )

        # default model parameters
        self.model_params = dict(
            memory_varaibles=None,      # can be ',' sep
            ignore_input_variables=None,# can be ',' sep
            model_config = self.model_type_lookup(self.model_type),
            lr=1e-3,
            use_output_scaling=False,
            replace_std_with_range=False,
            ckpt=None,
        )
        
        # general config
        self.config = dict(
            mode = self.mode,
            dataset = self.dataset,
            model_type = self.model_type,
            seed = self.seed,
            interpolation_params = self.interpolation_params,
            trainer_params = self.trainer_params,
            dataset_params = self.dataset_params,
            model_params = self.model_params
        )
        

    def merge_cli_args(self):
        """
        Merge cli args with default parameters
        """
        cli_args = dict(OmegaConf.from_cli())
        self.config = merge(self.config, cli_args)

        
    def model_type_lookup(self, model_type):
        """
        Define the model_configs for various model_types
        """
        
        if model_type == "baseline":
            model_config = {
                "model_type": "fcn",
                "num_layers": 7,
                "hidden_size": 512,
                "dropout": 0.01,
                "leaky_relu": 0.15
            }
        elif model_type == "memory":
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
        else:
            raise ValueError
        
        return model_config
    