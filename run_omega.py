import argparse
import os

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import glob
import numpy as np
from gaia.evaluate import process_results
from gaia import get_logger
import yaml

# from argparse import

logger = get_logger(__name__)

from gaia.training import main

from gaia.plot import plot_results
from gaia.config import Config

if __name__ == "__main__":
    config = Config.readCLIargs()
    
    main(mode=config.config['general_params']['mode'],
         trainer_params=config.config['trainer_params'],
         dataset_params=config.config['dataset_params'],
         model_params=config.config['model_params'],
         seed=config.config['general_params']['seed'],
         interpolation_params=config.config['general_params']['interpolation_params']
         )
