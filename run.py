import argparse
import os

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import glob
from mergedeep import Strategy
import numpy as np
from gaia.evaluate import process_results
from gaia import get_logger
import yaml

# from argparse import

logger = get_logger(__name__)


from gaia.training import (
    main,
    default_trainer_params,
    default_dataset_params,
    default_model_params,
)

from gaia.plot import plot_results

# "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4"

dataset_names = {
    "cam4": "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4",
    "spcam": "/ssddg1/gaia/spcam/spcamclbm-nx-16-20m-timestep_4",
}

mean_thres_defaults = {"cam4": 1e-13, "spcam": 1e-15}


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ignore_input_variables", default=None, type=str)
    parser.add_argument("--memory_variables", default=None, type=str)
    parser.add_argument(
        "--dataset",
        default="cam4",
        type=str,
    )
    parser.add_argument("--gpu", default=2, type=int)
    parser.add_argument("--model_type", default="baseline", type=str)
    parser.add_argument("--mode", default="train,val,test", type=str)
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_layers", default=7, type=int)
    parser.add_argument("--batch_size", default=24 * 96 * 144, type=int) #24 * 96 * 144 96 * 144 // 2
    parser.add_argument("--dropout", default=0.01, type=float)
    parser.add_argument("--mean_thres", default=None, type=float)
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--leaky_relu", default=0.15, type=float)
    parser.add_argument("--bottleneck", default=32, type = int)
    parser.add_argument("--pretrained", default=None, type = str)


    args = parser.parse_args()

    mean_thres_defaults = {"cam4": 1e-13, "spcam": 1e-15}
    args.mean_thres = mean_thres_defaults[args.dataset]

    args.dataset = dataset_names[args.dataset]

    if args.model_type == "baseline":

        model_config = {
            "model_type": "fcn",
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "leaky_relu": args.leaky_relu
            #   "num_output_layers": 6
        }
    elif args.model_type == "memory":
        model_config = {
            "model_type": "fcn_history",
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "leaky_relu": args.leaky_relu
            #   "num_output_layers": 6
        }

    elif args.model_type == "conv1d":
        model_config = {
            "model_type": "conv1d",
            "num_layers": 7,
            "hidden_size": 128,
            #   "num_output_layers": 6
        }

    elif args.model_type == "resdnn":
        model_config = {
            "model_type": "resdnn",
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "leaky_relu": args.leaky_relu
            #   "num_output_layers": 6
        }
    elif args.model_type == "encoderdecoder":
        model_config = {
            "model_type": "encoderdecoder",
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "leaky_relu": args.leaky_relu,
            "bottleneck_dim": args.bottleneck,
            #   "num_output_layers": 6
        }
    elif args.model_type == "transformer":
        model_config = {
            "model_type": "transformer",
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
        }
    else:
        raise ValueError

    main(
        args.mode,
        trainer_params=default_trainer_params(
            gpus=[args.gpu], precision=16, max_epochs=args.max_epochs
        ),
        dataset_params=default_dataset_params(
            base=args.dataset, batch_size=args.batch_size, mean_thres=args.mean_thres
        ),
        model_params=default_model_params(
            memory_variables=args.memory_variables.split(",")
            if args.memory_variables
            else None,
            ignore_input_variables=args.ignore_input_variables.split(",")
            if args.ignore_input_variables
            else None,
            lr=args.lr,
            use_output_scaling=False,
            replace_std_with_range=False,
            model_config=model_config,
            ckpt=args.ckpt,
            pretrained = args.pretrained,
            lr_schedule = "cosine"
        ),
        seed = True
    )


if __name__ == "__main__":
    run()
