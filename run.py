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

# files_or_pattern = glob.glob("/proj/gaia-climate/data/cesm106_cam4/*.nc")[:1]


# dataset = NcDatasetMem(files_or_pattern)

# ins, outs = dataset[100]

# print(ins)
# print(outs)

from gaia.training import (
    main,
    default_trainer_params,
    default_dataset_params,
    default_model_params,
)

from gaia.plot import plot_results


# model_config = {
#     "model_type": "fcn",
#     "num_layers": 7,
#     #   "num_output_layers": 6
# }

# # base = "/ssddg1/gaia/spcam/spcamclbm-nx-16-20m-timestep_4"
# base = "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4"
# # ignore_input_variables = None#["OMEGA"]

# # inputs="Q,T,U,V,OMEGA,PSL,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,Z3".split(",")
# inputs = "Q,T,U,V,PSL,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,Z3".split(",")


# read existing

# input_variables_ablation = []

# for model_dir in glob.glob("lightning_logs/version_*"):
#     try:
#         yaml_file = os.path.join(model_dir, "hparams.yaml")
#         params = yaml.unsafe_load(open(yaml_file))
#         if params["model_config"]["model_type"] == "fcn":
#             if "ignore_input_variables" in params:
#                 input_variables_ablation.extend(params["ignore_input_variables"])


#     except:
#         pass


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_var_ignore", default=None, type=str)
    parser.add_argument(
        "--dataset",
        default= "/ssddg1/gaia/cam4/cam4-famip-30m-timestep_4",
        type=str,
    )
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--model_type", default="baseline", type=str)
    parser.add_argument("--mode", default="train",type=str)
    parser.add_argument("--ckpt",default=None,type=str)
    parser.add_argument("--hidden_size", default=512, type = int)
    parser.add_argument("--lr", default=.001, type = float)
    parser.add_argument("--num_layers", default=7, type = int)
    parser.add_argument("--batch_size", default=10 * 96 * 144, type = int)
    parser.add_argument("--dropout", default=.01, type = float)

    args = parser.parse_args()

    if args.model_type == "baseline":

        model_config = {
            "model_type": "fcn",
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "dropout" : args.dropout
            #   "num_output_layers": 6
        }
    elif args.model_type == "memory":
        model_config = {
            "model_type": "convnet1d",
            "num_layers": 7,
            #   "num_output_layers": 6
        }
    elif args.model_type == "conv1d":
        model_config = {
            "model_type": "conv1d",
            "num_layers": 7,
            "hidden_size": 128, 
            #   "num_output_layers": 6
        }
    else:
        raise ValueError

    main(
        args.mode,
        trainer_params=default_trainer_params(gpus=[args.gpu], precision=16),
        dataset_params=default_dataset_params(
            base=args.dataset, batch_size=args.batch_size
        ),
        model_params=default_model_params(
            ignore_input_variables=None,#[args.input_var_ignore],
            lr=args.lr,
            use_output_scaling=False,
            replace_std_with_range=False,
            model_config=model_config,
            ckpt = args.ckpt,
        ),
    )


# print(input_variables_ablation, len(input_variables_ablation))
# print(inputs, len(inputs))

# to_do = set(inputs) - set(input_variables_ablation)
# print(list(to_do))

# ['FSNT', 'Z3', 'PSL', 'FLNT', 'SOLIN']

# N = 3
# i = 3
# for v in ['FSNT', 'Z3', 'PSL']:#, 'FLNT', 'SOLIN']:
# for v in [ 'FLNT', 'SOLIN']:
#     main(
#         "train",
#         trainer_params=default_trainer_params(gpus=[3], precision=16),
#         dataset_params=default_dataset_params(base=base, batch_size=10 * 96 * 144),
#         model_params=default_model_params(
#             ignore_input_variables=[v],
#             lr=1e-3,
#             use_output_scaling=False,
#             replace_std_with_range=False,
#             model_config=model_config,
#         ),
#     )

# version 10 fcn_history omega
# version 11 fcn

# main("predict,results",
#   trainer_params=default_trainer_params(gpus=[3], precision=16),
#   dataset_params=default_dataset_params(base=base, batch_size=10 * 96 * 144),
#   model_params=default_model_params(ckpt ="lightning_logs/version_21")
#   )


# for i in [0,1,2,3,4]:
#     model_dir = f"/proj/gaia-climate/team/kirill/gaia-surrogate/lightning_logs/version_{i}"


#     if not os.path.exists(os.path.join(model_dir, "predictions.pt")):
#         logger.info(f"predicting {model_dir}")
#         main("predict", trainer_params = default_trainer_params(gpus=[1],precision=16),
#                     dataset_params = default_dataset_params(),
#                     model_params =   default_model_params(ckpt =model_dir))

#     if not os.path.exists(os.path.join(model_dir, "results.pt")):
#         logger.info(f"processing results {model_dir}")
#         process_results(model_dir)

# process_results("/proj/gaia-climate/team/kirill/gaia-surrogate/lightning_logs/version_0")#, naive_memory = True)

# predict(gpus=[3])
# import shutil


# for v in [0]:
#     plot_results(f"lightning_logs/version_{v}")
#     if v == 0:
#         shutil.copy(f"lightning_logs/version_{v}/plots_naive.html", "docs/results/spcam/plots_naive.html")
#     elif v == 1:
#         shutil.copy(f"lightning_logs/version_{v}/plots.html", "docs/results/spcam/plots_baseline.html")
#     # elif v == 2:
#     #     shutil.copy(f"lightning_logs/version_{v}/plot.html", "docs/results/spcam/plots_naive.html")


if __name__ == "__main__":
    run()
