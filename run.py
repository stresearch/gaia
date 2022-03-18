
import os

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import glob
import numpy as np
from gaia.evaluate import process_results
from gaia import get_logger

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




model_config={
              "model_type": "fcn_history",
              "num_layers": 7,
             }



# main("train", trainer_params = default_trainer_params(gpus=[6],precision=16),
#               dataset_params = default_dataset_params(),
#               model_params =   default_model_params(memory_variables = ["PTEQ"],lr = 1e-3, use_output_scaling=False, replace_std_with_range = False, model_config = model_config))



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

process_results("/proj/gaia-climate/team/kirill/gaia-surrogate/lightning_logs/version_0", naive_memory = True)

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