import os

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import glob
import numpy as np
import pytube


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


# compute_stats()
# run(flatten = False, gpus=[3], lr = 1e-3, num_layers=3)
# construct_data()
# run(flatten = False, gpus=[1])


# base_lr = 1e-4
# base_batch_size = 1024
# batch_size = 32768
# lr = float(base_lr*np.sqrt(2)**(batch_size/base_batch_size-1))

# run(gpus=[4], lr = 1e-3, num_layers=7, batch_size=24*96*144,optimizer="lamb")

ckpt = "lightning_logs/version_3/checkpoints/epoch=999-step=78999.ckpt"

w = np.zeros(53)
w[:5] = 1.
w = w.tolist()

# model_config={
#             "model_type": "fcn",
#             "num_layers": 1,
#         }

main("train", trainer_params = default_trainer_params(gpus=[5],precision=16),
              dataset_params = default_dataset_params(subsample_factor=8, batch_size = 24*96*144, seperate_val_set=False, flatten=True),
              model_params =   default_model_params(lr = 1e-3, use_output_scaling=False, replace_std_with_range = False, loss_output_weights = w))

# ckpt = "lightning_logs/version_0/checkpoints/epoch=999-step=629999.ckpt"
# ckpt = "lightning_logs_old/version_12/checkpoints/epoch=854-step=478799.ckpt"
# ckpt = "lightning_logs_old/version_2/checkpoints/epoch=224-step=141299.ckpt"

# model_config = {
#     "input_size": 130,
#     "model_type": "conv",
#     "output_size": 79,
#     "num_layers": 7,
# }

# main(
#     "test",
#     trainer_params=default_trainer_params(gpus=[7],precision=16),
#     dataset_params=default_dataset_params(
#         # batch_size=24,
#         subsample_factor=8, batch_size = 24*96*144,
#         interleave=True,
#         # outputs=["PTEQ", "PTTEND", "PRECT", "TTEND_TOT"],
#     ),
#     model_params=default_model_params(
#         ckpt=ckpt,
#         # model_config=model_config,data_stats = None,
#     ),
# )


# model_params

# main(
#     "predict",
#     trainer_params=default_trainer_params(gpus=[1]),
#     dataset_params=default_dataset_params(
#         subsample_factor=1, batch_size=24*96*144, flatten=False
#     ),
#     model_params=dict(
#         # model_config={
#         #     "input_size": 130,
#         #     "model_type": "conv",
#         #     "output_size": 79,
#         #     "num_layers": 7,
#         # },
#         # ckpt="lightning_logs/version_2/checkpoints/epoch=224-step=141299.ckpt",
#         ckpt="lightning_logs/version_12/checkpoints/epoch=582-step=326479.ckpt",
#     ),
# )


# predict(gpus=[3])
