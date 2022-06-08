import json
from grpc import local_server_credentials
import holoviews as hv
import yaml
import os
import torch
from gaia import get_logger
from gaia.plot import levels as LEVELS, levels26 as LEVELS26, lats, lons

import numpy as np
import pandas as pd

logger = get_logger(__name__)
import panel as pn
from gaia import LAND_FILE

def get_skill_ave(y,yhat,reduce_dims = [0, 3], weights = None):
    y_var = y.var(reduce_dims, unbiased=False)
    mse = (y - yhat).square().mean(reduce_dims)
    skill = (1.0 - mse / y_var).clamp(0, 1)

    return dict(y_var = y_var,
                yhat_var = yhat.var(reduce_dims, unbiased=False), 
                mse=mse,
                skill = skill,
                y_mean =  y.mean(reduce_dims),
                y_hat_mean = yhat.mean(reduce_dims))




def process_results(model_dir, lons = lons, lats  = lats, levels = None, naive_memory = False, other_predictions = None):
    yaml_file = os.path.join(model_dir, "hparams.yaml")
    params = yaml.unsafe_load(open(yaml_file))

    if levels is None:
        if "spcam" in params["dataset_params"]["test"]["dataset_file"]:
            logger.info("setting levels to 30")
            levels = LEVELS
        else:
            logger.info("setting levels to 26")
            levels = LEVELS26


    test_data = torch.load(params["dataset_params"]["test"]["dataset_file"])["y"]
    if len(test_data.shape) == 5:
        # predictions = test_data[:,0,...]
        if naive_memory:
            predictions = test_data[:,0,...]
        test_data = test_data[:, 1, ...]  # keep the second time step
        

    


    output_map = params["output_index"]

    

    if not naive_memory:
        predictions = torch.load(os.path.join(model_dir, "predictions.pt"))

    loss_output_weights = params.get("loss_output_weights")
    if loss_output_weights:
        loss_output_weights = torch.tensor(loss_output_weights)
    else:
        loss_output_weights = torch.ones(predictions.shape[1])

    results = dict()

    if other_predictions:
        #make the main predictions "truth"
        test_data = predictions
        predictions = torch.load(other_predictions)


    for k, v in output_map.items():
        start, end = v
        y = test_data[:, start:end, ...]
        yhat = predictions[:, start:end, ...]

        results[k+"_zonal_ave"] = get_skill_ave(y,yhat,reduce_dims = [0,3])
        results[k+"_global_ave"] = get_skill_ave(y,yhat,reduce_dims = [0,2,3])
        
        if end - start == 1:
            # for scalars
            results[k+"_global"] = get_skill_ave(y,yhat,reduce_dims = [0,1])

            #move lons around
            lons = np.array(lons)  
            lons[lons>180] = lons[lons>180]-360
            new_index = lons.argsort()
            lons = lons[new_index]
            for ki in list(results[k+"_global"].keys()):
                results[k+"_global"][ki] = results[k+"_global"][ki][:,new_index]
            lons = lons.tolist()


    results["levels"] = levels
    results["lats"] = lats
    results["lons"] = lons

    file_name = "results.pt"

    if other_predictions:
        file_name = "results_with_other.pt"

    if naive_memory:
        torch.save(results, os.path.join(model_dir, "results_naive.pt"))
    else:
        torch.save(results, os.path.join(model_dir, file_name))



