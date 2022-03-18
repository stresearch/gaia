import json
from grpc import local_server_credentials
import holoviews as hv
import yaml
import os
import torch
from gaia import get_logger
from gaia.plot import levels, lats, lons

import numpy as np
import pandas as pd

logger = get_logger(__name__)
import panel as pn
from gaia import LAND_FILE

def get_skill_ave(y,yhat,reduce_dims = [0, 3]):
    y_var = y.var(reduce_dims, unbiased=False)
    mse = (y - yhat).square().mean(reduce_dims)
    skill = (1.0 - mse / y_var).clamp(0, 1)

    return dict(y_var = y_var,mse=mse, skill = skill )




def process_results(model_dir, lons = lons, lats  =lats, levels = levels, naive_memory = False):
    yaml_file = os.path.join(model_dir, "hparams.yaml")
    params = yaml.unsafe_load(open(yaml_file))

    test_data = torch.load(params["dataset_params"]["test"]["dataset_file"])["y"]
    if len(test_data.shape) == 5:
        # predictions = test_data[:,0,...]
        if naive_memory:
            predictions = test_data[:,0,...]
        test_data = test_data[:, 1, ...]  # keep the second time step
        

    output_map = params["output_index"]
    if not naive_memory:
        predictions = torch.load(os.path.join(model_dir, "predictions.pt"))

    

    results = dict()

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

    if naive_memory:
        torch.save(results, os.path.join(model_dir, "results_naive.pt"))
    else:
        torch.save(results, os.path.join(model_dir, "results.pt"))



