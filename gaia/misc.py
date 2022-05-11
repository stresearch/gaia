from sympy import maximum
import torch
import tqdm.auto as tqdm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from gaia.plot import get_land_polies
import io
import PIL
import imageio_ffmpeg
from gaia import get_logger

logger = get_logger(__name__)


def get_polies():
    polies = get_land_polies()
    combined_poly = []
    for p in polies:
        combined_poly.append(p)
        combined_poly.append(np.zeros((1,2))*np.nan)
    
    combined_poly = np.concatenate(combined_poly,axis=0)
    
    land = go.Scatter(x= combined_poly[:,0],\
                          y = combined_poly[:,1],line=dict(color= "black",simplify = True, width = 1),marker = dict(size = 0.1))
    
    return land


def make_figure_image_arrays(image_data, x, y, ts, land=None,  cmap_name="blues", title="", zmin=None, zmax=None, width = 2*480, height = 2*384):
    arrays = []
    for t in tqdm.tqdm(ts):
        fig  = px.imshow(image_data[t],
                         zmin  = zmin, zmax =zmax ,x = x, title = f"day:{int(t):03}, {title}", labels = {"x" : "lon", "y" : "lat"}, aspect = "auto",
                         y = y,origin = "lower", color_continuous_scale = cmap_name, width = width, height = height)
        if land:
            fig.add_traces(land)
            
        fig.update_layout(template = "plotly_white")


        b = io.BytesIO()
        fig.write_image(b)
        arrays.append(np.array(PIL.Image.open(b).convert("RGB"))[None,...])
        
    return np.concatenate(arrays)


def make_video(list_of_pil_images, use_numpy=False, fps=30, file_name="temp_video.mp4"):
    if use_numpy:
        size = list_of_pil_images[0].shape[:2][::-1]
    else:
        size = list_of_pil_images[0].size
        
    writer = imageio_ffmpeg.write_frames(
        file_name, size, fps=fps
    )  # size is (width, height)
    writer.send(None)  # seed the generator
    for frame in tqdm.tqdm(list_of_pil_images):
        if use_numpy:
            writer.send(np.array(frame))
        else:
            writer.send(frame)
    writer.close()


def make_video_for_two_model(model1, model2, output, file_name = "temp.mp4", symmetric = False, samples_per_day = 2):

    model1_name, model1_file = model1
    model2_name, model2_file = model2
    output_name, output_index = output

    def load_file(file):
        temp = torch.load(file)

        if isinstance(temp, dict):
            return temp["y"][:,1,...]

        else:
            return temp


    y1 = load_file(model1_file)[:,output_index,:,:]
    y2 = load_file(model2_file)[:,output_index,:,:]

    assert y1.shape == y2.shape

    # var_index = torch.load(var_index_file)["output_index"]
    # var_names = [f"{k}_{i:02}" if e-s > 1 else k for k,(s,e) in var_index["output_index"].items() for i in range(e-s)]


    from gaia.plot import lats, lons

    lats = np.array(lats)
    lons = np.array(lons)
    lons[lons > 180] = lons[lons > 180] - 360
    new_index = lons.argsort()
    lons = lons[new_index]

    y1 = y1[...,new_index]
    y2 = y2[...,new_index]

    land = get_polies()


    figures = dict()

    # plot outputs

    

    cmap_name = "blues" if not symmetric else "RdBu"
    

    ## reduce to 1 per day
    temp = y1.reshape(-1, samples_per_day, 96, 144).mean(1)
    mn = temp.mean()
    st = 3*temp.std()
    zmin =  (mn - st).clip(min =temp.min()).item()
    zmax =  (mn + st).clip(max = temp.max()).item()

    if symmetric:
        zmax = max([np.abs(zmax), np.abs(zmin)])
        zmin = -zmax

    days = range(temp.shape[0])

    temp = temp.numpy()

    title = f"model:{model1_name}, output:{output_name}"
    figures[f"{model1_name}-{output_name}"] = make_figure_image_arrays(temp, x = lons, y = lats, ts =days,
                                    land = land, zmin = zmin, zmax = zmax, title = title, cmap_name=cmap_name)


    temp = y2.reshape(-1, samples_per_day, 96, 144).mean(1).numpy()
    title = f"model:{model2_name}, output:{output_name}"
    figures[f"{model2_name}-{output_name}"] = make_figure_image_arrays(temp, x = lons, y = lats, ts =days,
                                    land = land, zmin = zmin, zmax = zmax, title = title,cmap_name=cmap_name)



    # plot abs error
    abs_err = (y1 - y2).abs().reshape(-1, samples_per_day, 96, 144).mean(1)
    zmax = (abs_err.mean() + 3*abs_err.std()).clip(max = abs_err.max()).item()
    temp = abs_err.numpy()

    title = f"output:{output_name}, abs_error({model1_name},{model2_name})"

    figures[f"abs_error-{model1_name}-{model2_name}-{output_name}"] = make_figure_image_arrays(temp, x = lons, y = lats, ts =days,
                                    land = land, zmin = 0, zmax = zmax, title = title, cmap_name="Reds")


    # plot abs perc error
    mean_per_loc = abs_err.mean(dim=[0],keepdims = True).clip(min = 1e-10)
    temp = (abs_err / mean_per_loc).clip(max = 1.).numpy()

    title = f"output:{output_name}, abs_error_perc({model1_name},{model2_name})"
    figures[f"abs_error_perc-{model1_name}-{model2_name}-{output_name}"] = make_figure_image_arrays(temp, x = lons, y = lats, ts =days,
                                    land = land, zmin = 0, zmax = 1, title = title, cmap_name="Oranges")

    
    make_video(figures[f"abs_error_perc-{model1_name}-{model2_name}-{output_name}"], use_numpy=True, fps = 10, file_name = file_name.replace(".mp4","-abs_error_perc.mp4"))

    # skill

    temp = (1. - (y1 - y2).square().reshape(-1, samples_per_day, 96, 144).mean(1) / y1.var(dim = [0],keepdims = True)).clip(0,1).numpy()

    title = f"output:{output_name}, skill-temporal({model1_name},{model2_name})"

    figures[f"skill-{model1_name}-{model2_name}-{output_name}"] = make_figure_image_arrays(temp, x = lons, y = lats, ts =days,
                                    land = land, zmin = 0, zmax = 1, title = title, cmap_name="Greens")


    

    row1 = np.concatenate([figures[f"{model1_name}-{output_name}"], figures[f"{model2_name}-{output_name}"]],axis = 2)
    row2 = np.concatenate([figures[f"abs_error-{model1_name}-{model2_name}-{output_name}"],
                           figures[f"skill-{model1_name}-{model2_name}-{output_name}"]], axis = 2)
    grid = np.concatenate([row1,row2], axis = 1)

    make_video(grid, use_numpy=True, fps = 10, file_name = file_name)















        








