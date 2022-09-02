import json
import holoviews as hv
import yaml
import os
import torch
from gaia import get_logger
import tqdm.auto as tqdm
import numpy as np
import pandas as pd

logger = get_logger(__name__)
import panel as pn
from gaia import LAND_FILE


hv.extension("bokeh")
pn.extension()

lons = [
    0.0,
    2.5,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    17.5,
    20.0,
    22.5,
    25.0,
    27.5,
    30.0,
    32.5,
    35.0,
    37.5,
    40.0,
    42.5,
    45.0,
    47.5,
    50.0,
    52.5,
    55.0,
    57.5,
    60.0,
    62.5,
    65.0,
    67.5,
    70.0,
    72.5,
    75.0,
    77.5,
    80.0,
    82.5,
    85.0,
    87.5,
    90.0,
    92.5,
    95.0,
    97.5,
    100.0,
    102.5,
    105.0,
    107.5,
    110.0,
    112.5,
    115.0,
    117.5,
    120.0,
    122.5,
    125.0,
    127.5,
    130.0,
    132.5,
    135.0,
    137.5,
    140.0,
    142.5,
    145.0,
    147.5,
    150.0,
    152.5,
    155.0,
    157.5,
    160.0,
    162.5,
    165.0,
    167.5,
    170.0,
    172.5,
    175.0,
    177.5,
    180.0,
    182.5,
    185.0,
    187.5,
    190.0,
    192.5,
    195.0,
    197.5,
    200.0,
    202.5,
    205.0,
    207.5,
    210.0,
    212.5,
    215.0,
    217.5,
    220.0,
    222.5,
    225.0,
    227.5,
    230.0,
    232.5,
    235.0,
    237.5,
    240.0,
    242.5,
    245.0,
    247.5,
    250.0,
    252.5,
    255.0,
    257.5,
    260.0,
    262.5,
    265.0,
    267.5,
    270.0,
    272.5,
    275.0,
    277.5,
    280.0,
    282.5,
    285.0,
    287.5,
    290.0,
    292.5,
    295.0,
    297.5,
    300.0,
    302.5,
    305.0,
    307.5,
    310.0,
    312.5,
    315.0,
    317.5,
    320.0,
    322.5,
    325.0,
    327.5,
    330.0,
    332.5,
    335.0,
    337.5,
    340.0,
    342.5,
    345.0,
    347.5,
    350.0,
    352.5,
    355.0,
    357.5,
]

lats = [
    -90.0,
    -88.10526315789474,
    -86.21052631578948,
    -84.3157894736842,
    -82.42105263157895,
    -80.52631578947368,
    -78.63157894736842,
    -76.73684210526316,
    -74.84210526315789,
    -72.94736842105263,
    -71.05263157894737,
    -69.15789473684211,
    -67.26315789473685,
    -65.36842105263158,
    -63.473684210526315,
    -61.578947368421055,
    -59.684210526315795,
    -57.78947368421053,
    -55.89473684210527,
    -54.0,
    -52.10526315789474,
    -50.21052631578947,
    -48.31578947368421,
    -46.42105263157895,
    -44.526315789473685,
    -42.631578947368425,
    -40.73684210526316,
    -38.8421052631579,
    -36.94736842105264,
    -35.05263157894737,
    -33.15789473684211,
    -31.263157894736842,
    -29.368421052631582,
    -27.473684210526322,
    -25.578947368421055,
    -23.684210526315795,
    -21.789473684210535,
    -19.89473684210526,
    -18.0,
    -16.10526315789474,
    -14.21052631578948,
    -12.31578947368422,
    -10.421052631578945,
    -8.526315789473685,
    -6.631578947368425,
    -4.736842105263165,
    -2.8421052631579045,
    -0.9473684210526301,
    0.9473684210526301,
    2.8421052631578902,
    4.73684210526315,
    6.631578947368411,
    8.526315789473685,
    10.421052631578945,
    12.315789473684205,
    14.210526315789465,
    16.105263157894726,
    18.0,
    19.89473684210526,
    21.78947368421052,
    23.68421052631578,
    25.57894736842104,
    27.473684210526315,
    29.368421052631575,
    31.263157894736835,
    33.157894736842096,
    35.052631578947356,
    36.94736842105263,
    38.84210526315789,
    40.73684210526315,
    42.63157894736841,
    44.52631578947367,
    46.42105263157893,
    48.31578947368419,
    50.21052631578948,
    52.10526315789474,
    54.0,
    55.89473684210526,
    57.78947368421052,
    59.68421052631578,
    61.57894736842104,
    63.4736842105263,
    65.36842105263156,
    67.26315789473682,
    69.15789473684211,
    71.05263157894737,
    72.94736842105263,
    74.84210526315789,
    76.73684210526315,
    78.63157894736841,
    80.52631578947367,
    82.42105263157893,
    84.31578947368419,
    86.21052631578945,
    88.10526315789474,
    90.0,
]

levels = [
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
]

levels26 = [
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


def get_levels(dataset):
    if "cam4" in dataset:
        return levels26
    else:
        return levels


def get_land_polies():
    polys = [
        np.array(p[0])
        for p in pd.read_json(LAND_FILE)["features"].apply(
            lambda a: a["geometry"]["coordinates"]
        )
    ]

    return polys


def get_land_outline():
    polys = [
        np.array(p[0])
        for p in pd.read_json(LAND_FILE)["features"].apply(
            lambda a: a["geometry"]["coordinates"]
        )
    ]
    polys = [dict(lat=p[:, 1], lon=p[:, 0]) for p in polys]
    return hv.Polygons(polys, kdims=["lon", "lat"]).opts(
        fill_color=None, line_color="orange"
    )


def plot_skill_at_levels(y, yhat, lats, levels, var_name=""):
    reduce_dims = [0, 3]
    y_var = y.var(reduce_dims, unbiased=False)
    mse = (y - yhat).square().mean(reduce_dims)
    skill = (1.0 - mse / y_var).clamp(0, 1)
    # rmse = mse.sqrt()
    # std = y_var.sqrt()

    width = height = 500

    opts = dict(
        colorbar=True,
        width=width,
        height=height,
        axiswise=True,
        invert_yaxis=True,
        tools=["hover"],
    )

    skill_plot = (
        hv.Image(
            (lats, levels, skill),
            kdims=["lat", "level"],
            vdims=["skill"],
            label=f"skill(y,yhat): {var_name}",
        )
        .opts(**opts)
        .redim.range(skill=(0, 1.0))
    )
    rmse_plot = hv.Image(
        (lats, levels, mse),
        kdims=["lat", "level"],
        vdims=["mse"],
        label=f"mse(y,yhat): {var_name}",
    ).opts(cmap="reds", **opts)
    std_plot = hv.Image(
        (lats, levels, y_var),
        kdims=["lat", "level"],
        vdims=["var"],
        label=f"var(y): {var_name}",
    ).opts(cmap="greens", **opts)

    return skill_plot + rmse_plot + std_plot


def plot_skill_scalar(y, yhat, lats, var_name=""):
    reduce_dims = [0, 1, 3]
    y_var = y.var(reduce_dims, unbiased=False)
    mse = (y - yhat).square().mean(reduce_dims)
    skill = (1.0 - mse / y_var).clamp(0, 1)
    # rmse = mse.sqrt()
    # std = y_var.sqrt()

    width = height = 500

    opts = dict(
        width=width, height=height, axiswise=True, line_width=2, tools=["hover"]
    )

    skill_plot = (
        hv.Curve(
            (lats, skill),
            kdims=["lat"],
            vdims=["skill"],
            label=f"skill(y,yhat): {var_name}",
        )
        .opts(**opts)
        .redim.range(skill=(0, 1.0))
    )
    rmse_plot = hv.Curve(
        (lats, mse), kdims=["lat"], vdims=["mse"], label=f"mse(y,yhat): {var_name}"
    ).opts(axiswise=True, **opts)
    std_plot = hv.Curve(
        (lats, y_var), kdims=["lat"], vdims=["var"], label=f"var(y): {var_name}"
    ).opts(axiswise=True, **opts)

    return skill_plot + rmse_plot + std_plot


def plot_skill_scalar_world(y, yhat, lons, lats, var_name=""):

    # make it so we can use land outlines
    lons = np.array(lons)
    lons[lons > 180] = lons[lons > 180] - 360
    new_index = lons.argsort()
    lons = lons[new_index]

    y = y[:, :, :, new_index]
    yhat = yhat[:, :, :, new_index]

    reduce_dims = [0, 1]
    y_var = y.var(reduce_dims, unbiased=False)
    mse = (y - yhat).square().mean(reduce_dims)
    skill = (1.0 - mse / y_var).clamp(0, 1)
    # rmse = mse.sqrt()
    # std = y_var.sqrt()

    width = height = 500

    opts = dict(
        colorbar=True, width=width, height=height, axiswise=True, tools=["hover"]
    )

    skill_plot = (
        hv.Image(
            (lons, lats, skill),
            kdims=["lon", "lat"],
            vdims=["skill"],
            label=f"skill(y,yhat): {var_name}",
        )
        .opts(**opts)
        .redim.range(skill=(0, 1.0))
    )
    rmse_plot = hv.Image(
        (lons, lats, mse),
        kdims=["lon", "lat"],
        vdims=["mse"],
        label=f"mse(y,yhat): {var_name}",
    ).opts(cmap="reds", **opts)
    std_plot = hv.Image(
        (lons, lats, y_var),
        kdims=["lon", "lat"],
        vdims=["var"],
        label=f"var(y): {var_name}",
    ).opts(cmap="greens", **opts)

    outline = get_land_outline()

    return skill_plot * outline + rmse_plot + std_plot


def plot_results(model_dir):

    yaml_file = os.path.join(model_dir, "hparams.yaml")
    params = yaml.unsafe_load(open(yaml_file))

    test_data = torch.load(params["dataset_params"]["test"]["dataset_file"])["y"]
    if len(test_data.shape) == 5:
        # predictions = test_data[:,0,...]
        test_data = test_data[:, 1, ...]  # keep the second time step

    output_map = params["output_index"]
    predictions = torch.load(os.path.join(model_dir, "predictions.pt"))

    results = json.load(open(os.path.join(model_dir, "test_results.json")))
    # results = dict()

    plots = dict()

    for k, v in output_map.items():
        start, end = v
        y = test_data[:, start:end, ...]
        yhat = predictions[:, start:end, ...]
        if end - start > 1:
            # plot levels
            plots[k] = plot_skill_at_levels(y, yhat, lats, levels, k)
        else:
            plots[k] = plot_skill_scalar_world(y, yhat, lons, lats, k)

    combined_plot = pn.Tabs(
        *[(k, p) for k, p in plots.items()]
        + [("results", results)]
        + [("params", params)]
    )
    combined_plot.save(os.path.join(model_dir, "plots_naive.html"))


def save_diagnostic_plot(
    model_dir,
    plot_types=["skill_vector", "skill_scalar", "skill_global", "correlation"],
    dataset = None
):

    import sys

    # sys.path.append("../../gaia-surrogate/")
    import torch
    import pandas as pd
    import holoviews as hv
    import numpy as np
    from gaia.training import load_hparams_file, get_dataset
    from pathlib import Path
    from gaia.plot import lats, lons
    from gaia.data import unflatten_tensor
    from gaia.config import get_levels
    import glob
    import panel as pn

    pn.extension()
    hv.extension("bokeh")

    plots_path = Path(model_dir) / "plots"
    plots_path.mkdir(exist_ok=True)

    # model_dir = "../../gaia-surrogate/lightning_logs_intergration/version_0/"
    params = load_hparams_file(model_dir)
    weights = (torch.tensor(params["loss_output_weights"]) > 0).float()
    yhat = torch.load(next(Path(model_dir).glob(f"predictions_{dataset}.pt")))
    yhat = yhat * weights[None, :, None, None]
    y, ydict = get_dataset(
        **params["dataset_params"]["test"], model_grid=params.get("model_grid", None)
    )
    y = unflatten_tensor(y["y"])
    output_index = params["output_index"]
    input_index = params["input_index"]
    levels = params.get("model_grid", None)
    if levels is None:
        levels = get_levels(params["dataset_params"]["dataset"])

    # import xarray as xr
    # temp  = xr.load_dataset("/proj/gaia-climate/data/cam4_v2/rF_AMIP_CN_CAM4--ctrl-daily-output-tstep.cam2.h1.1979-01-01-00000.nc")

    scalar_output_variables = [k for k, v in output_index.items() if v[1] - v[0] == 1]
    vector_output_variables = [k for k, v in output_index.items() if v[1] - v[0] > 1]

    if "skill_vector" in plot_types:
        logger.info("generating plots of skill for vector variables")

        quad_mesh_opts = hv.opts.QuadMesh(
            width=350,
            height=300,
            tools=["hover"],
            invert_yaxis=True,
            cformatter="%.1e",
            colorbar=True,
        )

        def plot_mean(var, kind):
            s, e = output_index[var]
            if kind == "simulation":
                temp = y[:, s:e, :, :].mean([0, 3]).numpy()
            else:
                temp = yhat[:, s:e, :, :].mean([0, 3]).numpy()

            # find robust range

            return hv.QuadMesh(
                (lats, levels, temp), ["lats", "levels"], [f"{var}_mean"], label="mean"
            ).opts(symmetric=True, cmap="RdBu")

        def plot_std(var, kind):
            s, e = output_index[var]
            if kind == "simulation":
                temp = y[:, s:e, :, :].std([0, 3]).numpy()
            else:
                temp = yhat[:, s:e, :, :].std([0, 3]).numpy()

            # find robust range

            return hv.QuadMesh(
                (lats, levels, temp), ["lats", "levels"], [f"{var}_std"], label="std"
            ).opts(symmetric=False, cmap="Blues")

        def plot_metrics(var, metric):

            s, e = output_index[var]

            mse = (y[:, s:e, :, :] - yhat[:, s:e, :, :]).square().mean([0, 3])
            vr = y[:, s:e, :, :].var([0, 3], unbiased=False)
            skill = (1 - mse / vr).clip(0, 1).numpy()

            # find robust range

            mse = mse.numpy()

            mse_max = mse.mean() + 3 * mse.std()

            # return skill

            if metric == "skill":
                return (
                    hv.QuadMesh(
                        (lats, levels, skill), ["lats", "levels"], [f"{var}_skill"]
                    )
                    .opts(symmetric=False, cmap="Greens")
                    .redim.range(**{f"{var}_skill": (0, 1)})
                )
            else:
                return (
                    hv.QuadMesh((lats, levels, mse), ["lats", "levels"], [f"{var}_mse"])
                    .opts(symmetric=False, cmap="Oranges", logz=False)
                    .redim.range(**{f"{var}_mse": (0, mse_max)})
                )

        p_mean = hv.DynamicMap(plot_mean, kdims=["variable", "kind"]).redim.values(
            variable=vector_output_variables, kind=["simulation", "surrogate"]
        )
        p_mean = p_mean.layout("kind").opts(quad_mesh_opts)

        p_std = hv.DynamicMap(plot_std, kdims=["variable", "kind"]).redim.values(
            variable=vector_output_variables, kind=["simulation", "surrogate"]
        )
        p_std = p_std.layout("kind").opts(quad_mesh_opts)

        # (p_mean + p_std).cols(1)

        p_metrics = hv.DynamicMap(
            plot_metrics, kdims=["variable", "metric"]
        ).redim.values(variable=vector_output_variables, metric=["skill", "mse"])
        p_metrics = p_metrics.layout("metric").opts(quad_mesh_opts)

        # p_metrics

        combined = (p_mean + p_std + p_metrics).cols(1)
        # combined

        p_pane_vector_valued = pn.pane.HoloViews(combined, widget_location="top")
        # p_pane_vector_valued
        p_pane_vector_valued.save(
            plots_path / "stats_and_metrics_vector_variables.html",
            "Vector Valued Outputs",
            max_opts=100,
            embed=True,
        )

    if "skill_scalar" in plot_types:
        logger.info("generating plots of skill for scalar variables")

        curve_opts = hv.opts.Curve(
            width=400, height=300, tools=["hover"], yformatter="%.1e"
        )

        def plot_mean_scale(var, kind):
            s, e = output_index[var]
            if kind == "simulation":
                temp = y[:, s:e, :, :].mean([0, 3]).numpy().ravel()
                temp_std = y[:, s:e, :, :].std([0, 3]).numpy().ravel()
            else:
                temp = yhat[:, s:e, :, :].mean([0, 3]).numpy().ravel()
                temp_std = yhat[:, s:e, :, :].std([0, 3]).numpy().ravel()

            # find robust range

            return hv.Spread(
                (lats, temp, temp_std), ["lats"], [f"{var}_mean", f"{var}_std"]
            ).opts(line_width=0, alpha=0.3) * hv.Curve(
                (lats, temp), ["lats"], [f"{var}_mean"]
            )

        def plot_mse_scale(var, kind):

            s, e = output_index[var]

            mse = (y[:, s:e, :, :] - yhat[:, s:e, :, :]).square().mean([0, 3])
            vr = y[:, s:e, :, :].var([0, 3], unbiased=False)
            skill = (1 - mse / vr).clip(0, 1).numpy().ravel()

            # find robust range

            mse = mse.numpy().ravel()

            mse_max = mse.mean() + 3 * mse.std()

            # find robust range

            if kind == "skill":
                return hv.Curve((lats, skill), ["lats"], [f"{var}_skill"]).redim.range(
                    **{f"{var}_skill": (0, 1)}
                )
            else:
                return hv.Curve((lats, mse), ["lats"], [f"{var}_mse"])

        p_mean = hv.DynamicMap(
            plot_mean_scale, kdims=["variable", "kind"]
        ).redim.values(
            variable=scalar_output_variables, kind=["simulation", "surrogate"]
        )
        p_mean = (
            p_mean.overlay("kind")
            .opts(legend_position="top")
            .layout("variable")
            .opts(curve_opts)
        )

        p_metric = hv.DynamicMap(
            plot_mse_scale, kdims=["variable", "metric"]
        ).redim.values(variable=scalar_output_variables, metric=["skill", "mse"])
        p_metric = p_metric.layout(["variable", "metric"]).opts(curve_opts)

        # p_metric.overlay("variable").opts(legend_position = "right").opts(curve_opts)
        plot_scalar = (p_mean.cols(1) + p_metric.cols(2)).cols(3)

        hv.save(plot_scalar, plots_path / "stats_and_metrics_scalar_variables.html")

    if "correlation" in plot_types:
        logger.info("generating plots of skill for vector variables")

        from holoviews.operation import datashader as ds

        def plot_correlation(var="PRECT", N=10000):

            if var not in scalar_output_variables:
                var2, level = var.split("_")
                level = int(level)
            else:
                level = 0
                var2 = var

            s, e = output_index[var2]
            s += level

            symmetric = "PREC" not in var

            temp = pd.DataFrame(
                dict(
                    yhat=yhat[:, s, :, :].ravel().numpy(),
                    y=y[:, s, :, :].ravel().numpy(),
                )
            )

            try:
                c = temp.y.corr(temp.yhat)
            except:
                c = np.nan

            limit_range = False

            mn = temp.values.mean()
            sd = temp.values.std()

            temp["error_std"] =(temp.y - temp.yhat).abs()/sd


            temp = temp.sample(N, weights = "error_std")


            # mask = ((temp-mn).abs() < 5 * sd).all(1)
            # temp_inliers = temp.loc[mask]
            # temp_inliers.loc[:,"kind"] = "inlier"

                # mask_outliers = (temp.abs() >= ub).any(1)
                # mask_outliers = ((temp-mn).abs() >= 5 * sd).any(1)

               

            rng = temp.loc[:,["y","yhat"]].abs().values.max()

            # mask_outliers = temp.diff(1).abs() >= sd

            if symmetric:
                rng = (-rng, rng)
            else:
                rng = (-rng* 1e-1, rng)

                # temp_outliers = temp.loc[mask_outliers]
                # if len(temp_outliers) > 0:
                #     temp_outliers.loc[:,"kind"] = "outlier"



                # temp = pd.concat([temp_inliers.sample(N), temp_outliers], ignore_index=True)



            return (
                (
                    hv.Points((temp.y, temp.yhat, temp.error_std), kdims = [f"y_{var}", f"yhat_{var}"], vdims = [f"error_std_{var}"]).opts(
                        width=400, height=400, padding=0.05, color = f"error_std_{var}", colorbar=True
                    )
                    * hv.Curve((rng, rng)).opts(
                        line_width=1, color="orange", title=f"corr: {c:.3f}"
                    )
                )
                .redim.range(**{f"y_{var}": rng, f"yhat_{var}": rng})
                .opts(xformatter="%.1e", yformatter="%.1e")
            )

        for v in tqdm.tqdm(vector_output_variables):

            try:
                p = hv.DynamicMap(plot_correlation, kdims=["var"]).redim.values(
                    var=[f"{v}_{i}" for i in range(26)]
                )
                # p.opts(axiswise = False)
                # plot_correlation("PTTEND_21")
                hv.save(
                    p.layout("var").cols(4), plots_path / f"corrs_{v}.html", title=v
                )
            except:
                pass

        p = hv.DynamicMap(plot_correlation, kdims=["var"]).redim.values(
            var=scalar_output_variables
        )
        hv.save(
            p.layout("var").cols(4),
            plots_path / f"corrs_scalar_variables.html",
            title=v,
        )

    if "skill_global" in plot_types:
        logger.info("generating plots for globa skill")

        quad_mesh_opts = hv.opts.QuadMesh(
            width=600,
            height=300,
            tools=["hover"],
            invert_yaxis=False,
            cformatter="%.1e",
            colorbar=True,
        )

        def plot_mean(var, kind):

            if var not in scalar_output_variables:
                var2, level = var.split("_")
                level = int(level)
            else:
                level = 0
                var2 = var

            s, e = output_index[var2]
            s += level

            if kind == "simulation":
                temp = y[:, s, :, :].mean([0]).numpy()
            else:
                temp = yhat[:, s, :, :].mean([0]).numpy()

            # find robust range

            return hv.QuadMesh(
                (lons, lats, temp), ["lons", "lats"], [f"{var}_mean"], label="mean"
            ).opts(symmetric=True, cmap="RdBu")

        def plot_std(var, kind):

            if var not in scalar_output_variables:
                var2, level = var.split("_")
                level = int(level)
            else:
                level = 0
                var2 = var

            s, e = output_index[var2]
            s += level

            if kind == "simulation":
                temp = y[:, s, :, :].std([0]).numpy()
            else:
                temp = yhat[:, s, :, :].std([0]).numpy()

            # find robust range

            return hv.QuadMesh(
                (lons, lats, temp), ["lons", "lats"], [f"{var}_std"], label="std"
            ).opts(symmetric=False, cmap="Blues")

        def plot_metrics(var, metric):

            if var not in scalar_output_variables:
                var2, level = var.split("_")
                level = int(level)
            else:
                level = 0
                var2 = var

            s, e = output_index[var2]
            s += level

            mse = (y[:, s, :, :] - yhat[:, s, :, :]).square().mean([0])
            vr = y[:, s, :, :].var([0], unbiased=False)
            skill = (1 - mse / vr).clip(0, 1).numpy()

            # find robust range

            mse = mse.numpy()

            mse_max = mse.mean() + 3 * mse.std()

            # return skill

            if metric == "skill":
                return (
                    hv.QuadMesh((lons, lats, skill), ["lons", "lats"], [f"{var}_skill"])
                    .opts(symmetric=False, cmap="Greens")
                    .redim.range(**{f"{var}_skill": (0, 1)})
                )
            else:
                return (
                    hv.QuadMesh((lons, lats, mse), ["lons", "lats"], [f"{var}_mse"])
                    .opts(symmetric=False, cmap="Oranges", logz=False)
                    .redim.range(**{f"{var}_mse": (0, mse_max)})
                )

        for var in tqdm.tqdm(vector_output_variables):  # = 'PTTEND'
            # for level_index in tqdm.trange(len(levels)):

            variables = [f"{var}_{i:02}" for i in range(len(levels))]

            p_mean = hv.DynamicMap(plot_mean, kdims=["variable", "kind"]).redim.values(
                variable=variables, kind=["simulation", "surrogate"]
            )
            p_mean = p_mean.layout("kind").opts(quad_mesh_opts)

            p_std = hv.DynamicMap(plot_std, kdims=["variable", "kind"]).redim.values(
                variable=variables, kind=["simulation", "surrogate"]
            )
            p_std = p_std.layout("kind").opts(quad_mesh_opts)

            # (p_mean + p_std).cols(1)

            p_metrics = hv.DynamicMap(
                plot_metrics, kdims=["variable", "metric"]
            ).redim.values(variable=variables, metric=["skill", "mse"])
            p_metrics = p_metrics.layout("metric").opts(quad_mesh_opts)

            # p_metrics

            combined = (p_mean + p_std + p_metrics).cols(1)

            for v in tqdm.tqdm(variables):
                hv.save(
                    combined.select(variable=v),
                    plots_path / f"stats_and_metrics_global_{v}.html",
                    title = f"States and Metrics Global: {v}",
                )

            # p_pane_vector_valued = pn.pane.HoloViews(combined, widget_location="top")

            # p_pane_vector_valued.save(
            # plots_path / f"stats_and_metrics_global_{var}.html",
            # "Vector Valued Outputs",
            # max_opts=100,
            # embed=True,
            # )

        variables = scalar_output_variables

        p_mean = hv.DynamicMap(plot_mean, kdims=["variable", "kind"]).redim.values(
            variable=variables, kind=["simulation", "surrogate"]
        )
        p_mean = p_mean.layout("kind").opts(quad_mesh_opts)

        p_std = hv.DynamicMap(plot_std, kdims=["variable", "kind"]).redim.values(
            variable=variables, kind=["simulation", "surrogate"]
        )
        p_std = p_std.layout("kind").opts(quad_mesh_opts)

        # (p_mean + p_std).cols(1)

        p_metrics = hv.DynamicMap(
            plot_metrics, kdims=["variable", "metric"]
        ).redim.values(variable=variables, metric=["skill", "mse"])
        p_metrics = p_metrics.layout("metric").opts(quad_mesh_opts)

        # p_metrics

        combined = (p_mean + p_std + p_metrics).cols(1)

        for v in tqdm.tqdm(variables):
            hv.save(
                combined.select(variable=v),
                plots_path / f"stats_and_metrics_global_{v}.html",
                title = f"States and Metrics Global: {v}",

            )

        # p_pane_vector_valued = pn.pane.HoloViews(combined, widget_location="top")

        # p_pane_vector_valued.save(
        # plots_path / f"stats_and_metrics_global_scalar_variables.html",
        # "Scalar Valued Outputs",
        # max_opts=100,
        # embed=True,
        # )
