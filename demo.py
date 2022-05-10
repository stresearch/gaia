import streamlit as st
import torch
import pandas
# import holoviews as hv
import os
import sys
import xarray
import hvplot.xarray

# sys.path.append("../../gaia-surrogate")
# from gaia.plot import get_land_outline, get_land_polies
# from gaia.plot import lats,levels26,lons
import plotly.express as px
# import plotly.graph_objects as go
import torch
import numpy as np
# hv.extension("plotly")



@st.cache(allow_output_mutation=True)
def load_data():
    return xarray.open_dataarray("/proj/gaia-climate/team/kirill/experiments/make_movie/saved_on_disk.nc")



st.title("hello")


data = load_data()

time = st.slider("day",0,364,0)
start = st.button("start")

day = st.empty()
plot  = st.empty()


zmin,zmax = (0,2e-7)

if start:
    for time in range(365):
        fig = px.imshow(data.sel(day = time,output = "PRECC_00", model = "cam4_sim"),zmin = zmin, zmax = zmax, x = data.lon, y = data.lat, color_continuous_scale='blues')
        day.header(time)
        plot.plotly_chart(fig, use_container_width=True)
else:
    fig = px.imshow(data.sel(day = time,output = "PRECC_00", model = "cam4_sim"),zmin = zmin, zmax = zmax,x = data.lon, y = data.lat, color_continuous_scale='blues')
    day.header(time)
    plot.plotly_chart(fig, use_container_width=True)
    





