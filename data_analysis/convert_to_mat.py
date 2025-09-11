import scipy.io as sio
import numpy as np
import datetime
import glob
import importlib
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib.pyplot as plt
import pandas as pd
from spacepy.pycdf import CDF as cdf

importlib.reload(lexi_functions)


input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time_left": "2025-03-06T15:00:00Z",
    "end_time_left": "2025-03-06T16:30:00Z",
    "start_time_right": "2025-03-08T11:50:00Z",
    "end_time_right": "2025-03-09T00:00:00Z",
    # "delta_time_left": 1200,
    "delta_time_right": 600,
    "bins": 100,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
    "save_arrays": True,
    "save_folder": "../data/histogram_data/",
    "save_name": "histogram_data",
    "plot_histogram": True,
}


df_left = lexi_functions.read_all_data_files(
    file_list=None,
    start_time=input_dict["start_time_left"],
    end_time=input_dict["end_time_left"],
    return_data_type="dataframe",
    kwargs={
        "data_folder_location": "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/",
        "start_time": input_dict["start_time_left"],
        "end_time": input_dict["end_time_left"],
    },
)

mat_data = {}
mat_data = {​​"data" df_left}​​
sio.savemat('TEST1.mat', mat_data)

