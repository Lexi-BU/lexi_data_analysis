import glob
import importlib
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spacepy.pycdf import CDF as cdf

importlib.reload(lexi_functions)


input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time_left": "2025-03-16T19:45:00Z",
    "end_time_left": "2025-03-16T21:15:00Z",
    "start_time_right": "2025-03-06T13:50:00Z",
    "end_time_right": "2025-03-06T17:00:00Z",
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
# hist_left, hist_right, hist_center, xedges_left, yedges_left, xedges_right, yedges_right = (
#     lexi_functions.get_histogram_arrays(**input_dict)
# )

lexi_functions.get_histogram_arrays(**input_dict)

# lexi_functions.plot_histograms(
#     hist_left=hist_left,
#     hist_right=hist_right,
#     hist_center=hist_center,
#     xedges_left=xedges_left,
#     yedges_left=yedges_left,
#     xedges_right=xedges_right,
#     yedges_right=yedges_right,
#     x_key=input_dict["x_key"],
#     y_key=input_dict["y_key"],
#     start_time_left=input_dict["start_time_left"],
#     end_time_left=input_dict["end_time_left"],
#     start_time_right=input_dict["start_time_right"],
#     end_time_right=input_dict["end_time_right"],
#     delta_time_left=input_dict["delta_time_left"],
#     delta_time_right=input_dict["delta_time_right"],
#     save_folder="../figures/histogram_diff/",
#     color_map="inferno",
#     save_plot=True,
# )

# df_left = lexi_functions.read_all_data_files(
#     file_list=None,
#     start_time=input_dict["start_time_left"],
#     end_time=input_dict["end_time_left"],
#     return_data_type="dataframe",
#     kwargs={
#         "data_folder_location": "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/",
#         "start_time": input_dict["start_time_left"],
#         "end_time": input_dict["end_time_left"],
#     },
# )

# filtered_files, file_list = lexi_functions.get_file_list(
#     data_folder_location="/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/",
#     start_time=input_dict["start_time_left"],
#     end_time=input_dict["end_time_left"],
# )
