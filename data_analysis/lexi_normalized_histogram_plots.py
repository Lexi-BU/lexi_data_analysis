import importlib
import pickle
import warnings
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

importlib.reload(lexi_functions)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Suppress dvision by zero warnings in numpy
np.seterr(divide="ignore", invalid="ignore")


input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    # "start_time": "2025-03-16T19:45:00Z",
    # "end_time": "2025-03-16T21:15:00Z",
    # "start_time": "2024-05-23T22:45:00Z",
    # "end_time": "2024-05-30T02:45:00Z",
    "start_time": "2025-03-06T15:05:00Z",
    "end_time": "2025-03-06T16:35:00Z",
    "bins": 200,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
}

read_data = False
normalize_against_ground = True
if (
    "org_hist" not in locals()
    or "org_xedges" not in locals()
    or "org_yedges" not in locals()
    or read_data
):
    org_hist, org_xedges, org_yedges, org_ra_median, org_dec_median = (
        lexi_functions.get_single_histogram_array(**input_dict)
    )
    # Save the histogram data to a pickle file along with start and end time
    save_folder = Path("../data/")
    save_folder.mkdir(parents=True, exist_ok=True)
    file_name = f"line_profile_histogram_data_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}.pkl"
    save_file = save_folder / file_name
    with open(save_file, "wb") as f:
        pickle.dump(
            {
                "hist": org_hist,
                "xedges": org_xedges,
                "yedges": org_yedges,
                "ra_median": org_ra_median,
                "dec_median": org_dec_median,
                "start_time": input_dict["start_time"],
                "end_time": input_dict["end_time"],
                "bins": input_dict["bins"],
                "bin_range": input_dict["bin_range"],
                "time_normalization": input_dict["time_normalization"],
                "x_key": input_dict["x_key"],
                "y_key": input_dict["y_key"],
            },
            f,
        )

    print("Histogram data loaded successfully.")
    if normalize_against_ground:
        ground_file_name = (
            "../data/ground_test_histogram_data_20240523_224500Z_20240530_024500Z.pkl"
        )
        with open(ground_file_name, "rb") as f:
            ground_data = pickle.load(f)
        ground_hist = ground_data["hist"]
        # Replace 0 values in ground_hist with nan
        ground_hist = np.where(ground_hist == 0, np.nan, ground_hist)
        scaled_hist = org_hist / ground_hist
    print("Histogram data normalized against ground data.")

# Plot the histogram
lexi_functions.plot_histograms(
    hist_left=org_hist,
    hist_right=ground_hist,
    hist_center=scaled_hist,
    xedges=org_xedges,
    yedges=org_yedges,
    x_key=input_dict["x_key"],
    y_key=input_dict["y_key"],
    start_time_left=input_dict["start_time"],
    end_time_left=input_dict["end_time"],
    start_time_right=ground_data["start_time"],
    end_time_right=ground_data["end_time"],
    delta_time_left=1,
    delta_time_right=1,
    save_folder="../figures/histogram_normalized_diff/",
    save_plot_name="histogram_normalized_diff",
    save_plot=True,
    color_map_center="viridis",
)
