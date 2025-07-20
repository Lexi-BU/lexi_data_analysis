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


rot_angle = 0  # 13.7  # rotation angle in degrees

input_dict = {
    "x_key": "photon_az",
    "y_key": "photon_el",
    # "start_time": "2025-03-16T19:45:00Z",  # sun-set
    # "end_time": "2025-03-16T21:15:00Z",
    "start_time": "2024-05-23T22:45:00Z",  # ground reference
    "end_time": "2024-05-30T02:45:00Z",
    # "start_time": "2025-03-06T03:30:00Z",  # reference
    # "end_time": "2025-03-06T04:30:00Z",
    # "start_time": "2025-03-06T15:05:00Z",  # sco-x
    # "end_time": "2025-03-06T16:35:00Z",
    # "start_time": "2025-03-08T05:00:00Z",  # high solar wind
    # "end_time": "2025-03-08T05:30:00Z",
    # "start_time": "2025-03-06T013:30:00Z",
    # "end_time": "2025-03-06T15:50:00Z",
    "bins": 200,
    "bin_range": [263, 281, 15, 33],
    "time_normalization": True,
    "rotate_data": False,
    "rotation_angle": rot_angle,
}

input_dict_2 = {
    "x_key": "photon_az",
    "y_key": "photon_el",
    "start_time": "2025-03-16T19:00:00Z",
    "end_time": "2025-03-16T21:15:00Z",
    "bins": 200,
    "bin_range": [263, 281, 15, 33],
    "time_normalization": True,
    "rotate_data": False,
    "rotation_angle": rot_angle,
}

normalize_against_ground = False

if "old_start_time" not in locals():
    old_start_time = input_dict["start_time"]
if "old_end_time" not in locals():
    old_end_time = input_dict["end_time"]

if old_start_time != input_dict["start_time"] or old_end_time != input_dict["end_time"]:
    read_data = True
else:
    read_data = False
read_data = False
if read_data:
    org_hist, org_xedges, org_yedges, org_ra_median, org_dec_median = (
        lexi_functions.get_single_histogram_array_l1c_files(**input_dict)
    )
    # Save the histogram data to a pickle file along with start and end time
    save_folder = Path("../data/")
    save_folder.mkdir(parents=True, exist_ok=True)
    file_name = f"line_profile_histogram_data_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}_xkey_{input_dict['x_key']}_ykey_{input_dict['y_key']}_l1c.pkl"
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
                "rotation_angle": input_dict["rotation_angle"],
            },
            f,
        )

    print("Histogram data loaded successfully.")
    if normalize_against_ground:
        # Save the histogram data to a pickle file along with start and end time
        save_folder = Path("../data/")
        save_folder.mkdir(parents=True, exist_ok=True)
        file_name = f"ground_histogram_data_{input_dict_2['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict_2['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}_xkey_{input_dict_2['x_key']}_ykey_{input_dict_2['y_key']}_l1c.pkl"
        save_file = save_folder / file_name
        # Check if the file already exists
        if save_file.exists():
            # Read the existing file
            with open(save_file, "rb") as f:
                ground_data = pickle.load(f)
            ground_hist = ground_data["hist"]
            ground_xedges = ground_data["xedges"]
            ground_yedges = ground_data["yedges"]
            ground_ra_median = ground_data["ra_median"]
        else:
            ground_hist, ground_xedges, ground_yedges, ground_ra_median, ground_dec_median = (
                lexi_functions.get_single_histogram_array_l1c_files(**input_dict_2)
            )
            ground_hist = np.where(ground_hist == 0, np.nan, ground_hist)
            with open(save_file, "wb") as f:
                pickle.dump(
                    {
                        "hist": ground_hist,
                        "xedges": ground_xedges,
                        "yedges": ground_yedges,
                        "ra_median": ground_ra_median,
                        "dec_median": ground_dec_median,
                        "start_time": input_dict_2["start_time"],
                        "end_time": input_dict_2["end_time"],
                        "bins": input_dict_2["bins"],
                        "bin_range": input_dict_2["bin_range"],
                        "time_normalization": input_dict_2["time_normalization"],
                        "x_key": input_dict_2["x_key"],
                        "y_key": input_dict_2["y_key"],
                        "rotation_angle": input_dict_2["rotation_angle"],
                    },
                    f,
                )
            print(f"Ground histogram data saved to {save_file}")
        scaled_hist = org_hist - ground_hist
    print("Histogram data normalized against ground data.")

# Plot the histogram
lexi_functions.plot_histograms(
    hist_left=org_hist,
    hist_right=ground_hist,
    hist_center=org_hist,
    xedges=org_xedges,
    yedges=org_yedges,
    x_key=input_dict["x_key"],
    y_key=input_dict["y_key"],
    start_time_left=input_dict["start_time"],
    end_time_left=input_dict["end_time"],
    # start_time_right=ground_data["start_time"],
    # end_time_right=ground_data["end_time"],
    start_time_right=input_dict_2["start_time"],
    end_time_right=input_dict_2["end_time"],
    delta_time_left=1,
    delta_time_right=1,
    save_folder="../figures/histogram_normalized_diff/single_histogram/l1c/",
    save_plot_name=f"histogram_{input_dict['x_key']}_{input_dict['y_key']}_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_normalized_l1c.png",
    save_plot=True,
    color_map_center="inferno",
    single_histogram=True,
)

old_start_time = input_dict["start_time"]
old_end_time = input_dict["end_time"]
old_rot_angle = input_dict["rotation_angle"]

print("Plot saved successfully.")
print("\a")
