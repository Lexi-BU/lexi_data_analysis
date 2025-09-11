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


start_time = "2024-05-23T22:45:00Z"
end_time = "2024-05-30T02:45:00Z"

delta_time_val = "24 hour"
delta_time = pd.Timedelta(delta_time_val)  # 1 hour delta time for the histogram

# Find the number of time intervals in the given time range
num_intervals = int((pd.to_datetime(end_time) - pd.to_datetime(start_time)) / delta_time)

# For each interval, calculate the start and end time and update the input dictionary
for i in range(num_intervals):
    start_time = pd.to_datetime(start_time) + delta_time
    end_time = pd.to_datetime(start_time) + delta_time
    start_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    input_dict = {
        "x_key": "x_volt",
        "y_key": "y_volt",
        # "start_time": "2025-03-16T19:45:00Z",  # sun-set
        # "end_time": "2025-03-16T21:15:00Z",
        "start_time": start_time,  # ground reference
        "end_time": end_time,
        # "start_time": "2025-03-06T03:30:00Z",  # reference
        # "end_time": "2025-03-06T04:30:00Z",
        # "start_time": "2025-03-06T15:05:00Z",  # sco-x
        # "end_time": "2025-03-06T16:35:00Z",
        # "start_time": "2025-03-08T05:00:00Z",  # high solar wind
        # "end_time": "2025-03-08T05:30:00Z",
        # "start_time": "2025-03-06T013:30:00Z",
        # "end_time": "2025-03-06T15:50:00Z",
        "bins": 200,
        "bin_range": [0.42, 0.55, 0.45, 0.58],  # [x_min, x_max, y_min, y_max]
        "time_normalization": True,
        "rotate_data": False,
    }

    try:
        read_data = True
        if read_data:
            print(f"Reading data for {input_dict['start_time']} to {input_dict['end_time']}...")
            org_hist, org_xedges, org_yedges, org_ra_median, org_dec_median = (
                lexi_functions.get_single_histogram_array(**input_dict)
            )
            # Save the histogram data to a pickle file along with start and end time

            print("Histogram data loaded successfully.")

        # Set all the zeros in the histogram to NaN
        org_hist = np.where(org_hist == 0, np.nan, org_hist)
        # Plot the histogram
        lexi_functions.plot_histograms(
            hist_center=org_hist,
            xedges=org_xedges,
            yedges=org_yedges,
            x_key=input_dict["x_key"],
            y_key=input_dict["y_key"],
            start_time_left=input_dict["start_time"],
            end_time_left=input_dict["end_time"],
            # start_time_right=ground_data["start_time"],
            # end_time_right=ground_data["end_time"],
            delta_time_left=1,
            delta_time_right=1,
            save_folder=f"/home/cephadrius/eng_drive/LEXI/04-00 Testing_and_Experiments/2023_06_08_HV_Testing/xy_histograms/volt_coordinate_no_lin/{delta_time_val}",
            save_plot_name=f"histogram_{input_dict['x_key']}_{input_dict['y_key']}_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}.png",
            save_plot=True,
            color_map_center="plasma",
            single_histogram=True,
            color_map_center_vmin=np.nanmin(org_hist),
            color_map_center_vmax=np.nanmax(org_hist),
        )

        old_start_time = input_dict["start_time"]
        old_end_time = input_dict["end_time"]

        print(f"Plot saved successfully for {old_start_time} to {old_end_time}.")
    except Exception:
        pass
        print(f"Error processing data for {input_dict['start_time']} to {input_dict['end_time']}.")
print("\a")
