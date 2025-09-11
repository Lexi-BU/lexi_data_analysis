import datetime
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder_name = "../data/line_profile_data/"
file_list = sorted(glob.glob(f"{folder_name}/*.csv"))

sunset_start_time = datetime.datetime(2025, 3, 16, 19, 38, 0)
sunset_end_time = datetime.datetime(2025, 3, 16, 20, 44, 0)
# define a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Loop through each file and plot the data
for file in file_list[:]:
    # Read the CSV file
    data = pd.read_csv(file)
    integration_time = file.split("/")[-1].split("_")[-2]
    # Extract the time and slope values
    start_time = pd.to_datetime(data["start_time"])
    end_time = pd.to_datetime(data["end_time"])
    # Get the middle time for plotting
    middle_time = start_time + (end_time - start_time) / 2
    slope = data["slope"]

    # Plot the data
    ax.scatter(middle_time, slope, label=integration_time, alpha=1, s=10)

# Add a vertical line for sunset start and end times
ax.axvline(
    sunset_start_time,
    color="orange",
    linestyle="--",
)
ax.axvline(
    sunset_end_time,
    color="red",
    linestyle="--",
)
# Add a label to the vertical lines
ax.text(
    sunset_start_time + datetime.timedelta(minutes=1),
    ax.get_ylim()[1] * 0.9,
    "Sunset Start",
    color="orange",
    ha="left",
    va="bottom",
)
ax.text(
    sunset_end_time - datetime.timedelta(minutes=1),
    ax.get_ylim()[1] * 0.9,
    "Sunset End",
    color="red",
    ha="right",
    va="bottom",
)
# Add a shaded region for the sunset period
ax.axvspan(
    sunset_start_time, sunset_end_time, color="yellow", alpha=0.05, label="Sunset Period", zorder=1
)
# Customize the plot
ax.set_title("Line Profile Slope over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Slope")
ax.legend()
file_name = "line_profile_slope_time.png"
save_folder = Path("../figures/line_profile_slope_time/")
save_folder.mkdir(parents=True, exist_ok=True)
fig.savefig(save_folder / file_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
