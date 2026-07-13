import datetime
import importlib
import pickle
import warnings
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import map_coordinates

importlib.reload(lexi_functions)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Suppress dvision by zero warnings in numpy
np.seterr(divide="ignore", invalid="ignore")


# THEMIS spacecraft
themis_sc = "c"

themis_file_name = (
    f"../data/themis_data/csv/themis_{themis_sc}_esa_parameters_2025-03-16_to_2025-03-17_flux.csv"
)
themis_df = pd.read_csv(themis_file_name)
# Rename the first column to 'epoch' and set it as the index
themis_df.rename(columns={themis_df.columns[0]: "epoch"}, inplace=True)
themis_df.set_index("epoch", inplace=True)
# Convert the index to datetime
themis_df.index = pd.to_datetime(themis_df.index, utc=True)
# Select only flux columns
flux_key = f"th{themis_sc}_peir_flux"
themis_df = themis_df[[flux_key]]
# Drop all rows with NaN values
themis_df.dropna(inplace=True)
# Get the rolling mean of the flux at 1 second intervals
themis_df = themis_df.rolling("1s").mean()
# Save the processed THEMIS DataFrame to a csv file
resampled_themis_file_name = f"../data/themis_data/csv/themis_{themis_sc}_esa_parameters_2025-03-16_to_2025-03-17_flux_resampled.csv"
themis_df.to_csv(resampled_themis_file_name)

# resampled_themis_file_name = f"../data/themis_data/csv/themis_{themis_sc}_esa_parameters_2025-03-16_to_2025-03-17_flux_resampled.csv"
# # Read the resampled THEMIS DataFrame
# themis_df = pd.read_csv(resampled_themis_file_name)
# # Rename the first column to 'epoch' and set it as the index
# themis_df.rename(columns={themis_df.columns[0]: "epoch"}, inplace=True)
# themis_df.set_index("epoch", inplace=True)
# # Convert the index to datetime
# themis_df.index = pd.to_datetime(themis_df.index, utc=True)

# Load the Lexi counts DataFrame
lexi_counts_file_name = (
    "/home/cephandrius/Desktop/git/Lexi-BU/lexi_data_pipeline/data/counts_per_second.csv"
)
lexi_counts_df = pd.read_csv(lexi_counts_file_name)
# Rename the first column to 'epoch' and set it as the index
lexi_counts_df.rename(columns={lexi_counts_df.columns[0]: "epoch"}, inplace=True)
lexi_counts_df.set_index("epoch", inplace=True)
# Convert the index to datetime
lexi_counts_df.index = pd.to_datetime(lexi_counts_df.index, utc=True)

start_time = datetime.datetime(2025, 3, 16, 19, 0, 0, tzinfo=datetime.timezone.utc)
end_time = datetime.datetime(2025, 3, 16, 21, 15, 0, tzinfo=datetime.timezone.utc)

# Specify the delta time over which to calculate the correlation
delta_time = datetime.timedelta(minutes=10)

# Between the start and end time, calculate the correlation for each delta time
current_time = start_time
correlation_results = []
for i in range(int((end_time - start_time) / delta_time)):
    current_start = current_time
    current_end = current_start + delta_time

    # Filter the THEMIS DataFrame for the current time range
    themis_filtered = themis_df.loc[current_start:current_end]
    # Filter the Lexi counts DataFrame for the current time range
    lexi_filtered = lexi_counts_df.loc[current_start:current_end]

    # Ensure both DataFrames have the same length
    min_length = min(len(themis_filtered), len(lexi_filtered))
    if min_length == 0:
        continue

    # Calculate the correlation
    correlation = stats.pearsonr(
        themis_filtered.iloc[:min_length, 0], lexi_filtered.iloc[:min_length, 0]
    )[0]
    middle_time = current_start + delta_time / 2
    # Store the result with the time range
    correlation_results.append(
        {
            "start_time": current_start,
            "end_time": current_end,
            "middle_time": middle_time,
            "correlation": correlation,
        }
    )

    # Move to the next time interval
    current_time += delta_time

# On a 3 by 1 subplot grid, make a plot of following items with time:
# 1. THEMIS flux
# 2. Lexi counts
# 3. Correlation values
mpl.style.use("dark_background")  # type: ignore
plt.rcParams.update(
    {
        "axes.facecolor": "#000000",
        "axes.edgecolor": "#ffffff",
        "figure.facecolor": "#020202",
        "figure.edgecolor": "#ffffff",
        "grid.color": "#8E8B8B",
        "text.color": "#ffffff",
        "xtick.color": "#ffffff",
        "ytick.color": "#ffffff",
    }
)

fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
# Set the horizontal gap between subplots to zero
fig.subplots_adjust(hspace=0, wspace=0)
# Plot THEMIS flux
axs[0].plot(themis_df.index, themis_df.iloc[:, 0], label="THEMIS Flux", color="blue", alpha=0.7)
axs[0].set_ylabel("Flux (particles/s/m^2)")
# axs[0].set_title("THEMIS Flux")
# axs[0].legend()
# Plot Lexi counts on the right y-axis
twinx_1 = axs[0].twinx()
twinx_1.plot(
    lexi_counts_df.index, lexi_counts_df.iloc[:, 0], label="Lexi Counts", color="orange", alpha=0.7
)
twinx_1.set_ylabel("Counts (particles/s)")

# Add a legend for both plots
axs[0].legend(loc="upper left")
twinx_1.legend(loc="upper right")

# axs[1].set_title("Lexi Counts")
# axs[1].legend()
# Plot correlation values
correlation_times = [result["middle_time"] for result in correlation_results]
correlation_values = [result["correlation"] for result in correlation_results]
axs[1].scatter(
    correlation_times, correlation_values, label="Correlation", color="green", s=10, ls="--", lw=0.5
)

axs[1].set_ylabel("Correlation Coefficient")
# axs[2].set_title("Correlation between THEMIS Flux and Lexi Counts")
axs[1].set_xlabel("Time [UTC]")
axs[1].legend()

# Set the x-axis to display the time in a readable format
axs[1].xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
# Set the x-axis limit to the start and end time
axs[1].set_xlim(start_time, end_time)
axs[1].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
# For each plot, draw a vertical line at every 5 minutes interval
for ax in axs:
    for i in range(int((end_time - start_time) / delta_time)):
        line_time = start_time + i * delta_time
        ax.axvline(line_time, color="gray", linestyle="--", linewidth=0.5)

    # Define sunset times
    sunset_start_time = datetime.datetime(2025, 3, 16, 19, 38, 0)
    sunset_end_time = datetime.datetime(2025, 3, 16, 20, 44, 0)
    # Add a vertical line for sunset start and end times
    ax.axvline(
        sunset_start_time,
        color="white",
        linestyle="--",
    )
    ax.axvline(
        sunset_end_time,
        color="grey",
        linestyle="--",
    )
    # Add a shaded region for the sunset period
    # ax.axvspan(
    #     sunset_start_time,
    #     sunset_end_time,
    #     color="yellow",
    #     alpha=0.1,
    #     label="Sunset Period",
    #     zorder=1,
    # )
    gradient = np.linspace(0, 1, 256).reshape(1, -1)  # Horizontal gradient
    ax.imshow(
        gradient,
        extent=[sunset_start_time, sunset_end_time, ax.get_ylim()[0], ax.get_ylim()[1]],
        aspect="auto",
        cmap="binary_r",  # Yellow to brown colormap, you can customize this
        alpha=0.3,
        zorder=1,
    )
    # Add a label to the vertical lines if it is first subplot
    if ax is axs[0]:
        ax.text(
            sunset_start_time + datetime.timedelta(minutes=1),
            ax.get_ylim()[1] * 0.8,
            f"Sunset Start\n {sunset_start_time.strftime('%H:%M')}",
            color="white",
            ha="left",
            va="bottom",
        )
        ax.text(
            sunset_end_time - datetime.timedelta(minutes=1),
            ax.get_ylim()[1] * 0.8,
            f"Sunset End\n {sunset_end_time.strftime('%H:%M')}",
            color="grey",
            ha="right",
            va="bottom",
        )
plt.tight_layout()
fig_folder = Path("../figures/themis_lexi_correlation")
fig_folder.mkdir(parents=True, exist_ok=True)
fig_name = f"themis_lexi_correlation_{delta_time.total_seconds()}_{themis_sc}.png"
fig.savefig(fig_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
