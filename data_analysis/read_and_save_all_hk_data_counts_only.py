import datetime
import glob
import importlib
from pathlib import Path

import matplotlib.dates as mdates

# import lexi_data_analysis_functions as lexi_functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spacepy.pycdf import CDF as cdf

# importlib.reload(lexi_functions)

"""
data_folder_location = "/mnt/cephadrius/bu_research/lexi_data/L1b/hk/cdf/"
start_time = "2025-01-16T00:00:00Z"
end_time = "2025-03-17T00:00:00Z"

filtered_files, file_list = lexi_functions.get_file_list(data_folder_location, start_time, end_time)

# Read the data from the CDF files
df = lexi_functions.read_all_data_files(
    file_list=filtered_files,
    start_time=start_time,
    end_time=end_time,
    return_data_type="dataframe",
    kwargs={"data_folder_location": data_folder_location},
)

columns_to_drop = [
    "HK_id",
    "Pinpuller_Armed",
    "Unused1",
    "Unused2",
    "HVmcpAuto",
    "HVmcpMan",
]

df = df.drop(columns=columns_to_drop)

df["all_counts"] = df["DeltaEvntCount"] + df["DeltaDroppedCount"] + df["DeltaLostEvntCount"]
df["HV_value"] = df["AnodeVoltMon"] * 599

# Save the data to a pickle file
start_time_str = start_time.replace(":", "-").replace("T", "_").replace("Z", "")
end_time_str = end_time.replace(":", "-").replace("T", "_").replace("Z", "")
file_name = f"../data/lexi_hk_data_{start_time_str}_to_{end_time_str}.pkl"
df.to_pickle(file_name)

hk_file_name = "../data/lexi_hk_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00.pkl"
look_direction_file_name = "../data/lexi_look_direction_data.csv"
# Read the data from the pickle file
df_pickle = pd.read_pickle(hk_file_name)

df_look_direction = pd.read_csv(look_direction_file_name)
# Set the Epoch column to index
df_look_direction["Epoch"] = pd.to_datetime(df_look_direction["Epoch"])
df_look_direction = df_look_direction.set_index("Epoch")

# Merge the two dataframes on the index
merged_df = pd.merge_asof(
    df_pickle,
    df_look_direction,
    left_index=True,
    right_index=True,
    direction="nearest",
    tolerance=pd.Timedelta("60s"),
)

merged_file_name = (
    "../data/merged_lexi_hk_look_direction_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00.pkl"
)
# Save the merged data to a pickle file
merged_df.to_pickle(merged_file_name)
"""
merged_file_name = (
    "../data/merged_lexi_hk_look_direction_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00.pkl"
)
# Save it to a csv file
# merged_file_name_csv = (
#     "../data/merged_lexi_hk_look_direction_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00.csv"
# )
# merged_df = pd.read_pickle(merged_file_name)
# merged_df.to_csv(merged_file_name_csv)
# Read the merged data from the pickle file
merged_df = pd.read_pickle(merged_file_name)

plot_start_time_start = "2025-03-16T19:00:00Z"
plot_end_time_start = "2025-03-16T21:59:00Z"

# Convert to datetime objects
start_time = datetime.datetime.strptime(plot_start_time_start, "%Y-%m-%dT%H:%M:%SZ")
end_time = datetime.datetime.strptime(plot_end_time_start, "%Y-%m-%dT%H:%M:%SZ")

# Generate time lists with 3-hour intervals
plot_start_time_list = []
plot_end_time_list = []

delta = datetime.timedelta(minutes=180)

current_time = start_time
while current_time < end_time:
    plot_start_time_list.append(current_time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    next_time = current_time + delta
    plot_end_time_list.append(min(next_time, end_time).strftime("%Y-%m-%dT%H:%M:%SZ"))
    current_time = next_time

# Set the start and end times for the plot
# plot_start_time_list = ["2025-03-06T14:00:00Z", "2025-03-08T23:00:00Z", "2025-03-06T14:42:00Z"]
# plot_end_time_list = ["2025-03-06T17:00:00Z", "2025-03-09T02:00:00Z", "2025-03-06T14:58:00Z"]

for plot_start_time, plot_end_time in zip(plot_start_time_list, plot_end_time_list):
    # Filter the data based on the start and end times
    selected_merged_df = merged_df[plot_start_time:plot_end_time]

    keys_to_plot = ["all_counts", "dec_lexi", "ra_lexi"]

    # Set the font size
    plt.rcParams.update({"font.size": 12})
    # Use the black background style
    plt.style.use("dark_background")
    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    plt.subplots_adjust(hspace=0.05, wspace=0.0)

    # Plot the counts data on the first axis
    ax.scatter(
        selected_merged_df.index,
        selected_merged_df["all_counts"],
        label="All Counts",
        color="w",
        s=1,
        alpha=1,
    )
    ax.scatter(
        selected_merged_df.index,
        selected_merged_df["DeltaDroppedCount"],
        label="DeltaDroppedCount",
        color="b",
        s=1,
        alpha=0.25,
    )
    ax.scatter(
        selected_merged_df.index,
        selected_merged_df["DeltaLostEvntCount"],
        label="DeltaLostEvntCount",
        color="g",
        s=1,
        alpha=0.25,
    )
    ax.set_ylabel("Counts [#]")
    # ax.set_yscale("log")
    twin_ax_1 = ax.twinx()
    twin_ax_1.scatter(
        selected_merged_df.index,
        selected_merged_df["DeltaEvntCount"],
        label="DeltaEvntCount",
        color="r",
        s=1,
        alpha=0.5,
    )
    # twin_ax_1.set_yscale("log")
    twin_ax_1.set_ylabel("Delta Event Count", color="r")
    # Set the legend
    # ax.legend(loc="upper left")
    ax.text(
        0.05,
        1.01,
        "All Counts",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="w",
    )
    ax.text(
        0.18,
        1.01,
        "Dropped Counts",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="b",
    )
    ax.text(
        0.40,
        1.01,
        "Lost Counts",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="g",
    )
    ax.text(
        0.75,
        1.01,
        "Event Counts (Sci)",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="y",
    )
    ax.text(
        0.98,
        1.01,
        "Event Counts (HK)",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="r",
    )
    ax.grid(color="c", linestyle="--", linewidth=0.5, alpha=0.5)

    # Set the spine color to match the line color
    # twin_ax.spines["left"].set_color("w")
    twin_ax_1.spines["right"].set_color("r")
    # ax.tick_params(axis="y", colors="w")
    twin_ax_1.tick_params(axis="y", colors="r")
    # twin_ax_1.set_ylim(100, 3200)

    # Set the x-axis label
    ax.set_xlabel(f"Time on {selected_merged_df.index.min().strftime('%Y-%m-%d')} [UTC]")
    # Set the x-axis limits to minimum and maximum time in the merged dataframe
    ax.set_xlim(
        selected_merged_df.index.min(),
        selected_merged_df.index.max(),
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=3))

    # Set maximum number of x-ticks on x-axis to 5
    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    min_time_str = selected_merged_df.index.min().strftime("%Y-%m-%dT%H:%M:%SZ").replace("T", " ").replace("Z", "")
    max_time_str = (
        selected_merged_df.index.max()
        .strftime("%Y-%m-%dT%H:%M:%SZ")
        .replace("T", " ")
        .replace("Z", "")
    )

    # Put all ticks inside the plot
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(which="both", direction="in", color="w")
    twin_ax_1.tick_params(which="both", direction="in", color="r")

    # Define sunset times
    # NOTE: These sunset start and end times are from FireFly. It was computed using a spherical
    # model of the moon. However, as Tim Stubbs pointed out, the sunset times are not accurate since
    # it does not take into account the local topography.
    # sunset_start_time = datetime.datetime(2025, 3, 16, 19, 38, 0)
    # sunset_end_time = datetime.datetime(2025, 3, 16, 20, 44, 0)
    # NOTE: Here is the note from Tim Stubbs:
    # These are the sunset start and end times that I estimated using LROC Quickmap, which includes the effects of terrain …

    # Start time: 2025-03-16   18:22 UTC
    # End time:   2025-03-16   19:29 UTC

    # The observation point was …
    # Latitude:  18.56220 deg
    # Longitude:  61.81021 deg
    # Height: -3.648 km (relative to reference radius)
    # This equates to a height above the surface of 2 m
    # The Firefly website stated that Blue Ghost was 2 m tall. LEXI was on the top deck.
    # These estimates are probably +/- ~10 minutes
    sunset_start_time = datetime.datetime(2025, 3, 16, 18, 22, 0, tzinfo=datetime.timezone.utc)
    sunset_end_time = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)

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
    # Add a label to the vertical lines
    # ax.text(
    #     sunset_start_time + datetime.timedelta(minutes=1),
    #     ax.get_ylim()[1] * 0.9,
    #     f"Sunset Start\n {sunset_start_time.strftime('%H:%M')}",
    #     color="white",
    #     ha="left",
    #     va="bottom",
    # )
    ax.text(
        sunset_end_time - datetime.timedelta(minutes=1),
        ax.get_ylim()[1] * 0.9,
        f"Sunset End\n {sunset_end_time.strftime('%H:%M')}",
        color="grey",
        ha="right",
        va="bottom",
    )

    counts_per_second_file = (
        "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_pipeline/data/counts_per_second.csv"
    )
    df_counts = pd.read_csv(counts_per_second_file, index_col=0, parse_dates=True)
    df_counts = df_counts.loc[plot_start_time:plot_end_time]
    # Add the df_counts data to the plot (left axis)
    twin_ax_1.scatter(
        df_counts.index,
        df_counts["0"],
        label="Counts per Second",
        color="y",
        s=1,
        alpha=0.5,
        zorder=22,
    )

    # Set the title
    fig.suptitle(
        f"LEXI Data from {min_time_str} to {max_time_str}\n The sunset period is from {sunset_start_time.strftime('%H:%M')} to {sunset_end_time.strftime('%H:%M')}",
        fontsize=14,
    )
    # ax.set_ylim(0, 1500)
    # # Add a vertical line at 19:45
    # vertical_line_time = datetime.datetime.strptime("2025-03-16T19:45:00Z", "%Y-%m-%dT%H:%M:%SZ")
    # ax.axvline(
    #     x=vertical_line_time,
    #     color="y",
    #     linestyle="--",
    #     linewidth=2,
    #     label="Vertical Line at 19:45 UTC",
    # )
    # ax.text(
    #     vertical_line_time + datetime.timedelta(minutes=2),
    #     ax.get_ylim()[1] * 0.9,
    #     "19:45 UTC",
    #     color="y",
    #     fontsize=10,
    #     horizontalalignment="left",
    #     verticalalignment="bottom",
    # )
    # Save the figure
    fig.savefig(
        f"../figures/lexi_data_{plot_start_time}_to_{plot_end_time}_all_counts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
