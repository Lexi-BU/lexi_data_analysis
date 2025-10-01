import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV file
data_folder = Path("../data/line_profile_data/bg_corrected/from_l2/")
csv_file = data_folder / "line_profile_fit_parameters_bg_corrected_1min.csv"
data_df = pd.read_csv(csv_file)

data_df["start_time"] = pd.to_datetime(data_df["start_time"], utc=True)
data_df["end_time"] = pd.to_datetime(data_df["end_time"], utc=True)
# Make time series plots of the fit parameters
data_df["mid_time"] = data_df["start_time"] + (data_df["end_time"] - data_df["start_time"]) / 2
data_df.set_index("mid_time", inplace=True)
# Convert the index to datetime if not already
data_df.index = pd.to_datetime(data_df.index, utc=True)
# Sort the dataframe by the datetime index
data_df.sort_index(inplace=True)


def plot_fit_parameters(data_df, key):
    """Plot the fit parameters over time."""

    alpha = 0.2
    line_thickness = 1
    marker_size = 5
    marker = "d"
    line_style = "--"
    plt.style.use("dark_background")
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True, sharex=True)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    mpl.rcParams.update({"font.size": 16})

    fig.suptitle(f"Line Profile Fit Parameters Over Time for {key}", fontsize=20)
    axs[0].plot(
        data_df.index,
        data_df[f"raw_counts_{key}"],
        color="cyan",
        ms=marker_size,
        marker=marker,
        linestyle=line_style,
        linewidth=line_thickness * 2,
    )
    axs[0].set_ylabel(f"Raw Counts")
    axs[0].set_yscale("linear")
    axs[0].grid(True, which="both", linestyle="--", alpha=alpha, linewidth=line_thickness)

    axs[1].plot(
        data_df.index,
        data_df[f"background_corrected_{key}"],
        color="magenta",
        ms=marker_size,
        marker=marker,
        linestyle=line_style,
        linewidth=line_thickness * 2,
    )
    # Add the y-label on right side
    axs[1].yaxis.set_label_position("right")
    axs[1].set_ylabel(f"Background-Corrected")
    axs[1].set_yscale("linear")
    axs[1].grid(True, which="both", linestyle="--", alpha=alpha, linewidth=line_thickness)

    axs[2].plot(
        data_df.index,
        data_df[f"background_flatfield_corrected_{key}"],
        color="orange",
        ms=marker_size,
        marker=marker,
        linestyle=line_style,
        linewidth=line_thickness * 2,
    )
    axs[2].set_ylabel(f"Flat-Field Corrected")
    axs[2].set_xlabel("Time [UTC]")
    axs[2].set_yscale("linear")
    axs[2].grid(True, which="both", linestyle="--", alpha=alpha, linewidth=line_thickness)

    sunset_start_time = datetime.datetime(2025, 3, 16, 18, 22, 0, tzinfo=datetime.timezone.utc)
    sunset_end_time = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)

    for ax in axs:
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
    # axs[2].text(
    #     sunset_end_time - datetime.timedelta(minutes=1),
    #     axs[2].get_ylim()[0],
    #     f"Sunset End\n {sunset_end_time.strftime('%H:%M')}",
    #     color="white",
    #     ha="right",
    #     va="bottom",
    # )

    # Find the value of raw_counts_key at max high voltage time
    max_hv_time = pd.Timestamp("2025-03-16 19:03:00", tz="UTC")
    idx = data_df.index.get_indexer([max_hv_time], method="nearest")[0]
    closest_time = data_df.index[idx]
    max_hv_value = data_df[f"raw_counts_{key}"].iloc[idx]

    # At the closest time add an arrow pointing to the point of max_hv_value with the text "Max High Voltage"
    axs[1].annotate(
        "Max High Voltage Reached",
        xy=(closest_time, max_hv_value),
        xytext=(closest_time + pd.Timedelta(minutes=1), max_hv_value + 1),
        arrowprops=dict(facecolor="white", shrink=0.05, width=1, headwidth=8, headlength=10),
        color="white",
    )
    # Set the x-axis limits to the data_df index range
    axs[2].set_xlim(
        # data_df.index.min() - pd.Timedelta(minutes=2.5),
        sunset_end_time,
        data_df.index.max() + pd.Timedelta(minutes=2.5),
    )

    data_df_selected = data_df[
        (data_df.index >= sunset_end_time) & (data_df.index <= data_df.index.max())
    ]
    for ax in axs:
        # Set the y-axis limits to the min and max of the selected data_df for the given key
        y_min = (
            data_df_selected[
                [
                    f"raw_counts_{key}",
                    f"background_corrected_{key}",
                    f"background_flatfield_corrected_{key}",
                ]
            ]
            .min()
            .min()
        )
        y_max = (
            data_df_selected[
                [
                    f"raw_counts_{key}",
                    f"background_corrected_{key}",
                    f"background_flatfield_corrected_{key}",
                ]
            ]
            .max()
            .max()
        )
        ax.set_ylim(y_min, y_max)

    """
    counts per second data
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
    """
    plt.tight_layout()
    # Save the figure
    output_folder = Path("../figures/line_profile_fit_parameters/")
    output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_folder / f"line_profile_fit_parameters_time_series_{key}_1min.png")
    plt.close(fig)


keys_to_plot = ["slope", "intercept", "r_squared", "residuals"]
for key in keys_to_plot:
    plot_fit_parameters(data_df, key)
    print(f"Plotted fit parameter time series for key: {key}")
