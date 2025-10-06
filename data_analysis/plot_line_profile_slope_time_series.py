import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fit_parameters(data_df, flux_data, key):
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

    data_df_selected = data_df[
        (data_df.index >= sunset_end_time) & (data_df.index <= data_df.index.max())
    ]
    for ax in axs:
        # Set the y-axis limits to the min and max of the selected data_df for the given key
        y_min = (
            data_df[
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
            data_df[
                [
                    f"raw_counts_{key}",
                    f"background_corrected_{key}",
                    f"background_flatfield_corrected_{key}",
                ]
            ]
            .max()
            .max()
        )
        ax.set_ylim(y_min * 1.1, y_max * 1.1)

    # Set the x-axis limits to the data_df index range
    axs[2].set_xlim(
        data_df.index.min() - pd.Timedelta(minutes=2.5),
        # sunset_end_time,
        data_df.index.max() + pd.Timedelta(minutes=2.5),
    )

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

        gradient = np.linspace(0, 1, 256).reshape(1, -1)  # Horizontal gradient
        ax.imshow(
            gradient,
            extent=[sunset_start_time, sunset_end_time, ax.get_ylim()[0], ax.get_ylim()[1]],
            aspect="auto",
            cmap="binary_r",  # Yellow to brown colormap, you can customize this
            alpha=0.3,
            zorder=1,
        )
        # Add an arrow pointing to the sunset end time with appropriate annotation
        ax.annotate(
            "Sunset End",
            xy=(sunset_end_time, ax.get_ylim()[1] * 0.9),
            xytext=(sunset_end_time + datetime.timedelta(minutes=5), ax.get_ylim()[1] * 0.9),
            arrowprops=dict(facecolor="white", shrink=0.05, width=1, headwidth=8, headlength=10),
            color="white",
            ha="left",
            va="bottom",
        )

    # Add a gradient on all the axes after the wake time until the end
    gradient = np.linspace(0, 1, 256).reshape(1, -1)  # Horizontal gradient
    for ax in axs:
        ax.axvline(
            wake_time,
            color="yellow",
            linestyle="--",
        )
        ax.imshow(
            gradient,
            extent=[
                wake_time,
                data_df.index.max() + pd.Timedelta(minutes=2.5),
                ax.get_ylim()[0],
                ax.get_ylim()[1],
            ],
            aspect="auto",
            cmap="YlGn",  # Yellow to brown colormap
            alpha=0.1,
            zorder=1,
        )

    # Add an arrow pointing to the wake time with appropriate annotation
    for ax in axs:
        ax.annotate(
            "Wake Time",
            xy=(wake_time, ax.get_ylim()[1] * 0.9),
            xytext=(wake_time + datetime.timedelta(minutes=5), ax.get_ylim()[1] * 0.9),
            arrowprops=dict(facecolor="yellow", shrink=0.05, width=1, headwidth=8, headlength=10),
            color="yellow",
            ha="left",
            va="bottom",
        )

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

    plt.tight_layout()
    # Save the figure
    output_folder = Path("../figures/line_profile_fit_parameters/")
    output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_folder / f"line_profile_fit_parameters_time_series_{key}_1min_flux_avg.png")
    plt.close(fig)


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

themis_spc = "c"
flux_file_name = (
    f"../data/lexi_themis_analysis/lexi_themis_{themis_spc}_analysis_lexi_spacecraft.csv"
)
if themis_spc == "b":
    wake_time = "2025-03-16 20:47"
elif themis_spc == "c":
    wake_time = "2025-03-16 20:55"

wake_time = pd.to_datetime(wake_time, utc=True)

flux_df = pd.read_csv(flux_file_name)
# Set Epoch as datetime index
flux_df["Epoch"] = pd.to_datetime(flux_df["Epoch"], utc=True)
flux_df.set_index("Epoch", inplace=True)
flux_df.sort_index(inplace=True)
flux_data = flux_df[f"th{themis_spc}_peef_flux"]

keys_to_plot = ["slope"]  # , "intercept", "r_squared", "residuals"]
for key in keys_to_plot:
    plot_fit_parameters(data_df, flux_data, key)
    print(f"Plotted fit parameter time series for key: {key}")
