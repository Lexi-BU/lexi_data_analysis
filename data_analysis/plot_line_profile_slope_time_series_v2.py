import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.patches import FancyArrowPatch

plt.style.use("default")
mpl.rcParams.update({"font.size": 22})
# Set the fonttype to 42 to avoid Type 3 fonts in the output PDF/PNGs
mpl.rcParams["pdf.fonttype"] = 42
# Set the the font to be arial-like for better readability
mpl.rcParams["font.family"] = "Arial"
# Set latex-style text rendering
mpl.rcParams["text.usetex"] = True


def horizontal_labeled_arrow(
    ax, x0, x1, y, label, facecolor="white", alpha=0.25, text_color="black"
):
    # draw the arrow
    arrow = FancyArrowPatch(
        (x0, y),
        (x1, y),
        arrowstyle="-|>",
        mutation_scale=20,
        lw=0.0,
        facecolor=facecolor,
        edgecolor=facecolor,
        alpha=alpha,
        transform=ax.transData,
        zorder=5,
    )
    ax.add_patch(arrow)
    # add the label in the middle of the arrow
    ax.text((x0 + (x1 - x0) / 2), y, label, ha="center", va="center", color=text_color, fontsize=12)


def plot_fit_parameters(data_df, flux_data, key):
    """Plot the fit parameters over time."""

    alpha = 0.2
    line_thickness = 1
    marker_size = 5
    marker = "d"
    line_style = "--"
    plt.style.use("dark_background")

    # Panels with zero gap
    fig, axs = plt.subplots(
        3, 1, figsize=(15, 12), sharex=True, gridspec_kw={"hspace": 0.0, "wspace": 0.0}
    )
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
    axs[0].set_ylabel("Raw Counts")
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
    axs[1].yaxis.set_label_position("right")
    axs[1].set_ylabel("Background-Corrected")
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
    axs[2].set_ylabel("Flat-Field Corrected")
    axs[2].set_xlabel("Time [UTC]")
    axs[2].set_yscale("linear")
    axs[2].grid(True, which="both", linestyle="--", alpha=alpha, linewidth=line_thickness)

    # Key times
    sunset_time = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)
    wake_time = datetime.datetime(
        2025, 3, 16, 20, 0, 0, tzinfo=datetime.timezone.utc
    )  # ensure defined

    # Global y-lims synced across panels for the three series
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
    for ax in axs:
        ax.set_ylim(y_min * 1.1, y_max * 1.1)

    # X-lims
    axs[2].set_xlim(
        data_df.index.min() - pd.Timedelta(minutes=2.5),
        data_df.index.max() + pd.Timedelta(minutes=2.5),
    )

    # Vertical lines for reference
    for ax in axs:
        ax.axvline(sunset_time, color="white", linestyle="--", linewidth=1)
        ax.axvline(wake_time, color="yellow", linestyle="--", linewidth=1)

    # Sunset arrow: point to the sunset time (left→right), text inside arrow
    # Arrow spans from 10 minutes before sunset to the sunset instant
    sunset_span = pd.Timedelta(minutes=10)
    # Wake arrow: point towards increasing time starting at wake time
    # Arrow spans 10 minutes after wake
    wake_span = pd.Timedelta(minutes=10)

    for ax in axs:
        y_arrow = ax.get_ylim()[1] * 0.9

        # arrow pointing to sunset (backwards)
    horizontal_labeled_arrow(
        ax,
        sunset_time - pd.Timedelta(minutes=10),
        sunset_time,
        y_arrow,
        "Sunset",
        facecolor="white",
    )

    # arrow pointing forward from wake time
    horizontal_labeled_arrow(
        ax, wake_time, wake_time + pd.Timedelta(minutes=10), y_arrow, "Wake", facecolor="yellow"
    )

    # Mark the HV time on middle panel
    max_hv_time = pd.Timestamp("2025-03-16 19:03:00", tz="UTC")
    idx = data_df.index.get_indexer([max_hv_time], method="nearest")[0]
    closest_time = data_df.index[idx]
    max_hv_value = data_df[f"raw_counts_{key}"].iloc[idx]
    axs[1].annotate(
        "Max High Voltage",
        xy=(closest_time, max_hv_value),
        xytext=(closest_time + pd.Timedelta(minutes=1), max_hv_value + 1),
        arrowprops=dict(facecolor="white", shrink=0.05, width=1, headwidth=8, headlength=10),
        color="white",
    )

    # Save
    output_folder = Path("../figures/line_profile_fit_parameters/")
    output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_folder / f"line_profile_fit_parameters_time_series_{key}_1min_flux_avg.png",
        dpi=200,
        bbox_inches="tight",
    )
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
