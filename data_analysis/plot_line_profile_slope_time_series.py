import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_fit_parameters(data_df, flux_data, key, time_resolution="5min"):
    """Plot the fit parameters over time."""

    alpha = 0.2
    line_thickness = 1
    marker_size = 3
    marker = "d"
    line_style = "--"
    # plt.style.use("dark_background")
    plt.style.use("default")
    # Set the font to Arial
    mpl.rcParams["font.family"] = "Arial"
    # Set latex style for plots
    mpl.rcParams["text.usetex"] = True
    fig, axs = plt.subplots(
        3, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.0, "wspace": 0.0}
    )
    mpl.rcParams.update({"font.size": 18})
    default_fontsize = mpl.rcParams["font.size"]
    fig.suptitle(f"Line Profile Fit Parameters Over Time for {key}", fontsize=20)

    fig.suptitle(f"Line Profile Fit Parameters Over Time for {key}", fontsize=20)
    # Normalize the data to the max value for better visualization
    
    axs[0].plot(
        data_df.index,
        data_df[f"background_flatfield_corrected_{key}"],
        color="k",
        ms=marker_size,
        marker=marker,
        linestyle=line_style,
        linewidth=line_thickness * 2,
    )

    axs[0].set_ylabel("Flat-Field Corrected Counts")
    axs[0].set_yscale("linear")
    axs[0].grid(True, which="both", linestyle="--", alpha=alpha, linewidth=line_thickness)

    axs[1].plot(
        data_df.index,
        data_df["background_flatfield_corrected_total_hist_counts"],
        color="magenta",
        ms=marker_size,
        marker=marker,
        linestyle=line_style,
        linewidth=line_thickness * 2,
    )

    # Add the y-label on right side
    axs[1].yaxis.set_label_position("right")
    axs[1].set_ylabel("Total Counts")
    axs[1].set_yscale("linear")
    axs[1].grid(True, which="both", linestyle="--", alpha=alpha, linewidth=line_thickness)

    axs[2].plot(
        flux_data.index,
        flux_data,
        color="orange",
        ms=marker_size,
        marker=marker,
        linestyle=line_style,
        linewidth=line_thickness * 2,
    )

    axs[2].set_ylabel("Solar Wind Flux")
    axs[2].set_xlabel("Time [UTC]")
    axs[2].set_yscale("linear")
    axs[2].grid(True, which="both", linestyle="--", alpha=alpha, linewidth=line_thickness)

    sunset_start_time = datetime.datetime(2025, 3, 16, 18, 22, 0, tzinfo=datetime.timezone.utc)
    sunset_end_time = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)

    data_df_selected = data_df[
        (data_df.index >= sunset_end_time) & (data_df.index <= data_df.index.max())
    ]
    # for ax in axs:
    #     # Set the y-axis limits to the min and max of the selected data_df for the given key
    #     y_min = (
    #         data_df[
    #             [
    #                 f"raw_counts_{key}",
    #                 f"background_corrected_{key}",
    #                 f"background_flatfield_corrected_{key}",
    #             ]
    #         ]
    #         .min()
    #         .min()
    #     )
    #     y_max = (
    #         data_df[
    #             [
    #                 f"raw_counts_{key}",
    #                 f"background_corrected_{key}",
    #                 f"background_flatfield_corrected_{key}",
    #             ]
    #         ]
    #         .max()
    #         .max()
    #     )
    #     ax.set_ylim(y_min * 1.1, y_max * 1.1)

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
            cmap="Greys",  # Yellow to brown colormap, you can customize this
            alpha=0.3,
            zorder=1,
        )
        # Set the parameters for x and y-ticks
        ax.tick_params(
            axis="x",
            which="major",
            direction="in",
            labelsize=default_fontsize * 0.7,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )
        ax.tick_params(
            axis="y",
            which="major",
            direction="in",
            labelsize=default_fontsize * 0.7,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )
    # Add an arrow pointing to the sunset end time with appropriate annotation
    axs[1].annotate(
        f"Sunset End at {sunset_end_time.strftime('%H:%M UTC')}",
        xy=(sunset_end_time, (axs[1].get_ylim()[0] + axs[1].get_ylim()[1]) * 0.5),
        xytext=(
            sunset_end_time + datetime.timedelta(minutes=15),
            (axs[1].get_ylim()[0] + axs[1].get_ylim()[1]) * 0.5,
        ),
        arrowprops=dict(facecolor="white", shrink=0.05, width=1, headwidth=8, headlength=10),
        color="k",
        fontsize=default_fontsize * 0.8,
        ha="left",
        va="center",
    )

    # Add a gradient on all the axes after the wake time until the end
    # gradient = np.linspace(0, 1, 256).reshape(1, -1)  # Horizontal gradient
    # for ax in axs:
    #     ax.axvline(
    #         wake_time,
    #         color="yellow",
    #         linestyle="--",
    #     )
    #     ax.imshow(
    #         gradient,
    #         extent=[
    #             wake_time,
    #             data_df.index.max() + pd.Timedelta(minutes=2.5),
    #             ax.get_ylim()[0],
    #             ax.get_ylim()[1],
    #         ],
    #         aspect="auto",
    #         cmap="YlGn",  # Yellow to brown colormap
    #         alpha=0.1,
    #         zorder=1,
    #     )

    # # Add an arrow pointing to the wake time with appropriate annotation
    # axs[1].annotate(
    #     f"Themis {themis_spc.upper()} Wake Begins",
    #     xy=(wake_time, (axs[1].get_ylim()[0] + axs[1].get_ylim()[1]) * 0.5),
    #     xytext=(
    #         wake_time + datetime.timedelta(minutes=5),
    #         (axs[1].get_ylim()[0] + axs[1].get_ylim()[1]) * 0.5,
    #     ),
    #     arrowprops=dict(facecolor="yellow", shrink=0.05, width=1, headwidth=8, headlength=10),
    #     color="yellow",
    #     ha="left",
    #     va="center",
    #     rotation=90,
    # )

    # Find the value of raw_counts_key at max high voltage time
    max_hv_time = pd.Timestamp("2025-03-16 19:03:00", tz="UTC")
    idx = data_df.index.get_indexer([max_hv_time], method="nearest")[0]
    closest_time = data_df.index[idx]
    max_hv_value = data_df["background_flatfield_corrected_slope"].iloc[idx]

    # At the closest time add an arrow pointing to the point of max_hv_value with the text "Max High
    # Voltage"
    max_diff = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]
    axs[0].annotate(
        "Max High Voltage Reached",
        xy=(closest_time, max_hv_value),
        xytext=(closest_time + pd.Timedelta(minutes=15), max_hv_value),
        arrowprops=dict(facecolor="white", shrink=0.05, width=1, headwidth=8, headlength=10),
        color="blue",
        fontsize=default_fontsize * 0.8,
        ha="left",
        va="center",
    )

    # plt.tight_layout()
    # Save the figure
    output_folder = Path("../overleaf_figures/")
    output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_folder / f"line_profile_fit_parameters_time_series_{key}_{time_resolution}.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
        transparent=True,
        pad_inches=0.1,
    )
    # plt.close(fig)

    # # Select the flux data between sunset and wake time
    # sw_flux = selected_flux_data[
    #     (selected_flux_data.index >= sunset_end_time) & (selected_flux_data.index < wake_time)
    # ]
    # hist_counts = data_df["background_flatfield_corrected_total_hist_counts"][
    #     (data_df.index >= sunset_end_time) & (data_df.index <= wake_time)
    # ]
    # # Find the correlation between flux and total counts
    # pearson_correlation, pearson_p_value = stats.pearsonr(sw_flux, hist_counts)
    # print(
    #     f"Correlation between flux and total counts for {key}: {pearson_correlation:.2f} (p-value: {pearson_p_value:.3f})"
    # )

    # spearman_correlation, spearman_p_value = stats.spearmanr(sw_flux, hist_counts)
    # print(
    #     f"Spearman correlation between flux and total counts for {key}: {spearman_correlation:.2f} (p-value: {spearman_p_value:.3f})"
    # )

    # # Set the x-axis limits to zoom in between sunset and wake time
    # axs[0].set_xlim(sunset_end_time, wake_time)
    # axs[1].set_xlim(sunset_end_time, wake_time)
    # axs[2].set_xlim(sunset_end_time, wake_time)
    # # Set the y-axis limits to zoom in between sunset and wake time
    # axs[0].set_ylim(
    #     data_df[f"background_flatfield_corrected_{key}"].loc[sunset_end_time:wake_time].min(),
    #     data_df[f"background_flatfield_corrected_{key}"].loc[sunset_end_time:wake_time].max(),
    # )
    # axs[1].set_ylim(
    #     data_df["background_flatfield_corrected_total_hist_counts"]
    #     .loc[sunset_end_time:wake_time]
    #     .min(),
    #     data_df["background_flatfield_corrected_total_hist_counts"]
    #     .loc[sunset_end_time:wake_time]
    #     .max(),
    # )
    # axs[2].set_ylim(
    #     selected_flux_data.loc[sunset_end_time:wake_time].min(),
    #     selected_flux_data.loc[sunset_end_time:wake_time].max(),
    # )
    # axs[1].set_yscale("log")
    # axs[2].set_yscale("log")
    # # Save the zoomed in figure between sunset and wake time
    # output_folder = Path("../figures/line_profile_fit_parameters/")
    # output_folder.mkdir(parents=True, exist_ok=True)
    # fig.savefig(output_folder / f"line_profile_fit_parameters_time_series_{key}_1min_zoomed.png")
    # plt.close(fig)


# Load the CSV file
data_folder = Path("../data/line_profile_data/bg_corrected/from_l2/")
time_resolution = "1min"
csv_file = data_folder / f"line_profile_fit_parameters_bg_corrected_{time_resolution}.csv"
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
selected_flux_data = flux_df[f"th{themis_spc}_peef_flux"]
# Name the flux column
selected_flux_data.name = f"th{themis_spc}_peef_flux"

keys_to_plot = ["slope"]  # , "intercept", "r_squared", "residuals"]
for key in keys_to_plot:
    plot_fit_parameters(data_df, selected_flux_data, key)
    print(f"Plotted fit parameter time series for key: {key}")
