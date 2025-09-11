import datetime
import importlib
import warnings
from pathlib import Path
from typing import Optional

import lexi_data_analysis_functions_istp as lexi_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from spacepy.pycdf import CDF as cdf

importlib.reload(lexi_functions)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Suppress dvision by zero warnings in numpy
np.seterr(divide="ignore", invalid="ignore")


def get_lexi_and_sw_data(
    df=None,
    h_flat=None,
    flat_field_data=None,
    start_time="2025-03-16T19:45:00Z",
    end_time="2025-03-16T19:50:00Z",
    delta_time=pd.Timedelta("5min"),
    x_key="photon_az",
    y_key="photon_el",
    bins=200,
    bin_range=[263, 281, 15, 33],
    time_normalization=True,
    flat_field_correction=True,
    themis_sc="c",
):
    """
    Function to get the histogram data for the given time range and keys.
    """
    selected_df = df.loc[start_time:end_time]
    input_dict = {
        "df": selected_df,
        # "dat_flat_field": flat_field_data,
        "H_flat": h_flat,
        "x_key": x_key,
        "y_key": y_key,
        "start_time": start_time,
        "end_time": end_time,
        "bins": bins,
        "bin_range": bin_range,
        "time_normalization": time_normalization,
        "flat_field_correction": flat_field_correction,
        "data_folder_location": "data",
        "verbose": True,
    }

    results = lexi_functions.plot_histogram(**input_dict)
    hist = results["H_result"]
    xedges = results["xedges"]
    yedges = results["yedges"]
    # print(
    #     f"The maximum value of the histogram is {np.nanmax(org_hist)}\n The minimum value is {np.nanmin(org_hist)}"
    # )

    x_offset = 271
    y_offset = 24.8

    # Compute the bin centres
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    # Create a meshgrid for the bin centers
    X, Y = np.meshgrid(x_centers, y_centers)

    # Compute the distances from the center
    distances = np.sqrt((X - x_offset) ** 2 + (Y - y_offset) ** 2)

    # Create a mask of the points within the circle of radius 4.5
    mask = distances <= 4.5

    # Sum the histogram data within the masked region
    hist_sum = np.nansum(hist[mask])

    hist_sum_no_mask = np.nansum(hist)

    # print(f"The sum of the histogram data within the circle is {hist_sum:.2f}\n ")
    # print(f"The sum of the histogram data without any mask is {hist_sum_no_mask:.2f}\n ")

    # Depending on the themis spacecraft, get the corresponding data
    themis_file_name = f"../data/themis_data/csv/themis_{themis_sc}_esa_parameters_2025-03-16_to_2025-03-17_flux.csv"
    themis_df = pd.read_csv(themis_file_name)
    # Rename the first column to 'epoch' and set it as the index
    themis_df.rename(columns={themis_df.columns[0]: "epoch"}, inplace=True)
    themis_df.set_index("epoch", inplace=True)
    # Convert the index to datetime
    themis_df.index = pd.to_datetime(themis_df.index, utc=True)

    # Columns to select
    select_columns = [
        f"th{themis_sc}_peer_density",
        f"th{themis_sc}_peef_density",
        f"th{themis_sc}_peeb_density",
        f"th{themis_sc}_peir_velocity_magnitude",
        f"th{themis_sc}_peif_velocity_magnitude",
        f"th{themis_sc}_peib_velocity_magnitude",
        f"th{themis_sc}_peir_flux",
        f"th{themis_sc}_peif_flux",
        f"th{themis_sc}_peib_flux",
    ]

    # Select the columns from the dataframe for the given time range
    themis_df = themis_df.loc[start_time:end_time, select_columns]

    # Remove rows where all values are NaN
    themis_df.dropna(how="all", inplace=True)

    # Get the median and mean values of the selected columns
    themis_df_mean = themis_df.mean()
    themis_df_median = themis_df.median()

    # Get the median time of the selected columns
    themis_df_median_time = themis_df.index[
        themis_df.index.get_loc(themis_df.index[len(themis_df) // 2])
    ]

    return hist_sum, themis_df_mean, themis_df_median, themis_df_median_time, hist_sum_no_mask


def plot_themis_lexi_hist_time_series(
    time_series: Optional[pd.DatetimeIndex] = None,
    sw_np: Optional[pd.Series] = None,
    sw_vp: Optional[pd.Series] = None,
    sw_flux: Optional[pd.Series] = None,
    hist_sum: Optional[pd.Series] = None,
    hist_sum_no_mask: Optional[pd.Series] = None,
    themis_sc: str = "c",
    freq: str = "5seconds",
    integration_time: str = "5minutes",
    x_limit: Optional[tuple] = None,
):
    """
    Function to plot the THEMIS and LEXI histogram data.
    """

    # if any(x is None for x in [time_series, sw_np, sw_vp, sw_flux, hist_sum, hist_sum_no_mask]):
    #     print("Missing data for plotting.")
    # return

    # Set the dark mode for the plot
    mpl.style.use("dark_background")
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
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.suptitle(
        f"THEMIS {themis_sc} ESA Parameters and LEXI Histogram Data\n{start_time} to {end_time}",
        fontsize=16,
    )
    # Plot the THEMIS data
    ax[0].plot(time_series, sw_np, label="Density (cm^-3)", color="blue")
    ax[0].set_ylabel("Density [cm^-3]", fontsize=14)
    ax[0].legend(loc="upper right", fontsize=10)
    ax[0].grid(True, linestyle="--", alpha=0.5)
    # ax[0].set_title(f"THEMIS {themis_sc} ESA Density", fontsize=16)
    if themis_sc == "c":
        ax[0].set_ylim(2, 6)
    elif themis_sc == "b":
        ax[0].set_ylim(3, 8)
    ax[0].set_yscale("linear")

    # Plot the THEMIS velocity data
    ax[1].plot(time_series, sw_vp, label="Velocity (km/s)", color="orange")
    ax[1].set_ylabel("Velocity [km/s]", fontsize=14)
    ax[1].legend(loc="upper right", fontsize=10)
    ax[1].grid(True, linestyle="--", alpha=0.5)
    # ax[1].set_title(f"THEMIS {themis_sc} ESA Velocity", fontsize=16)
    if themis_sc == "c":
        ax[1].set_ylim(300, 380)
    elif themis_sc == "b":
        ax[1].set_ylim(300, 380)

    # Plot the THEMIS flux data
    ax[2].plot(time_series, sw_np * sw_vp, label="np * vp", color="green")
    ax[2].plot(time_series, sw_flux, label="Flux (km cm^-3 s^-1)", color="b")
    ax[2].set_ylabel("Flux [km cm^-3 s^-1]", fontsize=14)
    ax[2].legend(loc="upper right", fontsize=10)
    ax[2].grid(True, linestyle="--", alpha=0.5)
    # ax[2].set_title(f"THEMIS {themis_sc} ESA Flux", fontsize=16)
    if themis_sc == "c":
        ax[2].set_ylim(6e2, 1.8e3)
    elif themis_sc == "b":
        ax[2].set_ylim(6e2, 3e3)
    ax[2].set_yscale("log")

    # Plot the LEXI histogram data
    ax[3].plot(time_series, hist_sum, label="Masked", color="red")
    twin_ax3 = ax[3].twinx()
    twin_ax3.plot(time_series, hist_sum_no_mask, label="Non-masked", color="orange")
    ax[3].set_ylabel("Histogram Sum", fontsize=14)
    twin_ax3.set_ylabel("Histogram Sum (No Mask)", fontsize=14)
    ax[3].legend(loc="upper left", fontsize=10)
    twin_ax3.legend(loc="upper right", fontsize=10)
    ax[3].grid(True, linestyle="--", alpha=0.5)
    # ax[3].set_title(f"LEXI Histogram Sum", fontsize=16)
    ax[3].set_xlabel("Time [UTC]", fontsize=14)

    # Set y-axis spine color
    twin_ax3.spines["left"].set_color("red")
    twin_ax3.spines["right"].set_color("orange")

    # Set the y-axis label colors
    ax[3].yaxis.label.set_color("red")
    twin_ax3.yaxis.label.set_color("orange")

    # Get the correlation coefficient between THEMIS flux and LEXI histogram sum
    if sw_flux is not None and hist_sum is not None:
        # select the data only between the start and end time
        start_time_timestamp = pd.to_datetime(start_time, utc=True)
        end_time_timestamp = pd.to_datetime(end_time, utc=True)
        sw_flux = sw_flux[
            (time_series >= start_time_timestamp) & (time_series <= end_time_timestamp)
        ]
        hist_sum = hist_sum[
            (time_series >= start_time_timestamp) & (time_series <= end_time_timestamp)
        ]
        hist_sum_no_mask = hist_sum_no_mask[
            (time_series >= start_time_timestamp) & (time_series <= end_time_timestamp)
        ]
        # Calculate the Pearson correlation coefficient for hist_sum
        if len(sw_flux) == 0 or len(hist_sum) == 0:
            print("No data available for correlation calculation.")
            correlation = np.nan
            spearman_correlation = np.nan
        elif len(sw_flux) != len(hist_sum):
            print(
                "Warning: The length of sw_flux and hist_sum are not equal. "
                "Correlation calculation may not be accurate."
            )
            min_length = min(len(sw_flux), len(hist_sum))
            sw_flux = sw_flux[:min_length]
            hist_sum = hist_sum[:min_length]
        else:
            # Calculate the Pearson correlation coefficient
            if np.all(np.isnan(sw_flux)) or np.all(np.isnan(hist_sum)):
                print("All values are NaN, cannot calculate correlation.")
                correlation = np.nan
                spearman_correlation = np.nan
        correlation = np.corrcoef(sw_flux, hist_sum)[0, 1]

        # Get the spearman correlation
        spearman_correlation = stats.spearmanr(sw_flux, hist_sum).correlation
        # ax[3].text(
        #     0.05,
        #     0.95,
        #     f"Pearson Correlation: {correlation:.2f}\nSpearman Correlation: {spearman_correlation:.2f}",
        #     transform=ax[3].transAxes,
        #     fontsize=12,
        #     verticalalignment="top",
        # )
    # Get the correlation coefficient for hist_sum_no_mask
    if sw_flux is not None and hist_sum_no_mask is not None:
        # select the data only between the start and end time
        start_time_timestamp = pd.to_datetime(start_time, utc=True)
        end_time_timestamp = pd.to_datetime(end_time, utc=True)
        # sw_flux = sw_flux[
        #     (time_series >= start_time_timestamp) & (time_series <= end_time_timestamp)
        # ]
        hist_sum_no_mask = hist_sum_no_mask[
            (time_series[:] >= start_time_timestamp) & (time_series[:] <= end_time_timestamp)
        ]
        # Calculate the Pearson correlation coefficient for hist_sum_no_mask
        if len(sw_flux) == 0 or len(hist_sum_no_mask) == 0:
            print("No data available for correlation calculation.")
            correlation_no_mask = np.nan
            spearman_correlation_no_mask = np.nan
        elif len(sw_flux) != len(hist_sum_no_mask):
            print(
                "Warning: The length of sw_flux and hist_sum_no_mask are not equal. "
                "Correlation calculation may not be accurate."
            )
            min_length = min(len(sw_flux), len(hist_sum_no_mask))
            sw_flux = sw_flux[:min_length]
            hist_sum_no_mask = hist_sum_no_mask[:min_length]
        else:
            # Calculate the Pearson correlation coefficient
            if np.all(np.isnan(sw_flux)) or np.all(np.isnan(hist_sum_no_mask)):
                print("All values are NaN, cannot calculate correlation.")
                correlation_no_mask = np.nan
                spearman_correlation_no_mask = np.nan
        correlation_no_mask = np.corrcoef(sw_flux, hist_sum_no_mask)[0, 1]

        # Get the spearman correlation
        spearman_correlation_no_mask = stats.spearmanr(sw_flux, hist_sum_no_mask).correlation

        ax[3].text(
            0.05,
            0.05,
            f"Pearson Correlation (Mask): {correlation:.2f}\nSpearman Correlation (Mask): {spearman_correlation:.2f}",
            transform=ax[3].transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="left",
        )

        ax[3].text(
            0.95,
            0.05,
            f"Pearson Correlation (No Mask): {correlation_no_mask:.2f}\nSpearman Correlation (No Mask): {spearman_correlation_no_mask:.2f}",
            transform=ax[3].transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
        )
    # Set the x-axis limits if provided
    if x_limit is not None:
        # Convert x_limit to datetime if it is not already
        x_limit = pd.to_datetime(x_limit, utc=True)
        ax[3].set_xlim(x_limit)
    elif time_series is not None and len(time_series) > 0:
        # Set the x-axis limits to the start and end time of the time series
        ax[3].set_xlim([time_series[0], time_series[-1]])
    ax[3].set_ylim(0.9 * np.nanmin(hist_sum), 1.1 * np.nanmax(hist_sum))
    twin_ax3.set_ylim(0.9 * np.nanmin(hist_sum_no_mask), 1.1 * np.nanmax(hist_sum_no_mask))
    ax[3].set_yscale("log")
    twin_ax3.set_yscale("log")
    folder_path = Path(f"../figures/lexi_sw/istp/themis_{themis_sc}/")
    folder_path.mkdir(parents=True, exist_ok=True)
    file_name = f"themis_{themis_sc}_esa_parameters_{start_time}_to_{end_time}_histogram_sum_{freq}_{integration_time}.png"
    file_path = folder_path / file_name
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    # close the figure
    plt.close(fig)
    print(f"Figure saved to {file_path}")
    return None


def plot_themis_lexi_hist_time_series_shifted(
    time_series: Optional[pd.DatetimeIndex] = None,
    sw_np: Optional[pd.Series] = None,
    sw_vp: Optional[pd.Series] = None,
    sw_flux: Optional[pd.Series] = None,
    hist_sum: Optional[pd.Series] = None,
    hist_sum_no_mask: Optional[pd.Series] = None,
    themis_sc: str = "c",
    freq: str = "5seconds",
    integration_time: str = "5minutes",
    x_limit: Optional[tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """
    Function to plot the THEMIS and LEXI histogram data, with time-shifted overlays.
    """

    # Set dark theme
    mpl.style.use("dark_background")
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

    # Create shifted versions of hist_sum
    hist_sum_shift_4min = hist_sum.copy()
    hist_sum_shift_4min.index += datetime.timedelta(minutes=4)

    hist_sum_shift_5min = hist_sum.copy()
    hist_sum_shift_5min.index += datetime.timedelta(minutes=4)

    hist_sum_shift_7min = hist_sum.copy()
    hist_sum_shift_7min.index += datetime.timedelta(minutes=7)

    # Start figure
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.suptitle(
        f"THEMIS {themis_sc} ESA Parameters and LEXI Histogram Data\n{start_time} to {end_time}",
        fontsize=16,
    )
    # ----------- DENSITY -----------
    ax[0].plot(time_series, sw_np, label="Density (cm^-3)", color="blue")
    twin_ax0 = ax[0].twinx()
    hist_sum_shift_4min.index = pd.to_datetime(hist_sum_shift_4min.index) + datetime.timedelta(0)
    twin_ax0.plot(
        hist_sum_shift_4min.index,
        hist_sum_shift_4min,
        label="Hist Sum (4m shift)",
        color="red",
        alpha=0.6,
    )
    ax[0].set_ylabel("Density [cm^-3]", fontsize=14)
    twin_ax0.set_ylabel("Hist Sum", fontsize=14, color="red")
    twin_ax0.tick_params(axis="y", colors="red")
    twin_ax0.spines["right"].set_color("red")
    ax[0].legend(loc="upper left", fontsize=10)
    twin_ax0.legend(loc="upper right", fontsize=10)
    ax[0].grid(True, linestyle="--", alpha=0.5)
    ax[0].set_ylim(2, 6)

    # ----------- VELOCITY -----------
    ax[1].plot(time_series, sw_vp, label="Velocity (km/s)", color="orange")
    twin_ax1 = ax[1].twinx()
    hist_sum_shift_5min.index = pd.to_datetime(hist_sum_shift_5min.index) + datetime.timedelta(0)
    twin_ax1.plot(
        hist_sum_shift_5min.index,
        hist_sum_shift_5min,
        label="Hist Sum (5m shift)",
        color="red",
        alpha=0.6,
    )
    ax[1].set_ylabel("Velocity [km/s]", fontsize=14)
    twin_ax1.set_ylabel("Hist Sum", fontsize=14, color="red")
    twin_ax1.tick_params(axis="y", colors="red")
    twin_ax1.spines["right"].set_color("red")
    ax[1].legend(loc="upper left", fontsize=10)
    twin_ax1.legend(loc="upper right", fontsize=10)
    ax[1].grid(True, linestyle="--", alpha=0.5)
    ax[1].set_ylim(300, 380)

    # ----------- FLUX -----------
    ax[2].plot(time_series, sw_np * sw_vp, label="np * vp", color="green")
    ax[2].plot(time_series, sw_flux, label="Flux (km cm^-3 s^-1)", color="blue")
    twin_ax2 = ax[2].twinx()
    hist_sum_shift_7min.index = pd.to_datetime(hist_sum_shift_7min.index) + datetime.timedelta(0)
    twin_ax2.plot(
        hist_sum_shift_7min.index,
        hist_sum_shift_7min,
        label="Hist Sum (7m shift)",
        color="red",
        alpha=0.6,
    )
    ax[2].set_ylabel("Flux [km cm^-3 s^-1]", fontsize=14)
    twin_ax2.set_ylabel("Hist Sum", fontsize=14, color="red")
    twin_ax2.tick_params(axis="y", colors="red")
    twin_ax2.spines["right"].set_color("red")
    ax[2].legend(loc="upper left", fontsize=10)
    twin_ax2.legend(loc="upper right", fontsize=10)
    ax[2].grid(True, linestyle="--", alpha=0.5)
    ax[2].set_ylim(6e2, 1.8e3)
    ax[2].set_yscale("log")

    # Plot Histogram sum and no mask
    ax[3].plot(time_series, hist_sum, label="Masked", color="red")
    twin_ax3 = ax[3].twinx()
    twin_ax3.plot(time_series, hist_sum_no_mask, label="Non-masked", color="orange")
    ax[3].set_ylabel("Histogram Sum", fontsize=14)
    twin_ax3.set_ylabel("Histogram Sum (No Mask)", fontsize=14)
    ax[3].legend(loc="upper left", fontsize=10)
    twin_ax3.legend(loc="upper right", fontsize=10)
    ax[3].grid(True, linestyle="--", alpha=0.5)
    ax[3].set_xlabel("Time [UTC]", fontsize=14)
    ax[3].set_yscale("log")
    twin_ax3.set_yscale("log")
    ax[3].spines["left"].set_color("red")
    twin_ax3.spines["right"].set_color("orange")
    ax[3].yaxis.label.set_color("red")
    twin_ax3.yaxis.label.set_color("orange")

    # Correlation calculations
    if sw_flux is not None and hist_sum is not None:
        ts = time_series
        # Set the timezone to UTC for start and end times
        start_time = pd.to_datetime(start_time, utc=True)
        end_time = pd.to_datetime(end_time, utc=True)
        sw_flux_cut = sw_flux[(ts >= start_time) & (ts <= end_time)]
        hist_sum_cut = hist_sum[(ts >= start_time) & (ts <= end_time)]
        hist_sum_no_mask_cut = hist_sum_no_mask[(ts >= start_time) & (ts <= end_time)]

        min_len = min(len(sw_flux_cut), len(hist_sum_cut), len(hist_sum_no_mask_cut))
        sw_flux_cut = sw_flux_cut[:min_len]
        hist_sum_cut = hist_sum_cut[:min_len]
        hist_sum_no_mask_cut = hist_sum_no_mask_cut[:min_len]

        if min_len > 0 and not (np.all(np.isnan(sw_flux_cut)) or np.all(np.isnan(hist_sum_cut))):
            correlation = np.corrcoef(sw_flux_cut, hist_sum_cut)[0, 1]
            spearman_correlation = stats.spearmanr(sw_flux_cut, hist_sum_cut).correlation
        else:
            correlation = spearman_correlation = np.nan

        if min_len > 0 and not (
            np.all(np.isnan(sw_flux_cut)) or np.all(np.isnan(hist_sum_no_mask_cut))
        ):
            correlation_no_mask = np.corrcoef(sw_flux_cut, hist_sum_no_mask_cut)[0, 1]
            spearman_correlation_no_mask = stats.spearmanr(
                sw_flux_cut, hist_sum_no_mask_cut
            ).correlation
        else:
            correlation_no_mask = spearman_correlation_no_mask = np.nan

        ax[3].text(
            0.05,
            0.05,
            f"Pearson Correlation (Mask): {correlation:.2f}\nSpearman Correlation (Mask): {spearman_correlation:.2f}",
            transform=ax[3].transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        ax[3].text(
            0.95,
            0.05,
            f"Pearson Correlation (No Mask): {correlation_no_mask:.2f}\nSpearman Correlation (No Mask): {spearman_correlation_no_mask:.2f}",
            transform=ax[3].transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    # Set x-axis limits
    if x_limit is not None:
        ax[3].set_xlim(pd.to_datetime(x_limit, utc=True))
    elif time_series is not None and len(time_series) > 0:
        ax[3].set_xlim([time_series[0], time_series[-1]])

    # Auto y-limits for log axis
    ax[3].set_ylim(0.9 * np.nanmin(hist_sum), 1.1 * np.nanmax(hist_sum))
    twin_ax3.set_ylim(0.9 * np.nanmin(hist_sum_no_mask), 1.1 * np.nanmax(hist_sum_no_mask))

    # Save figure
    folder_path = Path(f"../figures/lexi_sw/istp/themis_{themis_sc}/")
    folder_path.mkdir(parents=True, exist_ok=True)
    file_name = f"themis_{themis_sc}_esa_parameters_{start_time}_to_{end_time}_histogram_sum_{freq}_{integration_time}_shifted.png"
    file_path = folder_path / file_name
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Figure saved to {file_path}")


def get_h_flat():
    print("Reading flat field data...")
    flat_field_path = "data/flat_field_data/lexi_l1c_flat_field_data_20240524_20240530.cdf"
    dat_flat_field = cdf(flat_field_path)

    flat_x = dat_flat_field["photon_az"][...]
    flat_y = dat_flat_field["photon_el"][...]
    flat_epoch = pd.to_datetime(dat_flat_field["Epoch"][...])
    flat_index = pd.DatetimeIndex(flat_epoch).tz_localize("UTC")
    flat_duration = (flat_index[-1] - flat_index[0]).total_seconds()
    flat_weights = np.ones_like(flat_x) / flat_duration
    H_flat, _, _ = np.histogram2d(
        flat_x,
        flat_y,
        bins=200,
        range=[[263, 281], [15, 33]],
        weights=flat_weights,
    )
    return H_flat


start_time = "2025-03-16T19:30:00"
end_time = "2025-03-16T21:15:00"

# Get the THEMIS and LEXI histogram data in 1 minute intervals
freq = "1min"
themis_sc = "c"
delta_time = pd.Timedelta("5min")

recompute_data = False
if recompute_data:
    # Print the current time in utc
    print(
        f"Code execution time in UTC: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    # Get the flat field data
    H_flat = get_h_flat()

    time_series = pd.date_range(start=start_time, end=end_time, freq=freq)
    hist_sum_list = []
    hist_sum_no_mask_list = []
    sw_np_list = []
    sw_vp_list = []
    sw_flux_list = []
    time_series_list = []

    all_df = lexi_functions.read_all_data_files(
        file_list=None,
        start_time=start_time,
        end_time=end_time,
        return_data_type="dataframe",
        kwargs={
            "data_folder_location": "data",
            "version": "latest",
            "start_time": start_time,
            "end_time": end_time,
        },
        verbose=True,
    )
    for i, time_val in enumerate(time_series[:]):
        # Print the progress bar as percentage
        progress = (i + 1) / len(time_series) * 100
        print(f"Progress ==> {progress:.6f}%", end="\r")
        start = time_val.isoformat()
        end = (time_val + delta_time).isoformat()
        try:
            # for jjj in range(1):
            hist_sum, themis_df_mean, themis_df_median, themis_df_median_time, hist_sum_no_mask = (
                get_lexi_and_sw_data(
                    df=all_df,
                    h_flat=H_flat,
                    start_time=start,
                    end_time=end,
                    themis_sc=themis_sc,
                )
            )
        except Exception:
            continue
        hist_sum_list.append(hist_sum)
        hist_sum_no_mask_list.append(hist_sum_no_mask)
        sw_np = themis_df_mean[f"th{themis_sc}_peer_density"]
        sw_vp = themis_df_mean[f"th{themis_sc}_peir_velocity_magnitude"]
        sw_flux = themis_df_mean[f"th{themis_sc}_peir_flux"]
        time_series_val = themis_df_median_time

        sw_np_list.append(sw_np)
        sw_vp_list.append(sw_vp)
        sw_flux_list.append(sw_flux)
        time_series_list.append(time_series_val)

    # Convert the lists to numpy arrays
    hist_sum = np.array(hist_sum_list)
    hist_sum_no_mask = np.array(hist_sum_no_mask_list)
    sw_np = np.array(sw_np_list)
    sw_vp = np.array(sw_vp_list)
    sw_flux = np.array(sw_flux_list)
    time_series_array = np.array(time_series_list)

    # Save these values to a CSV file
    df = pd.DataFrame(
        {
            "time_series": time_series_array,
            "sw_np": sw_np,
            "sw_vp": sw_vp,
            "sw_flux": sw_flux,
            "hist_sum": hist_sum,
            "hist_sum_no_mask": hist_sum_no_mask,
        }
    )
    folder_path = Path("../data/lexi_sw/istp/")
    folder_path.mkdir(parents=True, exist_ok=True)
    delta_time_str = int(delta_time.total_seconds())
    file_name = f"istp_themis_{themis_sc}_esa_parameters_{start_time}_to_{end_time}_histogram_sum_{freq}_{delta_time_str}.csv"
    file_path = folder_path / file_name
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

else:
    folder_path = Path("../data/lexi_sw/istp/")
    delta_time_str = int(delta_time.total_seconds())
    file_name = f"istp_themis_{themis_sc}_esa_parameters_{start_time}_to_{end_time}_histogram_sum_{freq}_{delta_time_str}.csv"
    file_path = folder_path / file_name
    # Load the data from the CSV file
    df = pd.read_csv(file_path)
    time_series = pd.to_datetime(df["time_series"], utc=True)
    sw_np = df["sw_np"].values
    sw_vp = df["sw_vp"].values
    sw_flux = df["sw_flux"].values
    hist_sum = df["hist_sum"].values
    hist_sum_no_mask = df["hist_sum_no_mask"].values


def plot_themis_lexi_hist_time_series_dtw(
    time_series: Optional[pd.DatetimeIndex] = None,
    sw_np: Optional[pd.Series] = None,
    sw_vp: Optional[pd.Series] = None,
    sw_flux: Optional[pd.Series] = None,
    hist_sum: Optional[pd.Series] = None,
    hist_sum_no_mask: Optional[pd.Series] = None,
    themis_sc: str = "c",
    freq: str = "5seconds",
    integration_time: str = "5minutes",
    x_limit: Optional[tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    resample_freq: str = "1min",
):
    from dtaidistance import dtw

    # ==== Prepare dataframe ====
    df = pd.DataFrame(
        {
            "time_series": time_series,
            "sw_np": sw_np,
            "sw_vp": sw_vp,
            "sw_flux": sw_flux,
            "hist_sum": hist_sum,
            "hist_sum_no_mask": hist_sum_no_mask,
        }
    ).dropna()
    df = df.set_index("time_series").sort_index()

    # ==== DTW Function ====
    def dtw_align(reference: pd.Series, target: pd.Series) -> pd.Series:
        ref = reference.resample(resample_freq).mean().dropna()
        tgt = target.resample(resample_freq).mean().dropna()
        common = ref.index.intersection(tgt.index)
        if len(common) < 10:
            return pd.Series(index=ref.index, data=np.nan)
        ref = ref.loc[common]
        tgt = tgt.loc[common]
        x = (ref - ref.mean()) / ref.std()
        y = (tgt - tgt.mean()) / tgt.std()
        _, paths = dtw.warping_paths(x.values, y.values)
        path = dtw.best_path(paths)
        warped_values = y.values[[j for i, j in path]]
        warped_index = x.index[[i for i, j in path]]
        return pd.Series(index=warped_index, data=warped_values)

    # ==== Apply DTW for np, vp, flux ====
    dtw_np = dtw_align(df["sw_np"], df["hist_sum"])
    dtw_vp = dtw_align(df["sw_vp"], df["hist_sum"])
    dtw_flux = dtw_align(df["sw_flux"], df["hist_sum"])

    # ==== Plot Styling ====
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.style.use("dark_background")
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

    # ==== Plot ====
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(
        f"THEMIS {themis_sc} ESA Parameters and DTW-Aligned LEXI Histogram\n{start_time} to {end_time}",
        fontsize=16,
    )

    # ---- DENSITY ----
    ax[0].plot(df.index, df["sw_np"], label="Density (cm⁻³)", color="blue")
    twin0 = ax[0].twinx()
    twin0.plot(dtw_np.index, dtw_np, label="DTW Hist Sum (np)", color="red", alpha=0.6)
    ax[0].set_ylabel("Density", fontsize=14)
    twin0.set_ylabel("Hist Sum", fontsize=14, color="red")
    twin0.spines["right"].set_color("red")
    twin0.tick_params(axis="y", colors="red")
    ax[0].legend(loc="upper left")
    twin0.legend(loc="upper right")
    ax[0].grid(True, linestyle="--", alpha=0.5)

    # ---- VELOCITY ----
    ax[1].plot(df.index, df["sw_vp"], label="Velocity (km/s)", color="orange")
    twin1 = ax[1].twinx()
    twin1.plot(dtw_vp.index, dtw_vp, label="DTW Hist Sum (vp)", color="red", alpha=0.6)
    ax[1].set_ylabel("Velocity", fontsize=14)
    twin1.set_ylabel("Hist Sum", fontsize=14, color="red")
    twin1.spines["right"].set_color("red")
    twin1.tick_params(axis="y", colors="red")
    ax[1].legend(loc="upper left")
    twin1.legend(loc="upper right")
    ax[1].grid(True, linestyle="--", alpha=0.5)

    # ---- FLUX ----
    ax[2].plot(df.index, df["sw_flux"], label="Flux", color="green")
    twin2 = ax[2].twinx()
    twin2.plot(dtw_flux.index, dtw_flux, label="DTW Hist Sum (flux)", color="red", alpha=0.6)
    ax[2].set_ylabel("Flux", fontsize=14)
    twin2.set_ylabel("Hist Sum", fontsize=14, color="red")
    twin2.spines["right"].set_color("red")
    twin2.tick_params(axis="y", colors="red")
    ax[2].legend(loc="upper left")
    twin2.legend(loc="upper right")
    ax[2].grid(True, linestyle="--", alpha=0.5)
    ax[2].set_yscale("log")

    # ---- HISTOGRAM SUM ----
    ax[3].plot(df.index, df["hist_sum"], label="Masked", color="red")
    twin3 = ax[3].twinx()
    twin3.plot(df.index, df["hist_sum_no_mask"], label="Non-masked", color="orange")
    ax[3].set_ylabel("Hist Sum", fontsize=14)
    twin3.set_ylabel("Hist Sum (No Mask)", fontsize=14, color="orange")
    twin3.spines["right"].set_color("orange")
    twin3.tick_params(axis="y", colors="orange")
    ax[3].legend(loc="upper left")
    twin3.legend(loc="upper right")
    ax[3].grid(True, linestyle="--", alpha=0.5)
    ax[3].set_xlabel("Time [UTC]", fontsize=14)
    ax[3].set_yscale("log")
    twin3.set_yscale("log")

    # X-axis limits
    if x_limit is not None:
        ax[3].set_xlim(pd.to_datetime(x_limit, utc=True))
    else:
        ax[3].set_xlim([df.index[0], df.index[-1]])

    # Save
    from pathlib import Path

    out_dir = Path(f"../figures/lexi_sw/istp/themis_{themis_sc}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"themis_{themis_sc}_esa_dtw_all_{start_time}_to_{end_time}.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Figure saved to {out_path}")


def plot_themis_lexi_hist_time_series_dtw_with_path(
    time_series: pd.DatetimeIndex,
    sw_np: pd.Series,
    sw_vp: pd.Series,
    sw_flux: pd.Series,
    hist_sum: pd.Series,
    hist_sum_no_mask: pd.Series,
    themis_sc: str = "c",
    freq: str = "5seconds",
    integration_time: str = "5minutes",
    x_limit: tuple = None,
    start_time: str = None,
    end_time: str = None,
):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(
        f"THEMIS {themis_sc} ESA Parameters and DTW-Aligned LEXI Histogram\n{start_time} to {end_time}",
        fontsize=16,
    )

    # Helper function to warp series using DTW
    def warp_series(ref_series, target_series):
        path = dtw.warping_path(ref_series.values, target_series.values)
        warped = pd.Series(
            [target_series.values[j] for i, j in path],
            index=[ref_series.index[i] for i, j in path],
        )
        return warped, path

    # DTW alignments
    hist_sum_dtw_np, path_np = warp_series(sw_np, hist_sum)
    hist_sum_dtw_vp, path_vp = warp_series(sw_vp, hist_sum)
    hist_sum_dtw_flux, path_flux = warp_series(sw_flux, hist_sum)

    series_groups = [
        (sw_np, hist_sum_dtw_np, path_np, ax[0], "Density", "cm^-3", "blue"),
        (sw_vp, hist_sum_dtw_vp, path_vp, ax[1], "Velocity", "km/s", "orange"),
        (sw_flux, hist_sum_dtw_flux, path_flux, ax[2], "Flux", "km cm^-3 s^-1", "green"),
    ]

    for ref_series, target_series, path, ax_base, label, units, color in series_groups:
        ax_base.plot(ref_series.index, ref_series, label=f"{label}", color=color)
        ax_base.set_ylabel(f"{label} [{units}]", fontsize=12)
        ax_base.legend(loc="upper right", fontsize=10)
        # ax_base.grid(True, linestyle="--", alpha=0.5)

        ax_twin = ax_base.twinx()
        ax_twin.plot(target_series.index, target_series, label="DTW Hist Sum", color="red")
        ax_twin.set_ylabel("Hist Sum", color="red", fontsize=12)
        ax_twin.tick_params(axis="y", colors="red")
        ax_twin.legend(loc="upper left", fontsize=10)

        # Plot warping path every 10th point
        for count, (i, j) in enumerate(path):
            if count % 10 != 0:
                continue
            try:
                t_ref = ref_series.index[i]
                t_tgt = target_series.index[j]
                v_ref = ref_series.iloc[i]
                v_tgt = target_series.iloc[j]
                ax_base.plot(
                    [t_ref, t_tgt],
                    [v_ref, v_tgt],
                    color="white",
                    alpha=0.3,
                    linestyle="--",
                    linewidth=0.7,
                )
            except IndexError:
                continue

    # Plot original hist sums
    ax[3].plot(time_series, hist_sum, label="Masked", color="red")
    twin_ax3 = ax[3].twinx()
    twin_ax3.plot(time_series, hist_sum_no_mask, label="Non-masked", color="orange")
    ax[3].set_ylabel("Hist Sum", fontsize=12)
    twin_ax3.set_ylabel("Hist Sum (No Mask)", fontsize=12)
    ax[3].legend(loc="upper left", fontsize=10)
    twin_ax3.legend(loc="upper right", fontsize=10)
    ax[3].grid(True, linestyle="--", alpha=0.5)
    ax[3].set_xlabel("Time [UTC]", fontsize=14)

    if x_limit is not None:
        x_limit = pd.to_datetime(x_limit, utc=True)
        ax[3].set_xlim(x_limit)
    elif time_series is not None and len(time_series) > 0:
        ax[3].set_xlim([time_series[0], time_series[-1]])

    folder_path = Path(f"../figures/lexi_sw/istp/themis_{themis_sc}/")
    folder_path.mkdir(parents=True, exist_ok=True)
    file_name = f"themis_{themis_sc}_esa_dtw_warp_path_{start_time}_to_{end_time}.png"
    file_path = folder_path / file_name
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Figure saved to {file_path}")


def apply_dtw_and_warp(hist_sum_series, themis_series):
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_x.fit_transform(themis_series.values.reshape(-1, 1)).ravel()
    Y = scaler_y.fit_transform(hist_sum_series.values.reshape(-1, 1)).ravel()

    path = dtw.warping_path(X, Y)

    aligned_hist_sum = pd.Series(index=themis_series.index, dtype=float)
    for i, j in path:
        aligned_hist_sum.iloc[i] = hist_sum_series.iloc[j]

    aligned_hist_sum = aligned_hist_sum.interpolate(method="linear").ffill().bfill()
    return aligned_hist_sum, path


def plot_themis_lexi_dtw(
    time_series: pd.DatetimeIndex,
    sw_np: pd.Series,
    sw_vp: pd.Series,
    sw_flux: pd.Series,
    hist_sum: pd.Series,
    hist_sum_no_mask: pd.Series,
    themis_sc: str = "c",
    freq: str = "5seconds",
    integration_time: str = "5minutes",
    x_limit: Optional[tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    mpl.style.use("dark_background")
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

    hist_dtw_np, path_np = apply_dtw_and_warp(hist_sum, sw_np)
    hist_dtw_vp, path_vp = apply_dtw_and_warp(hist_sum, sw_vp)
    hist_dtw_flux, path_flux = apply_dtw_and_warp(hist_sum, sw_flux)

    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.suptitle(
        f"THEMIS {themis_sc} ESA Parameters and DTW-Aligned LEXI Histogram\n{start_time} to {end_time}",
        fontsize=16,
    )

    def plot_main_and_twin(ax_main, x, y_main, label_main, color_main, y_dtw, label_dtw):
        ax_main.plot(x, y_main, label=label_main, color=color_main)
        ax_main.set_ylabel(label_main, fontsize=14)
        ax_main.legend(loc="upper right", fontsize=10)
        # ax_main.grid(True, linestyle="--", alpha=0.5)

        ax_twin = ax_main.twinx()
        ax_twin.plot(x, y_dtw, label=label_dtw, color="red")
        ax_twin.set_ylabel("Hist Sum", color="red", fontsize=12)
        ax_twin.tick_params(axis="y", colors="red")
        ax_twin.legend(loc="upper left", fontsize=10)
        return ax_twin

    twin0 = plot_main_and_twin(
        ax[0], time_series, sw_np, "Density [cm^-3]", "blue", hist_dtw_np, "DTW Hist Sum"
    )
    twin1 = plot_main_and_twin(
        ax[1], time_series, sw_vp, "Velocity [km/s]", "orange", hist_dtw_vp, "DTW Hist Sum"
    )
    twin2 = plot_main_and_twin(
        ax[2], time_series, sw_flux, "Flux [km cm^-3 s^-1]", "green", hist_dtw_flux, "DTW Hist Sum"
    )

    ax[2].set_yscale("log")
    ax[3].set_yscale("log")
    twin0.set_yscale("log")
    twin1.set_yscale("log")
    twin2.set_yscale("log")

    # LEXI histogram plot
    ax[3].plot(time_series, hist_sum, label="Masked", color="red")
    twin3 = ax[3].twinx()
    twin3.plot(time_series, hist_sum_no_mask, label="Non-masked", color="orange")
    ax[3].set_ylabel("Hist Sum", fontsize=14)
    twin3.set_ylabel("Hist Sum (No Mask)", fontsize=14)
    ax[3].legend(loc="upper left", fontsize=10)
    twin3.legend(loc="upper right", fontsize=10)
    # ax[3].grid(True, linestyle="--", alpha=0.5)
    ax[3].set_xlabel("Time [UTC]", fontsize=14)

    # Warping path (every 10th point)
    for i, j in path_np[::10]:
        ax[0].plot(
            [time_series[i], time_series[i]],
            [sw_np.iloc[i], hist_sum.iloc[j]],
            color="white",
            alpha=0.5,
            lw=0.5,
        )
    for i, j in path_vp[::10]:
        ax[1].plot(
            [time_series[i], time_series[i]],
            [sw_vp.iloc[i], hist_sum.iloc[j]],
            color="white",
            alpha=0.5,
            lw=0.5,
        )
    for i, j in path_flux[::10]:
        ax[2].plot(
            [time_series[i], time_series[i]],
            [sw_flux.iloc[i], hist_sum.iloc[j]],
            color="white",
            alpha=0.5,
            lw=0.5,
        )

    # Set the y-axis limit for each subplot
    ax[0].set_ylim(2, 6)
    ax[1].set_ylim(300, 380)
    ax[2].set_ylim(400, 1500)
    if x_limit is not None:
        x_limit = pd.to_datetime(x_limit, utc=True)
        ax[3].set_xlim(x_limit)
    elif time_series is not None and len(time_series) > 0:
        ax[3].set_xlim([time_series[0], time_series[-1]])

    folder_path = Path(f"../figures/lexi_sw/istp/themis_{themis_sc}/")
    folder_path.mkdir(parents=True, exist_ok=True)
    file_name = f"themis_{themis_sc}_esa_dtw_warp_path_{start_time}_to_{end_time}.png"
    file_path = folder_path / file_name
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Figure saved to {file_path}")


# Set the "time_series" to index and set the timezone to utc
df.set_index("time_series", inplace=True)
df.index = pd.to_datetime(df.index, utc=True)

# Plot the THEMIS and LEXI histogram data
# plot_themis_lexi_hist_time_series_shifted(
#     time_series=df.index,
#     sw_np=df.sw_np,
#     sw_vp=df.sw_vp,
#     sw_flux=df.sw_flux,
#     hist_sum=df.hist_sum,
#     hist_sum_no_mask=df.hist_sum_no_mask,
#     themis_sc=themis_sc,
#     freq=freq,
#     integration_time=f"{int(delta_time.total_seconds())}sec",
#     start_time=start_time,
#     end_time=end_time,
#     x_limit=["2025-03-16T19:30:00Z", "2025-03-16T20:50:00Z"],
# )

# plot_themis_lexi_hist_time_series_dtw(
#     time_series=df.index,
#     sw_np=df["sw_np"],
#     sw_vp=df["sw_vp"],
#     sw_flux=df["sw_flux"],
#     hist_sum=df["hist_sum"],
#     hist_sum_no_mask=df["hist_sum_no_mask"],
#     themis_sc="c",
#     freq="5seconds",
#     integration_time=f"{int(delta_time.total_seconds())}sec",
#     x_limit=["2025-03-16T19:30:00Z", "2025-03-16T20:50:00Z"],
#     start_time=start_time,
#     end_time=end_time,
# )

# plot_themis_lexi_hist_time_series_dtw_with_path(
#     time_series=df.index,
#     sw_np=df["sw_np"],
#     sw_vp=df["sw_vp"],
#     sw_flux=df["sw_flux"],
#     hist_sum=df["hist_sum"],
#     hist_sum_no_mask=df["hist_sum_no_mask"],
#     themis_sc="c",
#     freq="5seconds",
#     integration_time=f"{int(delta_time.total_seconds())}sec",
#     x_limit=["2025-03-16T19:30:00Z", "2025-03-16T20:50:00Z"],
#     start_time=start_time,
#     end_time=end_time,
#     # warp_stride=10,  # show every 10th DTW path
# )

plot_themis_lexi_dtw(
    time_series=df.index,
    sw_np=df["sw_np"],
    sw_vp=df["sw_vp"],
    sw_flux=df["sw_flux"],
    hist_sum=df["hist_sum"],
    hist_sum_no_mask=df["hist_sum_no_mask"],
    themis_sc="c",
    freq="5seconds",
    integration_time=f"{int(delta_time.total_seconds())}sec",
    x_limit=["2025-03-16T19:30:00Z", "2025-03-16T20:50:00Z"],
    start_time=start_time,
    end_time=end_time,
)
