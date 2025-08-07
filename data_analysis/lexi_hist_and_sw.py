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
from pandas.tseries.frequencies import to_offset
from scipy import stats
from scipy.ndimage import map_coordinates

importlib.reload(lexi_functions)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Suppress dvision by zero warnings in numpy
np.seterr(divide="ignore", invalid="ignore")


def get_lexi_and_sw_data(
    start_time="2025-03-16T19:45:00Z",
    end_time="2025-03-16T19:50:00Z",
    x_key="photon_az",
    y_key="photon_el",
    bins=200,
    bin_range=[263, 281, 15, 33],
    time_normalization=True,
    normalize_against_ground=True,
    rotate_data=False,
    rotation_angle=0,
    themis_sc="c",
):
    """
    Function to get the histogram data for the given time range and keys.
    """
    input_dict = {
        "x_key": x_key,
        "y_key": y_key,
        "start_time": start_time,
        "end_time": end_time,
        "bins": bins,
        "bin_range": bin_range,
        "time_normalization": time_normalization,
        "rotate_data": rotate_data,
        "rotation_angle": rotation_angle,
        "version_number": "v0.1",
    }

    org_hist, xedges, yedges, _, _ = lexi_functions.get_single_histogram_array_l1c_files(
        **input_dict
    )
    # print(
    #     f"The maximum value of the histogram is {np.nanmax(org_hist)}\n The minimum value is {np.nanmin(org_hist)}"
    # )

    n_shift_bin_x = 0
    n_shift_bin_y = 0
    # alpha_list = [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25]
    alpha_list = [1.0]

    if normalize_against_ground:
        alpha_list = alpha_list
    else:
        alpha_list = [1.0]
    # print("Histogram data loaded successfully.")
    for alpha in alpha_list:
        if normalize_against_ground:
            ground_file_name = "../data/ground_histogram_data_20250316_163000Z_20250316_213000Z_rot_angle_0_xkey_photon_az_ykey_photon_el_l1c.pkl"
            with open(ground_file_name, "rb") as f:
                ground_data = pickle.load(f)
            ground_hist = ground_data["hist"]
            # Replace 0 values in ground_hist with nan
            ground_hist = np.where(ground_hist == 0, np.nan, ground_hist)

            # Shift ground_hist n_shift_bin bins to the left
            ground_hist = np.(ground_hist, shift=n_shift_bin_x, axis=0)  # y-direction
            ground_hist = np.roll(ground_hist, shift=n_shift_bin_y, axis=1)  # x-direction

            if n_shift_bin_y > 0:
                ground_hist[:n_shift_bin_y, :] = np.nan
            elif n_shift_bin_y < 0:
                ground_hist[n_shift_bin_y:, :] = (
                    np.nan
                )  # This works since negative indices wrap correctly

            # Invalidate wrapped-around columns (x-direction)
            if n_shift_bin_x > 0:
                ground_hist[:, :n_shift_bin_x] = np.nan
            elif n_shift_bin_x < 0:
                ground_hist[:, n_shift_bin_x:] = np.nan

            ny, nx = ground_hist.shape
            y_orig = np.linspace(0, ny - 1, ny)
            x_orig = np.linspace(0, nx - 1, nx)

            # Create coordinate arrays for the scaled grid Center of scaling is assumed to be the center
            # of the image
            y_center, x_center = ny // 2, nx // 2
            y_scaled = (y_orig - y_center) / alpha + y_center
            x_scaled = (x_orig - x_center) / alpha + x_center

            # Create meshgrid for interpolation
            X, Y = np.meshgrid(x_scaled, y_scaled)
            coordinates = np.array([Y.ravel(), X.ravel()])

            # Interpolate the ground histogram
            ground_hist_scaled = map_coordinates(
                ground_hist, coordinates, order=1, mode="constant", cval=np.nan
            )
            ground_hist_scaled = ground_hist_scaled.reshape(ground_hist.shape)

            # Normalize the histogram data against the scaled ground data
            hist = org_hist / ground_hist_scaled

            # Replace inf values with nan
            hist = np.where(np.isinf(hist), np.nan, hist)
        else:
            hist = org_hist
            # Replace inf values with nan
            hist = np.where(np.isinf(hist), np.nan, hist)

        # print("Histogram data normalized against ground data.")

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
    time_series=None,
    sw_np=None,
    sw_vp=None,
    sw_flux=None,
    hist_sum=None,
    hist_sum_no_mask=None,
    themis_sc="c",
    freq="5seconds",
    x_limit=None,
):
    """
    Function to plot the THEMIS and LEXI histogram data.
    """

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
    ax[3].plot(time_series, hist_sum, label="Histogram Sum", color="red")
    twin_ax3 = ax[3].twinx()
    twin_ax3.plot(time_series, hist_sum_no_mask, label="Histogram Sum (No Mask)", color="orange")
    ax[3].set_ylabel("Histogram Sum", fontsize=14)
    ax[3].legend(loc="upper right", fontsize=10)
    ax[3].grid(True, linestyle="--", alpha=0.5)
    # ax[3].set_title(f"LEXI Histogram Sum", fontsize=16)
    ax[3].set_xlabel("Time [UTC]", fontsize=14)

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
        ax[3].text(
            0.05,
            0.95,
            f"Pearson Correlation: {correlation:.2f}\nSpearman Correlation: {spearman_correlation:.2f}",
            transform=ax[3].transAxes,
            fontsize=12,
            verticalalignment="top",
        )
    # Get the correlation coefficient for hist_sum_no_mask
    if sw_flux is not None and hist_sum_no_mask is not None:
        # select the data only between the start and end time
        start_time_timestamp = pd.to_datetime(start_time, utc=True)
        end_time_timestamp = pd.to_datetime(end_time, utc=True)
        # sw_flux = sw_flux[
        #     (time_series >= start_time_timestamp) & (time_series <= end_time_timestamp)
        # ]
        hist_sum_no_mask = hist_sum_no_mask[
            (time_series[:-1] >= start_time_timestamp) & (time_series[:-1] <= end_time_timestamp)
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
            0.85,
            f"Pearson Correlation (No Mask): {correlation_no_mask:.2f}\nSpearman Correlation (No Mask): {spearman_correlation_no_mask:.2f}",
            transform=ax[3].transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        ax[3].text(
            0.05,
            0.05,
            f"Pearson Correlation (No Mask): {correlation_no_mask:.2f}\nSpearman Correlation (No Mask): {spearman_correlation_no_mask:.2f}",
            transform=ax[3].transAxes,
            fontsize=12,
            verticalalignment="top",
        )
    # Set the x-axis limits if provided
    if x_limit is not None:
        # Convert x_limit to datetime if it is not already
        x_limit = pd.to_datetime(x_limit, utc=True)
        ax[3].set_xlim(x_limit)
    elif time_series is not None and len(time_series) > 0:
        # Set the x-axis limits to the start and end time of the time series
        ax[3].set_xlim([time_series[0], time_series[-1]])
        print(time_series[0], time_series[-1])
    # ax[3].set_ylim(10000, 22000)
    folder_path = Path(f"../figures/lexi_sw/themis_{themis_sc}/")
    folder_path.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"themis_{themis_sc}_esa_parameters_{start_time}_to_{end_time}_histogram_sum_{freq}.png"
    )
    file_path = folder_path / file_name
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    # close the figure
    plt.close(fig)
    print(f"Figure saved to {file_path}")
    return None


start_time = "2025-03-16T19:30:00Z"
end_time = "2025-03-16T21:15:00Z"

# Get the THEMIS and LEXI histogram data in 5 minute intervals
freq = "5min"
themis_sc = "c"

recompute_data = False
if recompute_data:

    time_series = pd.date_range(start=start_time, end=end_time, freq=freq)
    hist_sum_list = []
    hist_sum_no_mask_list = []
    sw_np_list = []
    sw_vp_list = []
    sw_flux_list = []
    time_series_list = []

    # Print the current time in utc
    print(
        f"Code execution time in UTC: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    for i, time_val in enumerate(time_series):
        # Print the progress bar as percentage
        progress = (i + 1) / len(time_series) * 100
        print(f"Progress ==> {progress:.6f}%", end="\r")
        start = time_val.isoformat()
        end = (time_val + to_offset(freq)).isoformat()
        try:
            hist_sum, themis_df_mean, themis_df_median, themis_df_median_time, hist_sum_no_mask = (
                get_lexi_and_sw_data(
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
    time_series = np.array(time_series_list)

    # Save these values to a CSV file
    df = pd.DataFrame(
        {
            "time_series": time_series,
            "sw_np": sw_np,
            "sw_vp": sw_vp,
            "sw_flux": sw_flux,
            "hist_sum": hist_sum,
            "hist_sum_no_mask": hist_sum_no_mask,
        }
    )
    folder_path = Path("../data/lexi_sw/")
    folder_path.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"themis_{themis_sc}_esa_parameters_{start_time}_to_{end_time}_histogram_sum_{freq}.csv"
    )
    file_path = folder_path / file_name
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

else:
    folder_path = Path("../data/lexi_sw/")
    file_name = (
        f"themis_{themis_sc}_esa_parameters_{start_time}_to_{end_time}_histogram_sum_{freq}.csv"
    )
    file_path = folder_path / file_name
    # Load the data from the CSV file
    df = pd.read_csv(file_path)
    time_series = pd.to_datetime(df["time_series"], utc=True)
    sw_np = df["sw_np"].values
    sw_vp = df["sw_vp"].values
    sw_flux = df["sw_flux"].values
    hist_sum = df["hist_sum"].values
    hist_sum_no_mask = df["hist_sum_no_mask"].values


# Plot the THEMIS and LEXI histogram data
plot_themis_lexi_hist_time_series(
    time_series=time_series,
    sw_np=sw_np,
    sw_vp=sw_vp,
    sw_flux=sw_flux,
    hist_sum=hist_sum,
    hist_sum_no_mask=hist_sum_no_mask,
    themis_sc=themis_sc,
    freq=freq,
    x_limit=["2025-03-16T19:30:00Z", "2025-03-16T20:50:00Z"],
)
