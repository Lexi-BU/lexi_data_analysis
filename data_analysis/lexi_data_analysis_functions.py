import datetime
import glob
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
from matplotlib.ticker import FormatStrFormatter
from spacepy.pycdf import CDF as cdf

# Suppress user warnings from matplotlib
warnings.simplefilter("ignore", UserWarning)


def get_file_list(data_folder_location, start_time, end_time):
    """Get a list of CDF files within the specified time range."""

    start_time = parser.parse(start_time) if isinstance(start_time, str) else start_time
    end_time = parser.parse(end_time) if isinstance(end_time, str) else end_time
    # Construct the folder path
    folder_name = data_folder_location  # + start_time.strftime("%Y-%m-%d")

    # Get all .cdf files recursively
    file_list = sorted(glob.glob(folder_name + "/**/*.cdf", recursive=True))

    # Regex pattern to extract timestamps from filenames
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})"
    )

    filtered_files = []
    for file in file_list:
        match = pattern.search(file)
        if match:
            start_dt_str = match.group(1) + "T" + match.group(2).replace("-", ":") + "Z"
            end_dt_str = match.group(3) + "T" + match.group(4).replace("-", ":") + "Z"

            file_start_time = parser.parse(start_dt_str)
            file_end_time = parser.parse(end_dt_str)

            # Check if the file is within the time range
            if file_start_time <= end_time and file_end_time >= start_time:
                filtered_files.append(file)

    return filtered_files  # , file_list


def read_all_data_files(
    file_list=None, start_time=None, end_time=None, return_data_type="dataframe", **kwargs
):
    """Read data from CDF files and return as a DataFrame or dictionary."""
    if "kwargs" in kwargs:
        input_data = kwargs["kwargs"]
    if file_list is None:
        file_list = get_file_list(**input_data)
    if start_time is None:
        start_time = input_data["start_time"]
    if end_time is None:
        end_time = input_data["end_time"]

    all_data_dict = {}

    for i, file in enumerate(file_list):
        try:
            print(f"Reading file number {i + 1} of {len(file_list)}", end="\r")
            dat = cdf(file)
            # Get the list of variables in the file
            variables = dat.keys()
            # add the variables to the dictionary
            for var in variables:
                if var not in all_data_dict:
                    all_data_dict[var] = []
                all_data_dict[var].append(dat[var][:])
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

    for key in all_data_dict.keys():
        if isinstance(all_data_dict[key], list):
            all_data_dict[key] = np.concatenate(all_data_dict[key])

    if return_data_type == "dataframe":
        # Convert the dictionary to a pandas dataframe
        df = pd.DataFrame(all_data_dict)
        # If data is empty, return None
        if df.empty:
            return None
        # Set the time zone of Epoch to UTC
        df["Epoch"] = pd.to_datetime(df["Epoch"], unit="s", utc=True)
        # Convert the index to datetime
        try:
            df["Epoch"] = df["Epoch"].dt.tz_convert("UTC")
        except Exception:
            df["Epoch"] = df["Epoch"].dt.tz_localize("UTC")
        # Set the index to the Epoch column
        df.set_index("Epoch", inplace=True)
        # Sort the data by index
        df.sort_index(inplace=True)
        # Check for duplicate indices, keep the first one
        df = df[~df.index.duplicated(keep="first")]
        # Select only rows that are within the time range
        df = df.loc[start_time:end_time]
        print("\n")
        # Convert the index to datetime
        return df
    elif return_data_type == "dict":
        return all_data_dict
    else:
        return None


def get_histogram_arrays(
    df_left=None,
    df_right=None,
    x_key=None,
    y_key=None,
    start_time_left=None,
    end_time_left=None,
    start_time_right=None,
    end_time_right=None,
    delta_time_left=None,
    delta_time_right=None,
    bins=None,
    bin_range=None,
    time_normalization=True,
    mincnt=1,
    save_arrays=False,
    save_folder="../data/histogram_data/",
    save_name=None,
    plot_histogram=False,
):
    """Get histogram arrays from two dataframes."""

    if df_left is None or df_right is None:
        if df_left is None:
            if start_time_left is None or end_time_left is None:
                raise ValueError("Left dataframe is None and time range is not specified.")
            df_left = read_all_data_files(
                file_list=None,
                start_time=start_time_left,
                end_time=end_time_left,
                return_data_type="dataframe",
                kwargs={
                    "data_folder_location": "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/",
                    "start_time": start_time_left,
                    "end_time": end_time_left,
                },
            )
        if df_right is None:
            if start_time_right is None or end_time_right is None:
                raise ValueError("Right dataframe is None and time range is not specified.")
            df_right = read_all_data_files(
                file_list=None,
                start_time=start_time_right,
                end_time=end_time_right,
                return_data_type="dataframe",
                kwargs={
                    "data_folder_location": "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/",
                    "start_time": start_time_right,
                    "end_time": end_time_right,
                },
            )

    # Parse all the datetimes to ensure they are in the correct format
    start_time_right = (
        parser.parse(start_time_right) if isinstance(start_time_right, str) else start_time_right
    )
    end_time_right = (
        parser.parse(end_time_right) if isinstance(end_time_right, str) else end_time_right
    )
    start_time_left = (
        parser.parse(start_time_left) if isinstance(start_time_left, str) else start_time_left
    )
    end_time_left = parser.parse(end_time_left) if isinstance(end_time_left, str) else end_time_left
    # If time_normalization is True, normalize the histograms by the delta_time
    if delta_time_left is None:
        delta_time_left = (end_time_left - start_time_left).total_seconds()

    x_data_left = df_left.loc[start_time_left:end_time_left, x_key].values
    y_data_left = df_left.loc[start_time_left:end_time_left, y_key].values
    # Get the histogram arrays
    hist_left, xedges_left, yedges_left = np.histogram2d(
        x_data_left,
        y_data_left,
        bins=bins,
        range=[bin_range[0:2], bin_range[2:4]],
    )
    hist_left /= delta_time_left

    if delta_time_right is None:
        delta_time_right = (end_time_right - start_time_right).total_seconds()

    # Add delta_time_right to the start_time_right
    current_start_time_right = start_time_right
    current_end_time_right = current_start_time_right + datetime.timedelta(seconds=delta_time_right)
    while current_end_time_right < end_time_right:
        # select the data for the time between start_time_right and current_time_right
        x_data_right = df_right.loc[current_start_time_right:current_end_time_right, x_key].values
        y_data_right = df_right.loc[current_start_time_right:current_end_time_right, y_key].values

        hist_right, xedges_right, yedges_right = np.histogram2d(
            x_data_right,
            y_data_right,
            bins=bins,
            range=[bin_range[0:2], bin_range[2:4]],
        )

        hist_right /= delta_time_right

        # Get the difference between the two histograms
        hist_center = hist_left - hist_right

        # If save_arrays is True, save the arrays to a pickle file
        if save_arrays:
            if save_name is None:
                save_name = f"histogram_{x_key}_{y_key}_{start_time_left.strftime('%Y-%m-%d_%H-%M-%s')}_to_{end_time_left.strftime('%Y-%m-%d_%H-%M-%s')}_{current_start_time_right.strftime('%Y-%m-%d_%H-%M-%s')}_to_{current_end_time_right.strftime('%Y-%m-%d_%H-%M-%s')}_{delta_time_left}_{delta_time_right}_normalized_{time_normalization}.pkl"
            save_path = Path(save_folder)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / save_name, "wb") as f:
                np.savez(f, hist_left=hist_left, hist_right=hist_right, hist_center=hist_center)

        # If plot_histogram is True, plot the histograms
        if plot_histogram:
            plot_histograms(
                hist_left=hist_left,
                hist_right=hist_right,
                hist_center=hist_center,
                xedges_left=xedges_left,
                yedges_left=yedges_left,
                xedges_right=xedges_right,
                yedges_right=yedges_right,
                x_key=x_key,
                y_key=y_key,
                start_time_left=start_time_left,
                end_time_left=end_time_left,
                start_time_right=current_start_time_right,
                end_time_right=current_end_time_right,
                delta_time_left=delta_time_left,
                delta_time_right=delta_time_right,
                mincnt=mincnt,
                # save_folder=save_folder,
                save_name=save_name,
                # show_plot=True,
                save_plot=True,
                # save_plot_name=save_name,
                save_plot_format="png",
                color_map="inferno",
                color_map_center="coolwarm",
                # color_map_center_vmin=None,
                # color_map_center_vmax=None,
                fixed_colorbar_limits=False,
            )
            print(
                f"Histogram plotted for time range: {current_start_time_right} to {current_end_time_right}",
                end="\r",
            )
            current_start_time_right = current_end_time_right
            current_end_time_right = current_start_time_right + datetime.timedelta(
                seconds=delta_time_right
            )
        # Repeat until the end_time_right is reached
    # return (
    #     hist_left,
    #     hist_right,
    #     hist_center,
    #     xedges_left,
    #     yedges_left,
    #     xedges_right,
    #     yedges_right,
    #     start_time_left,
    #     end_time_left,
    #     start_time_right,
    #     end_time_right,
    # )


def get_single_histogram_array(
    df=None,
    x_key=None,
    y_key=None,
    start_time=None,
    end_time=None,
    bins=None,
    bin_range=None,
    time_normalization=True,
    mincnt=1,
):
    """Get a single histogram array from a dataframe."""

    # Parse the start and end times
    start_time = parser.parse(start_time) if isinstance(start_time, str) else start_time
    end_time = parser.parse(end_time) if isinstance(end_time, str) else end_time

    if df is None:
        df = read_all_data_files(
            file_list=None,
            start_time=start_time,
            end_time=end_time,
            return_data_type="dataframe",
            kwargs={
                "data_folder_location": "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/",
                "start_time": start_time,
                "end_time": end_time,
            },
        )
    if df is None or df.empty:
        raise ValueError("No data found in the specified time range.")

    lower_threshold = 2
    upper_threshold = 3.3
    # Select only rows where for all Channel1, Channel2, Channel3, Channel4 are between the
    # thresholds
    df = df[
        (df["Channel1"] >= lower_threshold)
        & (df["Channel1"] <= upper_threshold)
        & (df["Channel2"] >= lower_threshold)
        & (df["Channel2"] <= upper_threshold)
        & (df["Channel3"] >= lower_threshold)
        & (df["Channel3"] <= upper_threshold)
        & (df["Channel4"] >= lower_threshold)
        & (df["Channel4"] <= upper_threshold)
    ]
    # Select the data for the specified time range
    x_data = df.loc[start_time:end_time, x_key].values
    y_data = df.loc[start_time:end_time, y_key].values

    # Get the histogram array
    hist, xedges, yedges = np.histogram2d(
        x_data,
        y_data,
        bins=bins,
        range=[bin_range[0:2], bin_range[2:4]],
    )

    if time_normalization:
        delta_time = (end_time - start_time).total_seconds()
        hist /= delta_time

    # Get the pointing location
    pointing_file_name = (
        "../data/merged_lexi_hk_look_direction_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00.pkl"
    )
    df_pointing = pd.read_pickle(pointing_file_name)
    df_pointing = df_pointing.loc[start_time:end_time]
    ra_median = df_pointing["ra_lexi"].median()
    dec_median = df_pointing["dec_lexi"].median()

    return hist, xedges, yedges, ra_median, dec_median


def plot_histograms(
    hist_left=None,
    hist_right=None,
    hist_center=None,
    xedges_left=None,
    yedges_left=None,
    xedges_right=None,
    yedges_right=None,
    xedges_center=None,
    yedges_center=None,
    xedges=None,
    yedges=None,
    x_key=None,
    y_key=None,
    start_time_left=None,
    end_time_left=None,
    start_time_right=None,
    end_time_right=None,
    delta_time_left=None,
    delta_time_right=None,
    mincnt=1,
    save_folder="../figures/histogram_diff/",
    save_name=None,
    show_plot=True,
    save_plot=False,
    save_plot_name=None,
    save_plot_format="png",
    color_map="plasma",
    color_map_center="coolwarm",
    color_map_center_vmin=None,
    color_map_center_vmax=None,
    fixed_colorbar_limits=False,
):
    """
    Plot the histograms of two dataframes and save the plot.

    Parameters
    ----------
    hist_left : 2D array
        Histogram of the left dataframe.
    hist_right : 2D array
        Histogram of the right dataframe. This is the reference histogram.
    hist_center : 2D array
        Histogram of the difference between the left and right dataframes.
    xedges_left : 1D array
        xedges of the left histogram.
    yedges_left : 1D array
        yedges of the left histogram.
    xedges_right : 1D array
        xedges of the right histogram.
    yedges_right : 1D array
        yedges of the right histogram.
    x_key : str
        Key for the x-axis data.
    y_key : str
        Key for the y-axis data.
    start_time_left : str
        Start time of the left histogram.
    end_time_left : str
        End time of the left histogram.
    start_time_right : str
        Start time of the right histogram.
    end_time_right : str
        End time of the right histogram.
    delta_time_left : float
        Delta time of the left histogram.
    delta_time_right : float
        Delta time of the right histogram.
    mincnt : int
        Minimum count for the histogram.
    save_folder : str
        Folder to save the histogram data.
    save_name : str
        Name of the histogram data file.
    show_plot : bool
        Whether to show the plot.
    save_plot : bool
        Whether to save the plot.
    save_plot_name : str
        Name of the plot file.
    save_plot_format : str
        Format of the plot file.
    color_map : str
        Color map for the histogram (left and right). Default is "viridis".
    color_map_center : str
        Color map for the histogram (center). Default is "RdBu".
    color_map_center_vmin : float
        Minimum value for the color map (center).
    color_map_center_vmax : float
        Maximum value for the color map (center).

    Returns
    -------
    None
    """
    # Parse the start and end times
    start_time_right = (
        parser.parse(start_time_right) if isinstance(start_time_right, str) else start_time_right
    )
    end_time_right = (
        parser.parse(end_time_right) if isinstance(end_time_right, str) else end_time_right
    )
    start_time_left = (
        parser.parse(start_time_left) if isinstance(start_time_left, str) else start_time_left
    )
    end_time_left = parser.parse(end_time_left) if isinstance(end_time_left, str) else end_time_left

    if xedges_left is None or yedges_left is None:
        if xedges is None or yedges is None:
            raise ValueError(
                "xedges and yedges must be provided if xedges_left or yedges_left are None."
            )
        else:
            xedges_left = xedges
            yedges_left = yedges
    if xedges_right is None or yedges_right is None:
        if xedges is None or yedges is None:
            raise ValueError(
                "xedges and yedges must be provided if xedges_right or yedges_right are None."
            )
        else:
            xedges_right = xedges
            yedges_right = yedges
    if xedges_center is None or yedges_center is None:
        if xedges is None or yedges is None:
            raise ValueError(
                "xedges and yedges must be provided if xedges_center or yedges_center are None."
            )
        else:
            xedges_center = xedges
            yedges_center = yedges

    # Set the font size
    plt.rcParams.update({"font.size": 16})
    # Use the black background style
    plt.style.use("dark_background")

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 18), sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.15, wspace=0.05)

    # Plot the left histogram
    im_left = axs[0][0].imshow(
        hist_left.T,
        origin="lower",
        extent=[xedges_left[0], xedges_left[-1], yedges_left[0], yedges_left[-1]],
        aspect="equal",
        interpolation="nearest",
        cmap=color_map,
        # vmin=mincnt,
    )

    axs[0, 0].set_title(
        f"{start_time_left.strftime('%H:%M:%S')} to {end_time_left.strftime('%H:%M:%S')}"
    )
    axs[0, 0].set_xlabel(x_key)
    axs[0, 0].set_ylabel(y_key)
    # Set the aspect ratio to be equal
    axs[0, 0].set_aspect("equal", adjustable="box")
    # Add the colorbar
    cbar_left = fig.colorbar(
        im_left,
        ax=axs[0][0],
        orientation="horizontal",
        pad=0.1,
        aspect=70,
        fraction=0.02,
        location="top",
        shrink=0.8,
    )
    if fixed_colorbar_limits:
        # Set the color bar limits to be the same for both histograms
        cbar_left.set_clim(vmin=mincnt, vmax=np.max(hist_left))
    cbar_left.set_label("cts/s")
    cbar_left.ax.tick_params(labelsize=10)
    cbar_left.ax.set_xticklabels(cbar_left.get_ticks(), fontsize=10)
    cbar_left.ax.set_xticks(cbar_left.get_ticks())
    cbar_left.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))

    # Plot the right histogram
    im_right = axs[0][1].imshow(
        hist_right.T,
        origin="lower",
        extent=[xedges_right[0], xedges_right[-1], yedges_right[0], yedges_right[-1]],
        aspect="equal",
        interpolation="nearest",
        cmap=color_map,
        # vmin=mincnt,
    )
    axs[0, 1].set_title(
        f"{start_time_right.strftime('%H:%M:%S')} to {end_time_right.strftime('%H:%M:%S')}"
    )
    axs[0, 1].set_xlabel(x_key)
    axs[0, 1].set_ylabel(y_key)
    # Set the aspect ratio to be equal
    axs[0, 1].set_aspect("equal", adjustable="box")
    # Add the colorbar
    cbar_right = fig.colorbar(
        im_right,
        ax=axs[0][1],
        orientation="horizontal",
        pad=0.1,
        aspect=70,
        fraction=0.02,
        location="top",
        shrink=0.8,
    )
    cbar_right.set_label("cts/s")
    cbar_right.ax.tick_params(labelsize=10)
    cbar_right.ax.set_xticklabels(cbar_right.get_ticks(), fontsize=10)
    cbar_right.ax.set_xticks(cbar_right.get_ticks())
    cbar_right.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    # Set the maximum nuber of sig figures for colorbar ticks to 2 in normal notation

    # Plot the center histogram
    if color_map_center_vmin is None:
        color_map_center_vmin = -np.max(np.abs(hist_center))
    if color_map_center_vmax is None:
        color_map_center_vmax = np.max(np.abs(hist_center))
    im_center = axs[1][0].imshow(
        hist_center.T,
        origin="lower",
        extent=[xedges_center[0], xedges_center[-1], yedges_center[0], yedges_center[-1]],
        aspect="equal",
        interpolation="nearest",
        cmap=color_map_center,
        norm=mpl.colors.LogNorm(),
        # vmin=color_map_center_vmin,
        # vmax=color_map_center_vmax,
    )
    axs[1, 0].set_title(
        "Scaled/Difference Histogram"
        # f"{start_time_left.strftime('%H:%M:%S')} to {end_time_left.strftime('%H:%M:%S')} - {start_time_right.strftime('%H:%M:%S')} to {end_time_right.strftime('%H:%M:%S')}"
    )
    axs[1, 0].set_xlabel(x_key)
    axs[1, 0].set_ylabel(y_key)
    # Set the aspect ratio to be equal
    axs[1, 0].set_aspect("equal", adjustable="box")
    # Add the colorbar
    cbar_center = fig.colorbar(
        im_center,
        ax=axs[1][0],
        orientation="horizontal",
        pad=0.1,
        aspect=70,
        fraction=0.02,
        location="top",
        shrink=0.8,
    )
    cbar_center.set_label("Cts/s")
    cbar_center.ax.tick_params(labelsize=10)
    cbar_center.ax.set_xticklabels(cbar_center.get_ticks(), fontsize=10, rotation=45)
    cbar_center.ax.set_xticks(cbar_center.get_ticks())
    cbar_center.ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    # Rotate the colorbar ticks for better readability
    # cbar_center.ax.tick_params(axis="x", rotation=45)
    pointing_file_name = (
        "../data/merged_lexi_hk_look_direction_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00.pkl"
    )
    df_pointing = pd.read_pickle(pointing_file_name)
    df_pointing_right = df_pointing.loc[
        start_time_right:end_time_right
    ]  # For the right histogram time range
    ra_min = df_pointing_right["ra_lexi"].min()
    ra_max = df_pointing_right["ra_lexi"].max()
    dec_min = df_pointing_right["dec_lexi"].min()
    dec_max = df_pointing_right["dec_lexi"].max()
    # Make the RA and Dec plot for the right histogram time range
    axs[1, 1].scatter(
        df_pointing_right.index,
        df_pointing_right["dec_lexi"],
        color="c",
        s=1,
        alpha=1,
    )
    twin_ax = axs[1, 1].twinx()  # Create a twin axis for RA
    twin_ax.scatter(
        df_pointing_right.index,
        df_pointing_right["ra_lexi"],
        color="m",
        s=1,
        alpha=1,
    )
    axs[1, 1].set_title(
        f"Look Direction \n {start_time_right.strftime('%H:%M:%S')} to {end_time_right.strftime('%H:%M:%S')}"
    )
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Dec", color="c")
    twin_ax.set_ylabel("RA", color="m")

    # Set the grid for the pointing plot
    axs[1, 1].grid(color="c", linestyle="--", linewidth=0.5, alpha=0.5)

    # Set the spine color to match the line color
    twin_ax.spines["left"].set_color("c")  # Dec
    twin_ax.spines["right"].set_color("m")  # RA
    axs[1, 1].tick_params(axis="y", colors="c")  # Dec
    twin_ax.tick_params(axis="y", colors="m")  # RA

    # Set the y-axis limits for the pointing plot (if the difference between the min and max is less
    # than 1 degree then set it to 5 degrees)
    # dec_range = dec_max - dec_min
    # ra_range = ra_max - ra_min
    # if dec_range < 1:
    #     axs[1, 1].set_ylim(dec_min - 2.5, dec_max + 2.5)
    # else:
    #     axs[1, 1].set_ylim(dec_min, dec_max)
    # if ra_range < 1:
    #     twin_ax.set_ylim(ra_min - 2.5, ra_max + 2.5)
    # else:
    #     twin_ax.set_ylim(ra_min, ra_max)

    # axs[1][1].axis("off")
    # Set the title
    fig.suptitle(
        f"Histogram of {x_key} and {y_key} \n from {start_time_left.strftime('%Y-%m-%d %H:%M:%S')} to {end_time_left.strftime('%Y-%m-%d %H:%M:%S')} and {start_time_right.strftime('%Y-%m-%d %H:%M:%S')} to {end_time_right.strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=14,
    )

    # Manage the ticks
    for ax in axs.flat:
        ax.label_outer()
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            labelsize=12,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            labelsize=12,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=5)

    axs[0, 1].tick_params(
        axis="y",
        which="both",
        labelleft=False,
        labelright=True,
    )
    axs[0, 1].yaxis.set_label_position("right")

    axs[1, 0].set_ylabel("")
    axs[0, 0].grid(True, color="c", alpha=0.2)
    axs[1, 0].grid(True, color="k", alpha=0.2)
    axs[0, 1].grid(True, color="c", alpha=0.2)
    axs[1, 1].grid(True, color="c", alpha=0.2)

    # Rotate the x-axis labels for the bottom plots to be more readable
    axs[1, 1].set_xticklabels(
        [label if i % 2 == 0 else "" for i, label in enumerate(axs[1, 1].get_xticklabels())],
        rotation=45,
        ha="right",
    )  # Rotate every other label to avoid overlap
    axs[1, 1].tick_params(axis="x", which="major", labelsize=10)
    axs[1, 1].tick_params(axis="x", which="minor", labelsize=10)
    axs[1, 1].tick_params(axis="y", which="major", labelsize=10, labelleft=True)

    # Save the plot
    if save_plot:
        if save_plot_name is None:
            save_plot_name = f"histogram_{x_key}_{y_key}_{start_time_left.strftime('%Y-%m-%d_%H-%M-%s')}_to_{end_time_left.strftime('%Y-%m-%d_%H-%M-%s')}_{start_time_right.strftime('%Y-%m-%d_%H-%M-%s')}_to_{end_time_right.strftime('%Y-%m-%d_%H-%M-%s')}_{delta_time_left}_{delta_time_right}.{save_plot_format}"
        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_path / save_plot_name,
            format=save_plot_format,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()
