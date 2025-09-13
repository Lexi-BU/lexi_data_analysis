import datetime
import glob
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
from matplotlib.ticker import ScalarFormatter
from spacepy.pycdf import CDF as cdf

# Suppress user warnings from matplotlib
warnings.simplefilter("ignore", UserWarning)


def get_file_list(data_folder_location, start_time, end_time, version="latest"):
    """Get a list of CDF files within the specified time range and version preference.

    Parameters
    ----------
    data_folder_location : str
        The path to the folder containing the CDF files.
    start_time : datetime
        The start time for the time range.
    end_time : datetime
        The end time for the time range.
    version : str or tuple, optional
        The version preference for the files. Can be "latest", a version string (e.g. "v1.0"), or a tuple (major, minor).

    Returns
    -------
    list
        A list of CDF file paths that match the specified criteria.
    """

    start_time = parser.parse(start_time) if isinstance(start_time, str) else start_time
    end_time = parser.parse(end_time) if isinstance(end_time, str) else end_time

    file_list = sorted(glob.glob(str(Path(data_folder_location) / "**" / "*.cdf"), recursive=True))

    pattern = re.compile(r"lexi_l1c_(\d{10})_V(\d+)\.(\d+)\.cdf$")

    file_dict = {}

    for file in file_list:
        match = pattern.search(Path(file).name)
        if match:
            file_start_time = datetime.datetime.strptime(match.group(1), "%Y%m%d%H")
            file_end_time = file_start_time + datetime.timedelta(hours=1)

            if file_start_time <= end_time and file_end_time >= start_time:
                time_key = match.group(1)
                file_version = (int(match.group(2)), int(match.group(3)))

                if time_key not in file_dict:
                    file_dict[time_key] = []

                file_dict[time_key].append((file_version, file))

    filtered_files = []
    for time_key, files in file_dict.items():
        if version == "latest":
            selected_file = max(files, key=lambda x: x[0])[1]
        elif isinstance(version, tuple):
            matching_files = [f for v, f in files if v == version]
            if matching_files:
                selected_file = matching_files[0]
            else:
                continue
        else:
            major, minor = map(int, version.replace("v", "").split("."))
            matching_files = [f for v, f in files if v == (major, minor)]
            if matching_files:
                selected_file = matching_files[0]
            else:
                continue

        filtered_files.append(selected_file)

    print(
        f"Found {len(filtered_files)} files matching criteria: {start_time} to {end_time}, version: {version} --- {filtered_files} \n"
    )
    filtered_files.sort()
    return filtered_files


def read_all_data_files(
    file_list=None, start_time=None, end_time=None, return_data_type="dataframe", **kwargs
):
    """Read data from CDF files and return as a DataFrame or dictionary.

    Parameters
    ----------
    file_list : list, optional
        A list of CDF files to read. If not provided, files will be fetched based on the time range.

    start_time : datetime, optional
        The start time for the time range. Make sure that it is in the following format: YYYY-MM-DD
        HH:MM:SS. The timezone should be UTC.

    end_time : datetime, optional
        The end time for the time range. The timezone should be UTC.

    return_data_type : str, optional
        The type of data to return. Can be "dataframe" or "dict". Default is "dataframe".

    kwargs : dict, optional
        Additional keyword arguments to pass to the file reading function.

    Returns
    -------
    DataFrame or dict
        The data read from the CDF files, either as a pandas DataFrame or a dictionary.
    """
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
            print(f"Error reading file {file}: {e} \n")
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
        # print("\n")
        # Convert the index to datetime
        return df
    elif return_data_type == "dict":
        return all_data_dict
    else:
        return None


def plot_time_series(
    df=None,
    start_time=None,
    end_time=None,
    keys=None,
    x_axis="Epoch",
    data_folder_location=None,
    output_path="figures",
    figure_name=None,
):
    """Plot time series for selected keys from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The input dataframe containing the data to plot. If this is not provided, the function will attempt to read data from files.
    start_time : str, optional
        The start time for the time range. The timezone should be UTC.
    end_time : str, optional
        The end time for the time range. The timezone should be UTC.
    keys : list, optional
        The list of keys (columns) to plot. If None, all columns except 'Epoch' will be plotted.
    x_axis : str, optional
        The column to use for the x-axis. Can be 'Epoch' or 'Epoch_unix'. Default is 'Epoch'.
    data_folder_location : str, optional
        The folder location for the data files. Default is "data".
    output_path : str, optional
        The folder where the output figures will be saved. Default is "figures".
    figure_name : str, optional
        The name of the output figure file.

    Returns
    -------
    None
    """

    if df is None or df.empty:
        print("No data provided. Attempting to read from files... \n")
        df = read_all_data_files(
            file_list=None,
            start_time=start_time,
            end_time=end_time,
            return_data_type="dataframe",
            kwargs={
                "data_folder_location": data_folder_location,
                "version": "latest",
                "start_time": start_time,
                "end_time": end_time,
            },
        )

    # Ensure x_axis is valid
    if x_axis not in df.columns and x_axis != "Epoch":
        raise ValueError(f"x_axis must be one of 'Epoch' or a column in DataFrame, not '{x_axis}'.")

    # Filter based on time range
    if start_time:
        start_time = (
            pd.to_datetime(start_time).tz_localize("UTC")
            if pd.to_datetime(start_time).tzinfo is None
            else pd.to_datetime(start_time)
        )

        df = df[df.index >= start_time]
    if end_time:
        end_time = (
            pd.to_datetime(end_time).tz_localize("UTC")
            if pd.to_datetime(end_time).tzinfo is None
            else pd.to_datetime(end_time)
        )
        df = df[df.index <= end_time]

    # Auto-select keys if not specified
    if keys is None:
        keys = [col for col in df.columns if col not in ["Epoch", "Epoch_unix"]]

    # Create figure with subplots
    n_plots = len(keys)
    # Set the theme to dark background
    plt.style.use("dark_background")
    fig, axs = plt.subplots(n_plots, 1, figsize=(24, 3 * n_plots), sharex=True)

    if n_plots == 1:
        axs = [axs]  # Ensure axs is iterable

    # Set x-axis
    if x_axis == "Epoch":
        x = df.index
    else:  # "Epoch_unix"
        x = df["Epoch_unix"]

    # Set the base font size for all plots
    base_font_size = 12

    color_list = plt.cm.get_cmap("tab10", len(keys)).colors
    for ax, key, color in zip(axs, keys, color_list):
        print(f"Plotting {key}...")
        ax.plot(x, df[key], label=key, color=color, linewidth=0.7)
        # ax.scatter(x, df[key], label=key, s=1, color=color, alpha=0.7)
        ax.set_ylabel(key, fontsize=1.5 * base_font_size)
        # ax.legend(loc="upper right", fontsize=base_font_size)
        ax.grid(True)

    axs[-1].set_xlabel(
        f"{x_axis} [UTC] starting at {start_time:%Y-%m-%d %H:%M:%S}", fontsize=2.5 * base_font_size
    )
    fig.suptitle(
        f"Time Series Plot\n {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S} UTC",
        fontsize=2 * base_font_size,
    )
    # Set the tick label size for all axes
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=base_font_size)
        if x_axis == "Epoch":
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M:%S"))
            # Rotate x-axis tick labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            # Set the x-axis tick label font size
            for label in ax.get_xticklabels(which="major"):
                label.set_fontsize(1.6 * base_font_size)
        else:
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), fontsize=1.6 * base_font_size
            )
    # Set the x-axis limits
    if x_axis == "Epoch":
        axs[-1].set_xlim([start_time, end_time])
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # Make sure that the output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if figure_name is None:
        figure_name = f"time_series_plot_{start_time}_{end_time}.png"
    plt.savefig(Path(output_path) / figure_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Plot saved to {Path(output_path) / figure_name} \n")


def plot_histogram(
    df=None,
    dat_flat_field=None,
    H_flat=None,
    start_time=None,
    end_time=None,
    x_key=None,
    y_key=None,
    bins=200,
    bin_range=[263, 281, 15, 33],
    cmap="viridis",
    norm_scale="log",
    time_normalization=False,
    data_folder_location=None,
    output_path="figures",
    flat_field_correction=False,
    v_min_orig=None,
    v_max_orig=None,
    v_min_flat=None,
    v_max_flat=None,
    v_min_result=None,
    v_max_result=None,
    plot_flat_field=False,
    plot_result_hist=False,
    verbose=False,
):
    """Plot a 2D histogram between two keys from the dataframe, optionally normalized by flat field
    data.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The input dataframe containing the data to plot. If this is not provided, the function will attempt to read data from files.
    start_time : str, optional
        The start time for filtering the data. Must be in UTC format. The input string must be in a format recognized by pandas.
    end_time : str, optional
        The end time for filtering the data. Must be in UTC format. The input string must be in a format recognized by pandas.
    x_key : str, optional
        The key for the x-axis data.
    y_key : str, optional
        The key for the y-axis data.
    bins : int, optional
        The number of bins for the histogram.
    cmap : str, optional
        The colormap to use for the histogram.
    norm_scale : str, optional
        The normalization scale to use for the histogram.
    time_normalization : bool, optional
        Whether to normalize the histogram by time.
    data_folder_location : str, optional
        The folder location for the data files. Default is "data".
    output_path : str, optional
        The folder where the output figures will be saved. Default is "figures".
    flat_field_correction : bool, optional
        Whether to use flat field data for normalization.

    Returns
    -------
    None
    """

    # Convert time boundaries to UTC-aware timestamps
    def to_utc(ts):
        ts = pd.to_datetime(ts)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts

    if df is None or df.empty:
        if verbose:
            print("No data provided. Attempting to read from files... \n")
        df = read_all_data_files(
            file_list=None,
            start_time=start_time,
            end_time=end_time,
            return_data_type="dataframe",
            kwargs={
                "data_folder_location": data_folder_location,
                "version": "latest",
                "start_time": start_time,
                "end_time": end_time,
            },
        )
    time_tolerance = datetime.timedelta(seconds=5)
    # Check if the minimum and maximum value of index are within the specified range
    if not (
        df.index.min() >= to_utc(start_time) - time_tolerance
        and df.index.max() <= to_utc(end_time) + time_tolerance
    ):
        if verbose:
            print(
                f"Warning: Data index range {df.index.min()} to {df.index.max()} is not within the specified time range {start_time} to {end_time}. \n Attempting to read additional data files..."
            )
            df = read_all_data_files(
                file_list=None,
                start_time=start_time,
                end_time=end_time,
                return_data_type="dataframe",
                kwargs={
                    "data_folder_location": data_folder_location,
                    "version": "latest",
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )

    if x_key is None or y_key is None:
        raise ValueError("Both x_key and y_key must be specified.")

    if start_time:
        df = df[df.index >= to_utc(start_time)]
    if end_time:
        df = df[df.index <= to_utc(end_time)]

    if df.empty:
        if verbose:
            print("No data available in the given time range. Exiting function. \n")
        return

    x = df[x_key].values
    y = df[y_key].values

    # Time normalization
    if time_normalization:
        duration_sec = (df.index[-1] - df.index[0]).total_seconds()
        weights = np.ones_like(x) / duration_sec
        unit_label = "Counts/sec"
    else:
        weights = None
        unit_label = "Counts"

    # Compute original 2D histogram
    fig, ax = plt.subplots(figsize=(10, 8))
    # Temporary histogram to get values
    H_orig, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[bin_range[0], bin_range[1]], [bin_range[2], bin_range[3]]],
        weights=weights,
    )

    if norm_scale == "log":
        if v_min_orig is not None and v_max_orig is not None:
            norm = mpl.colors.LogNorm(vmin=v_min_orig, vmax=v_max_orig)
        else:
            norm = mpl.colors.LogNorm(vmin=min(1, np.nanmin(H_orig)), vmax=np.nanmax(H_orig))
    else:
        if v_min_orig is not None and v_max_orig is not None:
            norm = mpl.colors.Normalize(vmin=v_min_orig, vmax=v_max_orig)
        else:
            norm = mpl.colors.Normalize(vmin=np.nanmin(H_orig), vmax=np.nanmax(H_orig))

    H_orig, xedges_orig, yedges_orig, img = ax.hist2d(
        x,
        y,
        bins=bins,
        range=[[bin_range[0], bin_range[1]], [bin_range[2], bin_range[3]]],
        weights=weights,
        cmap=cmap,
        norm=norm,
    )

    # Prepare output folder
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if flat_field_correction:
        if dat_flat_field is None and H_flat is None:
            flat_field_path = "data/flat_field_data/lexi_l1c_flat_field_data_20240524_20240530.cdf"
            if verbose:
                print(f"Reading flat field data from {flat_field_path} \n")
            with cdf(flat_field_path) as cdf_file:
                flat_x = cdf_file[x_key][...]
                flat_y = cdf_file[y_key][...]
                flat_epoch = pd.to_datetime(cdf_file["Epoch"][...])
                flat_index = pd.DatetimeIndex(flat_epoch).tz_localize("UTC")
        elif dat_flat_field is not None and H_flat is None:
            if verbose:
                print(f"Using provided flat field data from {dat_flat_field} \n")
            flat_x = dat_flat_field[x_key][...]
            flat_y = dat_flat_field[y_key][...]
            flat_epoch = pd.to_datetime(dat_flat_field["Epoch"][...])
            flat_index = pd.DatetimeIndex(flat_epoch).tz_localize("UTC")

        # Compute flat field histogram
        if H_flat is None:
            # Time-normalized weights if needed
            if time_normalization:
                flat_duration = (flat_index[-1] - flat_index[0]).total_seconds()
                flat_weights = np.ones_like(flat_x) / flat_duration
            else:
                flat_weights = None

            H_flat, _, _ = np.histogram2d(
                flat_x,
                flat_y,
                bins=[xedges_orig, yedges_orig],
                weights=flat_weights,
            )
        if plot_flat_field:
            # Plot and save flat field histogram
            fig_ff, ax_ff = plt.subplots(figsize=(10, 8))
            if norm_scale == "log":
                norm = mpl.colors.LogNorm(vmin=1e-6, vmax=np.nanmax(H_flat))
            else:
                norm = mpl.colors.Normalize(vmin=np.nanmin(H_flat), vmax=np.nanmax(H_flat))

            mesh_ff = ax_ff.pcolormesh(xedges, yedges, H_flat.T, cmap=cmap, norm=norm)
            cbar_ff = plt.colorbar(mesh_ff, ax=ax_ff)
            cbar_ff.set_label(f"{unit_label} (flat field)")
            ax_ff.set_xlabel(f"{x_key} [cm]" if "mcp" in x_key else f"{x_key} [deg]")
            ax_ff.set_ylabel(f"{y_key} [cm]" if "mcp" in y_key else f"{y_key} [deg]")
            ax_ff.set_aspect("equal", adjustable="box")
            ax_ff.set_title(f"Flat Field Histogram of {y_key} vs {x_key}")
            plt.tight_layout()
            ff_name = f"histogram_{x_key}_vs_{y_key}_flat_field.png"
            plt.savefig(Path(output_path) / ff_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
            if verbose:
                print(f"Flat field histogram saved to {Path(output_path) / ff_name} \n")

        # Normalize by flat field histogram
        with np.errstate(divide="ignore", invalid="ignore"):
            H_result = np.divide(H_orig, H_flat)
            H_result[~np.isfinite(H_result)] = 0

        if plot_result_hist:
            # Plot normalized histogram
            fig_norm, ax_norm = plt.subplots(figsize=(10, 8))
            if norm_scale == "log":
                if v_min_result is not None and v_max_result is not None:
                    norm = mpl.colors.LogNorm(vmin=v_min_result, vmax=v_max_result)
                else:
                    norm = mpl.colors.LogNorm(vmin=1e-2, vmax=np.nanmax(H_result))
            else:
                if v_min_result is not None and v_max_result is not None:
                    norm = mpl.colors.Normalize(vmin=v_min_result, vmax=v_max_result)
                else:
                    norm = mpl.colors.Normalize(vmin=np.nanmin(H_result), vmax=np.nanmax(H_result))

            mesh_norm = ax_norm.pcolormesh(xedges, yedges, H_result.T, cmap=cmap, norm=norm)
            cbar_norm = plt.colorbar(mesh_norm, ax=ax_norm)
            cbar_norm.set_label(f"{unit_label} (normalized)")
            ax_norm.set_xlabel(f"{x_key} [cm]" if "mcp" in x_key else f"{x_key} [deg]")
            ax_norm.set_ylabel(f"{y_key} [cm]" if "mcp" in y_key else f"{y_key} [deg]")
            ax_norm.set_aspect("equal", adjustable="box")
            ax_norm.set_title(f"Flat-Field Normalized 2D Histogram of {y_key} vs {x_key}")
            plt.tight_layout()
            norm_name = f"histogram_{x_key}_vs_{y_key}_normalized.png"
            plt.savefig(Path(output_path) / norm_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
            if verbose:
                print(f"Normalized histogram saved to {Path(output_path) / norm_name} \n")
    else:
        if plot_result_hist:
            # If not flat field corrected, save original histogram
            cbar = plt.colorbar(img, ax=ax)
            cbar.set_label(unit_label)
            ax.set_xlabel(f"{x_key} [cm]" if "mcp" in x_key else f"{x_key} [deg]")
            ax.set_ylabel(f"{y_key} [cm]" if "mcp" in y_key else f"{y_key} [deg]")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"2D Histogram of {y_key} vs {x_key}")
            plt.tight_layout()
            orig_name = f"histogram_{x_key}_vs_{y_key}.png"
            plt.savefig(Path(output_path) / orig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
            if verbose:
                print(f"Histogram saved to {Path(output_path) / orig_name} \n")

    return {
        "H_orig": H_orig,
        "H_flat": H_flat if flat_field_correction else None,
        "H_result": H_result if flat_field_correction else H_orig,
        "xedges": xedges_orig if flat_field_correction else xedges,
        "yedges": yedges_orig if flat_field_correction else yedges,
    }


def centers_to_corners_2d(ra_c, dec_c):
    """
    ra_c, dec_c: (H, W) center maps in degrees.
    Returns: RAcorn, DECcorn with shape (H+1, W+1) for pcolormesh.
    Uses midpoint edges and linear extrapolation at boundaries.
    """
    ra_c = np.asarray(ra_c, float)
    dec_c = np.asarray(dec_c, float)
    H, W = ra_c.shape

    # Midpoints between adjacent centers (internal edges)
    ra_i = 0.5 * (ra_c[1:, :] + ra_c[:-1, :])  # (H-1, W)
    ra_j = 0.5 * (ra_c[:, 1:] + ra_c[:, :-1])  # (H, W-1)
    dec_i = 0.5 * (dec_c[1:, :] + dec_c[:-1, :])
    dec_j = 0.5 * (dec_c[:, 1:] + dec_c[:, :-1])

    # Extrapolate outer edges along i (rows)
    ra_top = ra_c[0, :] - (ra_i[0, :] - ra_c[0, :])
    ra_bot = ra_c[-1, :] + (ra_c[-1, :] - ra_i[-1, :])
    dec_top = dec_c[0, :] - (dec_i[0, :] - dec_c[0, :])
    dec_bot = dec_c[-1, :] + (dec_c[-1, :] - dec_i[-1, :])

    # Extrapolate outer edges along j (cols)
    ra_left = ra_c[:, 0] - (ra_j[:, 0] - ra_c[:, 0])
    ra_right = ra_c[:, -1] + (ra_c[:, -1] - ra_j[:, -1])
    dec_left = dec_c[:, 0] - (dec_j[:, 0] - dec_c[:, 0])
    dec_right = dec_c[:, -1] + (dec_c[:, -1] - dec_j[:, -1])

    # Build (H+1, W+1) corners by averaging edges appropriately
    RAcorn = np.empty((H + 1, W + 1), float)
    DECcorn = np.empty((H + 1, W + 1), float)

    # Internal corners
    RAcorn[1:H, 1:W] = 0.25 * (ra_c[:-1, :-1] + ra_c[1:, :-1] + ra_c[:-1, 1:] + ra_c[1:, 1:])
    DECcorn[1:H, 1:W] = 0.25 * (dec_c[:-1, :-1] + dec_c[1:, :-1] + dec_c[:-1, 1:] + dec_c[1:, 1:])

    # Edges (average adjacent edge lines with neighbors)
    RAcorn[0, 1:W] = 0.5 * (ra_top[:-1] + ra_top[1:])
    RAcorn[-1, 1:W] = 0.5 * (ra_bot[:-1] + ra_bot[1:])
    RAcorn[1:H, 0] = 0.5 * (ra_left[:-1] + ra_left[1:])
    RAcorn[1:H, -1] = 0.5 * (ra_right[:-1] + ra_right[1:])

    DECcorn[0, 1:W] = 0.5 * (dec_top[:-1] + dec_top[1:])
    DECcorn[-1, 1:W] = 0.5 * (dec_bot[:-1] + dec_bot[1:])
    DECcorn[1:H, 0] = 0.5 * (dec_left[:-1] + dec_left[1:])
    DECcorn[1:H, -1] = 0.5 * (dec_right[:-1] + dec_right[1:])

    # Four corners (average of adjacent edges)
    RAcorn[0, 0] = 0.5 * (ra_top[0] + ra_left[0])
    RAcorn[0, -1] = 0.5 * (ra_top[-1] + ra_right[0])
    RAcorn[-1, 0] = 0.5 * (ra_bot[0] + ra_left[-1])
    RAcorn[-1, -1] = 0.5 * (ra_bot[-1] + ra_right[-1])

    DECcorn[0, 0] = 0.5 * (dec_top[0] + dec_left[0])
    DECcorn[0, -1] = 0.5 * (dec_top[-1] + dec_right[0])
    DECcorn[-1, 0] = 0.5 * (dec_bot[0] + dec_left[-1])
    DECcorn[-1, -1] = 0.5 * (dec_bot[-1] + dec_right[-1])

    return RAcorn, DECcorn


def plot_on_ra_dec(
    ax,
    ra_corners,
    dec_corners,
    data,
    title=None,
    cbar_title=None,
    time_range=None,
    vmin=None,
    vmax=None,
    norm="linear",
):
    if norm == "linear":
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == "log":
        if vmin is None:
            if np.nanmin(data) > 0:
                vmin = np.nanmin(data)
            else:
                vmin = 1e-3 * np.nanmax(data)
        if vmax is None:
            vmax = np.nanmax(data)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    pm = ax.pcolormesh(
        ra_corners,
        dec_corners,
        data,
        shading="auto",
        # vmin=vmin,
        # vmax=vmax,
        cmap="plasma",
        norm=norm,
    )
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(title)

    # spc_df = pd.read_csv(
    #     "./data/pointing/lexi_look_direction_data_resampled_interpolated_2025-03-02_00-00-00_to_2025-03-16_23-59-59_v0.0.csv"
    # )
    # spc_df["RA"] = spc_df["ra_lexi"]
    # spc_df["DEC"] = spc_df["dec_lexi"]
    #
    # # Set Epoch as index and convert to datetime
    # spc_df["Epoch"] = pd.to_datetime(spc_df["Epoch"], utc=True)
    # spc_df.set_index("Epoch", inplace=True)
    # # Select the dataframe within the time range
    # time_range = pd.to_datetime(time_range, utc=True)
    # spc_df = spc_df.loc[time_range[0] : time_range[1]]
    # ra_center = spc_df["RA"].median()
    # dec_center = spc_df["DEC"].median()
    # ax.set_aspect("equal", adjustable="box")
    # circle = plt.Circle((ra_center, dec_center), 4.55, color="white", fill=False)
    # # Put a dot at the center
    # ax.plot(ra_center, dec_center, marker="o", color="k", markersize=5)
    # # Add an annotation with the center coordinates
    # ax.annotate(
    #     f"({ra_center:.2f}, {dec_center:.2f})",
    #     (ra_center + 0.5, dec_center + 0.5),
    #     color="white",
    #     fontsize=12,
    #     weight="bold",
    #     bbox=dict(facecolor="black", alpha=0.5, pad=2),
    # )
    #
    # ax.add_artist(circle)
    # ax.set_aspect("equal")
    cbar = plt.colorbar(
        pm, ax=ax, label=cbar_title, orientation="vertical", fraction=0.046, pad=0.00
    )

    # Force scientific notation with offset at the top
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))  # force sci notation outside [-1e-3, 1e3]
    cbar.ax.yaxis.set_major_formatter(formatter)
    # Hide minor ticks
    cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())

    # Move the offset (×10^n) to the top of the colorbar
    cbar.ax.yaxis.get_offset_text().set(size=14)  # font size if you like
    cbar.ax.yaxis.get_offset_text().set_position((1.15, 1))
    return pm


def overlay_ra_dec_contours(ax, ra_c, dec_c, n_ra=7, n_dec=7, **kw):
    ra_levels = np.linspace(np.nanmin(ra_c), np.nanmax(ra_c), n_ra)
    dec_levels = np.linspace(np.nanmin(dec_c), np.nanmax(dec_c), n_dec)
    ax.contour(
        ra_c, levels=ra_levels, colors="k", linewidths=0.5, alpha=0.4, extent=None
    )  # drawn in pixel coords for quick look
    ax.contour(dec_c, levels=dec_levels, colors="w", linewidths=0.5, alpha=0.4, extent=None)


def plot_exposure_and_counts(dat: dict, f: str):
    """
    Plot exposure maps and counts from LEXI L2 data.
    Parameters
    ----------
    dat : dict
        Dictionary containing LEXI L2 data arrays.
    f : str
        File path of the LEXI L2 data file (for title and saving).
    Returns
    -------
    None
    """

    ra_c = np.asarray(dat["ra_bin_map"][...])[0]
    dec_c = np.asarray(dat["dec_bin_map"][...])[0]
    time_range = [dat["epoch_start"][...][0], dat["epoch_end"][...][0]]

    RAcorn, DECcorn = centers_to_corners_2d(ra_c, dec_c)
    # Plot the exposure maps and counts
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    # Set the hspace and wspace
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    # Set the default font size
    mpl.rcParams.update({"font.size": 16})
    fig.suptitle(f"LEXI L2 Data from {Path(f).name}", fontsize=20)

    plot_on_ra_dec(
        axs[0, 0],
        RAcorn,
        DECcorn,
        np.asarray(dat["exposure_map"][...])[0],
        time_range=time_range,
        title="Exposure Map",
        cbar_title="Exposure Time (s)",
        norm="log",
        vmin=1e0,
        vmax=3e2,
    )
    plot_on_ra_dec(
        axs[0, 1],
        RAcorn,
        DECcorn,
        np.asarray(dat["flat_field_map"][...])[0],
        title="Flat Field Map",
        cbar_title="Normalized Counts",
        time_range=time_range,
        vmin=1e-1,
        vmax=1e0,
        norm="log",
    )
    plot_on_ra_dec(
        axs[0, 2],
        RAcorn,
        DECcorn,
        np.asarray(dat["background_map"][...])[0],
        title="Background Map",
        cbar_title="Counts/pixel",
        time_range=time_range,
        norm="log",
        vmin=1e-3,
        vmax=1e-1,
    )
    plot_on_ra_dec(
        axs[1, 0],
        RAcorn,
        DECcorn,
        np.asarray(dat["lexi_hist"][...])[0],
        time_range=time_range,
        title="Raw Counts",
        cbar_title="Counts/sec",
        norm="log",
    )
    plot_on_ra_dec(
        axs[1, 1],
        RAcorn,
        DECcorn,
        np.asarray(dat["lexi_histogram_bgnd_corrected"][...])[0],
        time_range=time_range,
        title="Background-Corrected Counts",
        cbar_title="Counts/sec",
        norm="log",
    )

    plot_on_ra_dec(
        axs[1, 2],
        RAcorn,
        DECcorn,
        np.asarray(dat["lexi_histogram_bgnd_flat_corrected"][...])[0],
        time_range=time_range,
        title="Background & Flat-Field Corrected Counts",
        cbar_title="Counts/sec",
        norm="log",
    )

    for ax in axs.flatten():
        ax.contour(RAcorn[:-1, :-1], DECcorn[:-1, :-1], ra_c, colors="c", linewidths=0.6, alpha=0.5)
        ax.contour(
            RAcorn[:-1, :-1], DECcorn[:-1, :-1], dec_c, colors="k", linewidths=0.6, alpha=0.5
        )
        ax.set_aspect("equal")

    figure_path = Path("./figures/exposure_maps/bg_corrected/from_l2/new/")
    figure_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figure_path / (Path(f).stem + "_exposure_and_counts.png"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    return fig
