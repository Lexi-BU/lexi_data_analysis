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
from matplotlib.ticker import FormatStrFormatter
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
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)

    if n_plots == 1:
        axs = [axs]  # Ensure axs is iterable

    # Set x-axis
    if x_axis == "Epoch":
        x = df.index
    else:  # "Epoch_unix"
        x = df["Epoch_unix"]

    for ax, key in zip(axs, keys):
        print(f"Plotting {key}...")
        ax.plot(x, df[key], label=key)
        ax.set_ylabel(key)
        ax.legend(loc="upper right")
        ax.grid(True)

    axs[-1].set_xlabel(f"{x_axis} [UTC]")
    fig.suptitle("Time Series Plot", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # Make sure that the output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if figure_name is None:
        figure_name = f"time_series_plot_{start_time}_{end_time}.png"
    plt.savefig(Path(output_path) / figure_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Plot saved to {Path(output_path) / figure_name} \n")


def plot_histogram(
    df=None,
    start_time=None,
    end_time=None,
    x_key=None,
    y_key=None,
    bins=100,
    cmap="viridis",
    norm_scale="log",
    time_normalization=False,
    data_folder_location=None,
    output_path="figures",
    flat_field_data=False,
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
    flat_field_data : bool, optional
        Whether to use flat field data for normalization.

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

    if x_key is None or y_key is None:
        raise ValueError("Both x_key and y_key must be specified.")

    # Convert time boundaries to UTC-aware timestamps
    def to_utc(ts):
        ts = pd.to_datetime(ts)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts

    if start_time:
        df = df[df.index >= to_utc(start_time)]
    if end_time:
        df = df[df.index <= to_utc(end_time)]

    if df.empty:
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
    H_orig, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)

    if norm_scale == "log":
        norm = mpl.colors.LogNorm(vmin=min(1, np.nanmin(H_orig)), vmax=np.nanmax(H_orig))
    else:
        norm = mpl.colors.Normalize(vmin=np.nanmin(H_orig), vmax=np.nanmax(H_orig))

    H_orig, xedges, yedges, img = ax.hist2d(x, y, bins=bins, weights=weights, cmap=cmap, norm=norm)

    # Prepare output folder
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if flat_field_data:
        flat_field_path = "data/flat_field_data/lexi_l1c_flat_field_data_20240524_20240530.cdf"
        print(f"Reading flat field data from {flat_field_path} \n")
        with cdf(flat_field_path) as cdf_file:
            flat_x = cdf_file[x_key][...]
            flat_y = cdf_file[y_key][...]
            flat_epoch = pd.to_datetime(cdf_file["Epoch"][...])
            flat_index = pd.DatetimeIndex(flat_epoch).tz_localize("UTC")

        # Time-normalized weights if needed
        if time_normalization:
            flat_duration = (flat_index[-1] - flat_index[0]).total_seconds()
            flat_weights = np.ones_like(flat_x) / flat_duration
        else:
            flat_weights = None

        # Compute flat field histogram
        H_flat, _, _ = np.histogram2d(flat_x, flat_y, bins=[xedges, yedges], weights=flat_weights)

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
        print(f"Flat field histogram saved to {Path(output_path) / ff_name} \n")

        # Normalize by flat field histogram
        with np.errstate(divide="ignore", invalid="ignore"):
            H_result = np.divide(H_orig, H_flat)
            H_result[~np.isfinite(H_result)] = 0

        # Plot normalized histogram
        fig_norm, ax_norm = plt.subplots(figsize=(10, 8))
        if norm_scale == "log":
            norm = mpl.colors.LogNorm(vmin=1e-2, vmax=np.nanmax(H_result))
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
        print(f"Normalized histogram saved to {Path(output_path) / norm_name} \n")
    else:
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
        print(f"Histogram saved to {Path(output_path) / orig_name} \n")
