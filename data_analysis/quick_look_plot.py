import glob
import re
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
from spacepy.pycdf import CDF as cdf


def get_file_list(data_folder_location, start_time, end_time):
    # Construct the folder path
    folder_name = data_folder_location + start_time.strftime("%Y-%m-%d")

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

    return filtered_files


def read_all_data_files(file_list, return_data_type="dataframe"):
    all_data_dict = {}
    for file in file_list:
        try:
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
        # Set the time zone of Epoch to UTC
        df["Epoch"] = pd.to_datetime(df["Epoch"], unit="s", utc=True)
        # Convert the index to datetime
        try:
            df["Epoch"] = df["Epoch"].dt.tz_convert("UTC")
        except Exception:
            df["Epoch"] = df["Epoch"].dt.tz_localize("UTC")
        # Set the index to the Epoch column
        df.set_index("Epoch", inplace=True)
        # Select only rows that are within the time range
        df = df.loc[start_time:end_time]
        # Convert the index to datetime
        return df
    elif return_data_type == "dict":
        return all_data_dict
    else:
        return None


def make_2d_histogram(df, key1, key2, bins=100, norm_style="log", save_fig=False):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))
    mincnt = 1
    if norm_style == "log":
        norm = mpl.colors.LogNorm(vmin=mincnt)
    else:
        norm = mpl.colors.Normalize(vmin=0)
    # Create a hexbin plot
    ax.hexbin(
        df[key1],
        df[key2],
        gridsize=bins,
        cmap="inferno",
        mincnt=mincnt,
        norm=norm,
        edgecolors="face",
        linewidths=0.1,
        # bins="log",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(key1)
    ax.set_ylabel(key2)

    cb = plt.colorbar(
        ax.collections[0],
        ax=ax,
        orientation="vertical",
        pad=0.01,
        aspect=50,
        shrink=0.8,
        fraction=0.02,
        label="Counts",
        extend="max",
        extendfrac=0.07,
        extendrect=True,
        location="right",
    )
    cb.ax.xaxis.set_ticks_position("top")
    ax.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_xlim(df[key1].min(), df[key1].max())
    ax.set_ylim(df[key2].min(), df[key2].max())
    plt.title(f"{key1} vs {key2} 2D Histogram")
    plt.tight_layout()
    if save_fig:
        save_fig_folder = Path("../figures")
        save_fig_folder.mkdir(parents=True, exist_ok=True)
        min_time = df.index.min().strftime("%Y-%m-%d_%H-%M-%S")
        max_time = df.index.max().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(
            save_fig_folder / f"{key1}_vs_{key2}_2D_histogram_{min_time}_{max_time}.png",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()
    return fig, ax


data_folder_location = "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/"

start_time = "2025-03-06T15:05:00.00Z"
end_time = "2025-03-06T16:34:00.00Z"

start_time = parser.parse(start_time)
end_time = parser.parse(end_time)

# Get the list of files
file_list = get_file_list(data_folder_location, start_time, end_time)

# Read all the data files
df = read_all_data_files(file_list, return_data_type="dataframe")

# Make a 2D histogram of x_cm vs y_cm
fig, ax = make_2d_histogram(
    df,
    key1="x_cm",
    key2="y_cm",
    bins=200,
    norm_style="linear",
    save_fig=True,
)
