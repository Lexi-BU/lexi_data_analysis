import datetime
import glob
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
from spacepy.pycdf import CDF as cdf

# Get the coolwarm colormap
# coolwarm = plt.get_cmap("coolwarm")
#
# # Create a new colormap that transitions from coolwarm's blue to black to coolwarm's red
# colors = [
#     coolwarm(0.0),  # Blue from coolwarm
#     (0.0, 0.0, 0.0, 1.0),  # Black for the center
#     coolwarm(1.0),  # Red from coolwarm
# ]
#
# # Create the custom colormap
# n_bins = 120  # Number of bins for smooth transitions
# cmap_name = "coolwarm_black_center"
# blue_black_red = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


def get_file_list(data_folder_location, start_time, end_time):
    """Get a list of CDF files within the specified time range."""

    start_time = parser.parse(start_time)
    end_time = parser.parse(end_time)
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
        # Select only rows that are within the time range
        df = df.loc[start_time:end_time]
        # Convert the index to datetime
        return df
    elif return_data_type == "dict":
        return all_data_dict
    else:
        return None


def plot_histogram_diff(
    df=None,
    df_lexi=None,
    start_time=None,
    end_time=None,
    delta_time=None,
    bins=None,
    x_key=None,
    y_key=None,
    extent=None,
    norm_style="log",
    mincnt=None,
    save_folder=None,
    save_figures=False,
):
    """Plot the difference between two histograms."""
    plt.style.use("dark_background")
    # Set the default font size
    plt.rcParams.update({"font.size": 18})

    if df.empty:
        print("Dataframe is empty")
        return
    if type(start_time) == str:
        start_time = parser.parse(start_time)
    if type(end_time) == str:
        end_time = parser.parse(end_time)
    if type(delta_time) == str:
        delta_time_str = delta_time
        delta_time = datetime.timedelta(seconds=float(delta_time))

    if mincnt is None:
        mincnt = 1
    else:
        mincnt = mincnt
    if norm_style == "log":
        norm = mpl.colors.LogNorm(vmin=mincnt)
    else:
        norm = mpl.colors.Normalize(vmin=mincnt)
    if (df.index[-1] - df.index[0]).total_seconds() < delta_time.total_seconds():
        print("Time range is smaller than delta_time")
        return
    else:
        time_ranges = pd.date_range(start=start_time, end=end_time, freq=delta_time)
        time_ranges = time_ranges[time_ranges <= end_time]

        for i in range(len(time_ranges) - 2):
            try:
                start_time_left = time_ranges[i]
                end_time_left = time_ranges[i + 1]
                start_time_right = time_ranges[i + 1]
                end_time_right = time_ranges[i + 2]

                df_left = df.loc[start_time_left:end_time_left]
                df_right = df.loc[start_time_right:end_time_right]

                x_left = df_left[x_key]
                y_left = df_left[y_key]
                x_right = df_right[x_key]
                y_right = df_right[y_key]

                hist_left, xedges_left, yedges_left = np.histogram2d(
                    x_left, y_left, bins=bins, range=[extent[0:2], extent[2:]]
                )
                hist_right, xedges_right, yedges_right = np.histogram2d(
                    x_right, y_right, bins=bins, range=[extent[0:2], extent[2:]]
                )

                hist_center = hist_right - hist_left

                # Normalize the hist_center so that the maximum value is 1
                hist_center = hist_center / np.max(np.abs(hist_center))

                hist_center = abs(hist_center)

                # If the value fo hist_center is within 1e-5 of 0, set it to NaN
                # hist_center[np.abs(hist_center) < 0.2] = np.nan
                # Make a 1 by 3 grid to plot the three histograms
                fig, axs = plt.subplots(2, 2, figsize=(16, 16))
                fig.suptitle(
                    f"{start_time_left.strftime('%Y-%m-%d %H:%M:%S')} to {end_time_right.strftime('%Y-%m-%d %H:%M:%S')}\n delta_time = {delta_time.total_seconds()} seconds"
                )

                # Plot the left histogram
                im_left = axs[0, 0].imshow(
                    hist_left.T,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    norm=norm,
                    cmap="inferno",
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
                    ax=axs[0, 0],
                    orientation="horizontal",
                    pad=0.1,
                    aspect=70,
                    fraction=0.02,
                    location="top",
                )

                # Plot the center histogram
                im_center = axs[1, 0].imshow(
                    hist_center.T,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    norm=mpl.colors.Normalize(),
                    cmap="Blues",
                )
                axs[1, 0].set_title("Normalized Diff")
                axs[1, 0].set_xlabel(x_key)
                axs[1, 0].set_ylabel(y_key)
                # Set the aspect ratio to be equal
                axs[1, 0].set_aspect("equal", adjustable="box")
                # Add the colorbar
                cbar_center = fig.colorbar(
                    im_center,
                    ax=axs[1, 0],
                    orientation="horizontal",
                    pad=0.1,
                    aspect=70,
                    fraction=0.02,
                    location="top",
                )

                # Plot the right histogram
                im_right = axs[0, 1].imshow(
                    hist_right.T,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    norm=norm,
                    cmap="inferno",
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
                    ax=axs[0, 1],
                    orientation="horizontal",
                    pad=0.1,
                    aspect=70,
                    fraction=0.02,
                    location="top",
                )

                # Select the lexi_data within the time range
                df_lexi_range = df_lexi.loc[time_ranges[i] : time_ranges[i + 2]]
                # df_lexi_range = df_lexi
                # Make the RA and Dec plots
                axs[1, 1].scatter(
                    df_lexi_range.index, df_lexi_range["ra_lexi"], label="RA", s=1, color="c"
                )
                axs[1, 1].set_ylabel("RA (deg)")
                axs_dec = axs[1, 1].twinx()
                axs_dec.scatter(
                    df_lexi_range.index,
                    df_lexi_range["dec_lexi"],
                    label="Dec",
                    s=1,
                    color="magenta",
                )
                axs_dec.set_ylabel("Dec (deg)")
                # Rotate the x-axis tick labels for better readability
                plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)
                axs[1, 1].set_xlabel("Time [UTC]")
                # plt.xlim(start_time_left, end_time_right)
                # plt.tight_layout()

                # Managing the tickmarks
                # 1. Set all tickmarks to be inside the plot
                axs[0, 0].tick_params(
                    axis="both", direction="in", left=True, right=True, bottom=True, top=True
                )
                axs[1, 0].tick_params(
                    axis="both", direction="in", left=True, right=True, bottom=True, top=True
                )
                axs[0, 1].tick_params(
                    axis="both", direction="in", left=True, right=True, bottom=True, top=True
                )
                # 2. Set the maximum number of ticks to 5 for each axis
                axs[0, 0].locator_params(axis="x", nbins=5)
                axs[0, 0].locator_params(axis="y", nbins=5)
                axs[1, 0].locator_params(axis="x", nbins=5)
                axs[1, 0].locator_params(axis="y", nbins=5)
                axs[0, 1].locator_params(axis="x", nbins=5)
                axs[0, 1].locator_params(axis="y", nbins=5)
                # 3. For axs[0, 1], set the y-axis labels to the right, and hide the y-axis labels for axs[1, 0]
                axs[0, 1].yaxis.set_label_position("right")
                axs[1, 0].set_ylabel("")
                # 4. Turn on the grid
                axs[0, 0].grid(True, color="c", alpha=0.2)
                axs[1, 0].grid(True, color="k", alpha=0.2)
                axs[0, 1].grid(True, color="c", alpha=0.2)

                if save_figures:
                    save_fig_folder = Path(save_folder)
                    save_fig_folder.mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        save_fig_folder
                        / f"{delta_time_str}_{start_time_left.strftime('%Y-%m-%d_%H-%M-%S')}_{end_time_right.strftime('%Y-%m-%d_%H-%M-%S')}_{x_key}_vs_{y_key}_2D_Histogram.png"
                    )
                    plt.close(fig)
                    print(
                        f"Figure saved as {delta_time_str}_{start_time_left.strftime('%Y-%m-%d_%H-%M-%S')}_{end_time_right.strftime('%Y-%m-%d_%H-%M-%S')}_{x_key}_vs_{y_key}_2D_Histogram.png",
                        end="\r",
                    )
            except Exception as e:
                print(f"Error plotting histogram: {e}")
                pass
    return


df_lexi = pd.read_csv("../data/lexi_look_direction_data.csv")
# Set the Epoch column to index
df_lexi["Epoch"] = pd.to_datetime(df_lexi["Epoch"])
df_lexi = df_lexi.set_index("Epoch")

start_date = 10
start_hour = 0
end_hour = 24
if __name__ == "__main__":
    for month in range(3, 4):
        for day in range(start_date, start_date + 1):
            for hour in range(start_hour, end_hour):
                try:
                    start_time = f"2025-{month:02d}-{day:02d}T{hour:02d}:00:00Z"
                    end_time = f"2025-{month:02d}-{day:02d}T{hour:02d}:59:59Z"
                    input_data = {
                        "data_folder_location": "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/",
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                    df = read_all_data_files(kwargs=input_data)

                    plot_histogram_diff(
                        df=df,
                        df_lexi=df_lexi,
                        start_time=input_data["start_time"],
                        end_time=input_data["end_time"],
                        delta_time="600",
                        bins=100,
                        mincnt=10,
                        x_key="x_volt_lin",
                        y_key="y_volt_lin",
                        extent=[-0.1, 0.1, -0.1, 0.1],
                        norm_style="linear",
                        save_folder=f"../figures/diff_histograms/{start_date:02d}-{month:02d}",
                        save_figures=True,
                    )
                except Exception as e:
                    print(f"Error processing: {e}")
                    pass
