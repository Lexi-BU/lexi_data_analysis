import importlib
import pickle
import warnings
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

importlib.reload(lexi_functions)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Suppress dvision by zero warnings in numpy
np.seterr(divide="ignore", invalid="ignore")


def get_histogram_values_along_line_both_directions(
    hist, xedges, yedges, theta, x_offset, y_offset
):
    """
    Extract values from a 2D histogram along a specified line in both directions.
    Parameters:
    - hist: 2D numpy array of histogram values
    - xedges, yedges: Bin edges for x and y dimensions
    - theta: Angle in degrees (0 is horizontal, 90 is vertical)
    - x_offset, y_offset: Point that the line must pass through
    Returns:
    - x_line: X coordinates of the line
    - y_line: Y coordinates of the line
    - distance: Distances from (x_offset, y_offset) along the line
    - value: Histogram values along the line
    """
    # Convert angle to radians
    theta = np.deg2rad(theta)
    # Convert theta to a direction vector
    dx = np.cos(theta)
    dy = np.sin(theta)

    # Define line endpoints (adjust range if needed)
    line_length = max(xedges[-1] - xedges[0], yedges[-1] - yedges[0]) * 2
    x0 = x_offset - dx * line_length / 2
    x1 = x_offset + dx * line_length / 2
    y0 = y_offset - dy * line_length / 2
    y1 = y_offset + dy * line_length / 2

    # Generate points along the line
    num_points = 1000
    x_line = np.linspace(x0, x1, num_points)
    y_line = np.linspace(y0, y1, num_points)

    # Find bin indices for each point
    x_indices = np.searchsorted(xedges, x_line) - 1
    y_indices = np.searchsorted(yedges, y_line) - 1

    # Clip to valid range
    x_indices = np.clip(x_indices, 0, hist.shape[0] - 1)
    y_indices = np.clip(y_indices, 0, hist.shape[1] - 1)

    # Get unique bin indices the line passes through
    bin_indices = set(zip(x_indices, y_indices))

    # Print values and indices
    # print("Bins and values the line passes through:")
    value = []
    distance = []
    perp_value = {}
    perp_distance = {}
    x_perp_line = {}
    y_perp_line = {}
    for xi, yi in bin_indices:
        # Compute bin center
        x_center = 0.5 * (xedges[xi] + xedges[xi + 1])
        y_center = 0.5 * (yedges[yi] + yedges[yi + 1])
        # Find the equation of a line passing through x_center, y_center and perpendicular to the
        # line passing thorugh x_line and y_line
        perp_line_length = 0.1
        x0_offset = x_center - perp_line_length * dy / 2
        y0_offset = y_center + perp_line_length * dx / 2
        x1_offset = x_center + perp_line_length * dy / 2
        y1_offset = y_center - perp_line_length * dx / 2
        perp_x_offset = x0_offset
        perp_y_offset = y0_offset
        perp_num_points = 1000
        x_line_perp = np.linspace(x0_offset, x1_offset, perp_num_points)
        y_line_perp = np.linspace(y0_offset, y1_offset, perp_num_points)
        x_indices_perp = np.searchsorted(xedges, x_line_perp) - 1
        y_indices_perp = np.searchsorted(yedges, y_line_perp) - 1
        x_indices_perp = np.clip(x_indices_perp, 0, hist.shape[0] - 1)
        y_indices_perp = np.clip(y_indices_perp, 0, hist.shape[1] - 1)
        bin_indices_perp = set(zip(x_indices_perp, y_indices_perp))
        perp_value_list = []
        perp_distance_list = []
        for xi_perp, yi_perp in bin_indices_perp:
            # Compute bin center
            x_center_perp = 0.5 * (xedges[xi_perp] + xedges[xi_perp + 1])
            y_center_perp = 0.5 * (yedges[yi_perp] + yedges[yi_perp + 1])
            dx_bin_perp = x_center_perp - perp_x_offset
            dy_bin_perp = y_center_perp - perp_y_offset
            directed_distance_perp = dx_bin_perp * dx + dy_bin_perp * dy
            perp_distance_list.append(directed_distance_perp)
            perp_value_list.append(hist[xi_perp, yi_perp])
            # print(
            #     f"Perpendicular Bin: ({x_center_perp:.3f}, {y_center_perp:.3f}), "
            #     f"Distance: {directed_distance_perp:.3f}, Value: {hist[xi_perp, yi_perp]}"
            # )
        # Add the perp values to the dictionary using xi and yi as keys
        perp_distance[xi, yi] = perp_distance_list
        perp_value[xi, yi] = perp_value_list
        x_perp_line[xi, yi] = x_line_perp
        y_perp_line[xi, yi] = y_line_perp

        # Compute bin center
        dx_bin = x_center - x_offset
        dy_bin = y_center - y_offset
        # Project onto direction vector (theta)
        directed_distance = dx_bin * dx + dy_bin * dy
        distance.append(directed_distance)
        value.append(hist[xi, yi])

    return x_line, y_line, distance, value, perp_distance, perp_value, x_perp_line, y_perp_line


def plot_line_profile(hist, xedges, yedges, theta, x_offset, y_offset, start_date, end_data):
    """
    Plot the histogram and the line profile.
    """
    # Use the dark background for better visibility
    plt.style.use("dark_background")
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Plot 2D histogram
    im = ax1.imshow(
        hist.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )
    fig.colorbar(im, ax=ax1, label="Counts")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("2D Histogram with Line")

    ax1.set_xlim(-0.1, 0.1)
    ax1.set_ylim(-0.1, 0.1)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    for i in range(len(x_centers)):
        for j in range(len(y_centers)):
            value = hist[i, j]
            if value > 0:  # Only display text for non-zero bins
                ax1.text(
                    x_centers[i],
                    y_centers[j],
                    f"{value:.1f}",
                    color="green",
                    ha="center",
                    va="center",
                    fontsize=5,
                )
    # Get primary line profile and its perpendicular slope
    x_line, y_line, dist1, values1, perp_dist, perp_value, x_perp_line, y_perp_line = (
        get_histogram_values_along_line_both_directions(
            hist, xedges, yedges, theta, x_offset, y_offset
        )
    )

    # Find the distances between x_offset, y_offset and each of the x_perp_line and y_perp_line
    dist2 = []
    values2 = []
    for (xi, yi), perp_dist_list in perp_dist.items():
        perp_line_x = x_perp_line[xi, yi]
        perp_line_y = y_perp_line[xi, yi]
        # Find the closest point to (x_offset, y_offset)
        closest_dist = np.sqrt((perp_line_x - x_offset) ** 2 + (perp_line_y - y_offset) ** 2)
        min_dist = np.min(closest_dist)
        dist2_list = []
        values2_list = []
        if min_dist <= 0.1:
            # Select all the values where the distance is within 0.1
            for dist, value in zip(perp_dist_list, perp_value[xi, yi]):
                if abs(dist) <= 0.1:
                    dist2_list.append(dist)
                    values2_list.append(value)
            dist2.append(min_dist)
            values2.append(np.sum(values2_list))

    # print(f"Perpendicular distances: {dist2} \n")
    # print(f"Perpendicular values: {values2}")
    # Sum values along the primary line profile (where dist1 is between -0.1 and 0.1)
    mask = (np.array(dist1) >= -0.1) & (np.array(dist1) <= 0.1)
    sum_values1 = np.sum(np.array(values1)[mask])

    ax1.plot(x_line, y_line, color="green", linestyle="--", linewidth=1)
    ax1.plot(x_offset, y_offset, "ko", markersize=4)  # Mark the offset point
    # Put a marker at 0.1 distance from the line in the direction of the line
    line_length = np.sqrt((x_line[1] - x_line[0]) ** 2 + (y_line[1] - y_line[0]) ** 2)
    if line_length > 0:
        # Normalize the direction vector
        dx = (x_line[1] - x_line[0]) / line_length
        dy = (y_line[1] - y_line[0]) / line_length
        # Point 0.1 units away from (x_offset, y_offset) in the line direction
        x_marker = x_offset + 0.1 * dx
        y_marker = y_offset + 0.1 * dy
        ax1.plot(x_marker, y_marker, "ro", markersize=4)  # Mark the 0.1 point
        x_marker_neg = x_offset - 0.1 * dx
        y_marker_neg = y_offset - 0.1 * dy
        ax1.plot(x_marker_neg, y_marker_neg, "ro", markersize=4)  # Mark the -0.1 point

    # Select perp_dist list and perp_value only if that line is withing 0.1 distance from the
    # x_offset, y_offset
    perp_dist_list = []
    for (xi, yi), perp_dist_list in perp_dist.items():
        if len(perp_dist_list) > 0:
            perp_dist_list = [dist for dist in perp_dist_list if abs(dist) <= 0.1]
            perp_dist[(xi, yi)] = perp_dist_list
    perp_value_list = []
    for (xi, yi), perp_value_list in perp_value.items():
        if len(perp_value_list) > 0:
            perp_value_list = [
                value
                for dist, value in zip(perp_dist[(xi, yi)], perp_value_list)
                if abs(dist) <= 0.1
            ]
            perp_value[(xi, yi)] = perp_value_list
    # Plot all the perpendicular line profiles
    for (xi, yi), perp_dist_list in perp_dist.items():
        if len(perp_dist_list) > 0:
            ax1.plot(
                x_perp_line[xi, yi],
                y_perp_line[xi, yi],
                color="cyan",
                linestyle="--",
                linewidth=1,
            )
            # Add the perp_value beside the perpendicular line profile
            ax1.text(
                x_perp_line[xi, yi][0],
                y_perp_line[xi, yi][0],
                f"{sum(perp_value[xi, yi]):.1f}",
                color="cyan",
                fontsize=5,
                ha="center",
                va="bottom",
            )

    # Plot primary line profile (left axis)
    # print(f"Distance values along the line: {np.array(dist1)[mask]}")
    # print(f"Histogram values along the line: {np.array(values1)[mask]}")
    ax2.scatter(
        np.array(dist1)[mask],
        np.array(values1)[mask],
        color="lime",
        marker="o",
        label=f"θ={theta}°",
        s=2,
    )
    ax2.scatter(
        dist2,
        values2,
        color="red",
        marker="d",
        label="Perpendicular Values",
        s=2,
    )
    # At each point, write down the hist value beside it
    for dist, value in zip(
        np.array(dist1)[mask],
        np.array(values1)[mask],
    ):
        ax2.text(
            dist,
            value,
            f"{value:.1f}",
            color="lime",
            fontsize=5,
            ha="left",
            va="bottom",
        )
    for dist, value in zip(dist2, values2):
        ax2.text(
            dist,
            value,
            f"{value:.1f}",
            color="red",
            fontsize=5,
            ha="left",
            va="bottom",
        )
    ax2.set_xlabel("Distance from (x_offset, y_offset) along the line")
    ax2.set_ylabel("Histogram Value", color="lime")
    ax2.tick_params(axis="y", labelcolor="lime")

    # At top left of the plot, display the sum_values1 and the theta
    ax2.text(
        0.02,
        0.98,
        f"Sum counts: {sum_values1:.3f}\nθ={theta:.1f}°",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="k", alpha=0.5, edgecolor="none"),
    )
    # ax2.minor_ticks_on(a)
    ax2.grid(axis="both", which="major", linestyle="-", linewidth=0.5, alpha=0.5)

    ax2.grid(axis="both", which="minor", linestyle=":", linewidth=0.1, alpha=0.5)
    # Set the number of minor ticks
    ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

    # Set the number of major ticks
    ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    # Create twin axis for perpendicular line profile
    # ax2b = ax2.twinx()
    # ax2b.scatter(dist2, values2, color="cyan", marker="d", label=f"θ={theta_perp}°", s=2)
    # ax2b.set_ylabel("Histogram Value", color="cyan")
    # ax2b.tick_params(axis="y", labelcolor="cyan")
    # ax2b.set_ylim(0.0, 0.04)

    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    # lines2, labels2 = ax2b.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    ax2.set_title(f"Line Profiles through ({x_offset}, {y_offset})")

    ax2.set_xlim(-0.1, 0.1)
    # ax2.set_ylim(0.0, 0.04)
    ax2.set_yscale("linear")
    # Set the maximum number of ticks for both axes to avoid clutter
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

    # Set tickmarks inside the plots on all sides
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("w")
        ax.tick_params(
            which="both",
            direction="in",
            length=6,
            width=0.5,
            colors="w",
            grid_color="c",
            left=True,
            right=True,
            top=True,
            bottom=True,
        )
    # Set the spine color to match the line color
    # ax2.spines["left"].set_color("lime")
    # Set the tick labels color to match the line color
    # ax2.tick_params(axis="y", colors="lime")
    # ax2b.spines["right"].set_color("cyan")
    plt.tight_layout()

    save_theta = np.round(theta, 1)
    save_theta = str(save_theta).zfill(6)
    start_date_str = start_date.replace(":", "").replace("-", "").replace("T", "_")
    end_date_str = end_data.replace(":", "").replace("-", "").replace("T", "_")
    save_folder = Path(f"../figures/line_profiles_v2/{start_date_str}_{end_date_str}/")
    save_folder.mkdir(parents=True, exist_ok=True)
    fig_name = f"sunset_single_linear_line_profile_theta_{save_theta}_offset_{x_offset:0.3f}_{y_offset:0.3f}.png"
    fig.savefig(save_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    # print(f"Line profile plot saved to {save_folder / fig_name}")

    plt.close(fig)
    return theta, sum_values1, perp_value, perp_dist


input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time": "2025-03-16T20:01:00Z",
    "end_time": "2025-03-16T20:55:00Z",
    # "start_time": "2025-03-06T15:05:00Z",
    # "end_time": "2025-03-06T16:35:00Z",
    "bins": 10,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
}


read_data = False
if "hist" not in locals() or "xedges" not in locals() or "yedges" not in locals() or read_data:
    hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
        **input_dict
    )

theta_list = np.linspace(0, 180, num=181, endpoint=True)
theta_index = 30

# Find the maximum value in the histogram and its corresponding coordinates
max_index = np.unravel_index(np.argmax(hist, axis=None), hist.shape)
# x_offset = (xedges[max_index[0]] + xedges[max_index[0] + 1]) / 2
# y_offset = (yedges[max_index[1]] + yedges[max_index[1] + 1]) / 2
x_offset = 0.0  # X coordinate of the point the line must pass through
y_offset = 0.0  # Y coordinate of the point the line must pass through

# x_offset_list = np.linspace(-0.1, 0.1, num=10)
# y_offset_list = np.linspace(-0.1, 0.1, num=10)
sum_values = []
start_date = input_dict["start_time"]
end_date = input_dict["end_time"]
for theta in theta_list[theta_index : theta_index + 1]:
    print(f"Processing line profile for theta = {theta:0.2f} degrees", end="\r")
    # Ensure the histogram is loaded
    if "hist" not in locals():
        hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
            **input_dict
        )

    # Generate the line profile and plot it
    theta, sum_values1, perps_value, perp_dist = plot_line_profile(
        hist, xedges, yedges, theta, x_offset, y_offset, start_date, end_date
    )
    sum_values.append(sum_values1)

"""
# Save, theta, sum_values, x_offset, y_offset, hist, xedges, yedges, ra_median, dec_median to a
# pickle file
save_folder = Path("../data/")
save_folder.mkdir(parents=True, exist_ok=True)
file_name = f"line_profile_data_{theta_list[0]}_{theta_list[-1]}_{len(theta_list)}.pkl"
save_file = save_folder / file_name
with open(save_file, "wb") as f:
    pickle.dump(
        {
            "theta_list": theta_list,
            "sum_values": sum_values,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "hist": hist,
            "xedges": xedges,
            "yedges": yedges,
            "ra_median": ra_median,
            "dec_median": dec_median,
        },
        f,
    )
# Load the data from the pickle file
# with open(save_file, "rb") as f:
#     data = pickle.load(f)
#     theta_list = data["theta_list"]
#     sum_values = data["sum_values"]
#     x_offset = data["x_offset"]
#     y_offset = data["y_offset"]
#     hist = data["hist"]
#     xedges = data["xedges"]
#     yedges = data["yedges"]
#     ra_median = data["ra_median"]
#     dec_median = data["dec_median"]


plt.style.use("dark_background")
# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 2D histogram
im = ax1.imshow(
    hist.T,
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
)
fig.colorbar(im, ax=ax1, label="Counts")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("2D Histogram with Line")

ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)

# Plot the sum of values for each theta
ax2.plot(theta_list, sum_values, color="lime", marker="o", markersize=2)
ax2.set_xlabel("Theta (degrees)")
ax2.set_ylabel("Sum of Histogram Values")
ax2.set_title("Sum of Histogram Values vs. Theta")
ax2.set_xlim(0, 180)
# ax2.set_ylim(0, 0.04)
ax2.set_yscale("linear")
# Set the maximum number of ticks for both axes to avoid clutter
ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
# Set tickmarks inside the plots on all sides
for ax in (ax1, ax2):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("w")
    ax.tick_params(
        which="both",
        direction="in",
        length=6,
        width=0.5,
        colors="w",
        grid_color="c",
        left=True,
        right=True,
        top=True,
        bottom=True,
    )
# Set the spine color to match the line color
ax2.spines["left"].set_color("lime")
# Set the tick labels color to match the line color
ax2.tick_params(axis="y", colors="lime")
plt.tight_layout()
save_theta = np.round(theta, 1)
save_theta = str(save_theta).zfill(6)
save_folder = Path("../figures/line_profiles_v2/")
save_folder.mkdir(parents=True, exist_ok=True)
fig_name = f"20250407_sunset_double_linear_line_profile_sum_values.png"
fig.savefig(save_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
# close the figure
plt.close(fig)
# Generate the plot
# plot_line_profile(hist, xedges, yedges, theta, x_offset, y_offset)
# plt.show()
"""
