import importlib
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


def get_line_profile(hist, xedges, yedges, theta, x_offset, y_offset):
    """
    Extract values from a 2D histogram along a specified line.

    Parameters:
    - hist: 2D numpy array of histogram values
    - xedges, yedges: Bin edges for x and y dimensions
    - theta: Angle in degrees (0 is horizontal, 90 is vertical)
    - x_offset, y_offset: Point that the line must pass through

    Returns:
    - distances: Distances from (x_offset, y_offset) along the line
    - values: Histogram values along the line
    """
    # Convert angle to radians
    theta_rad = np.deg2rad(theta)

    # Calculate centers of bins
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Create meshgrid of centers
    X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")

    # Calculate distance from each point to the line
    # Line equation: (y - y0) = tan(theta) * (x - x0)
    if theta == 90:  # Vertical line (avoid division by zero)
        distances = X - x_offset
        mask = np.isclose(X, x_offset, atol=(xedges[1] - xedges[0]) / 2)
    else:
        slope = np.tan(theta_rad)
        # Distance along line direction (parametric form)
        # Projection of (x-x0, y-y0) onto line direction (cosθ, sinθ)
        distances = (X - x_offset) * np.cos(theta_rad) + (Y - y_offset) * np.sin(theta_rad)
        # Check if point is on the line within tolerance
        mask = np.isclose(Y - y_offset, slope * (X - x_offset), atol=(yedges[1] - yedges[0]) / 2)
        perpendicular_slope = -1 / slope if slope != 0 else np.inf

    # Get distances and values for points on/near the line
    line_distances = distances[mask]
    line_values = hist[mask]

    # Sort by distance
    sort_idx = np.argsort(line_distances)
    line_distances = line_distances[sort_idx]
    line_values = line_values[sort_idx]

    return line_distances, line_values


def plot_line_profile(hist, xedges, yedges, theta, x_offset, y_offset):
    """
    Plot the histogram and the line profile.
    """
    # Use the dark background for better visibility
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

    # Get primary line profile and its perpendicular slope
    dist1, values1 = get_line_profile(hist, xedges, yedges, theta, x_offset, y_offset)

    sum_values1 = np.sum(values1)
    # Calculate angle for perpendicular line (add 90 degrees)
    theta_perp = theta + 90
    if theta_perp >= 360:
        theta_perp -= 360

    # Get perpendicular line profile
    dist2, values2 = get_line_profile(hist, xedges, yedges, theta_perp, x_offset, y_offset)

    # Plot both lines on the histogram
    def plot_line(ax, theta, color):
        """Helper function to plot a line at given angle through offset point"""
        theta_rad = np.deg2rad(theta)
        if theta == 90:  # Vertical line
            x_line = [x_offset, x_offset]
            y_line = [yedges[0], yedges[-1]]
        else:
            slope = np.tan(theta_rad)
            x_min, x_max = xedges[0], xedges[-1]
            y_min, y_max = yedges[0], yedges[-1]

            endpoints = []
            # Intersection with left boundary
            y = y_offset + slope * (x_min - x_offset)
            if y_min <= y <= y_max:
                endpoints.append((x_min, y))
            # Intersection with right boundary
            y = y_offset + slope * (x_max - x_offset)
            if y_min <= y <= y_max:
                endpoints.append((x_max, y))
            # Intersection with bottom boundary
            if slope != 0:
                x = x_offset + (y_min - y_offset) / slope
                if x_min <= x <= x_max:
                    endpoints.append((x, y_min))
                # Intersection with top boundary
                x = x_offset + (y_max - y_offset) / slope
                if x_min <= x <= x_max:
                    endpoints.append((x, y_max))

            if len(endpoints) >= 2:
                max_dist = 0
                best_pair = (0, 1)
                for i in range(len(endpoints)):
                    for j in range(i + 1, len(endpoints)):
                        dist = (endpoints[i][0] - endpoints[j][0]) ** 2 + (
                            endpoints[i][1] - endpoints[j][1]
                        ) ** 2
                        if dist > max_dist:
                            max_dist = dist
                            best_pair = (i, j)
                x_line = [endpoints[best_pair[0]][0], endpoints[best_pair[1]][0]]
                y_line = [endpoints[best_pair[0]][1], endpoints[best_pair[1]][1]]
            else:
                x_line = [x_offset, x_offset]
                y_line = [y_offset, y_offset]

        ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=1)
        ax.plot(x_offset, y_offset, "ko", markersize=4)  # Mark the offset point
        # Put a marker at 0.1 distance from the line in the direction of the line
        line_length = np.sqrt((x_line[1] - x_line[0]) ** 2 + (y_line[1] - y_line[0]) ** 2)
        if line_length > 0:
            # Normalize the direction vector
            dx = (x_line[1] - x_line[0]) / line_length
            dy = (y_line[1] - y_line[0]) / line_length
            # Point 0.1 units away from (x_offset, y_offset) in the line direction
            x_marker = x_offset + 0.1 * dx
            y_marker = y_offset + 0.1 * dy
            ax.plot(x_marker, y_marker, "ro", markersize=4)  # Mark the 0.1 point
            x_marker_neg = x_offset - 0.1 * dx
            y_marker_neg = y_offset - 0.1 * dy
            ax.plot(x_marker_neg, y_marker_neg, "ro", markersize=4)  # Mark the -0.1 point

    # Plot primary line
    plot_line(ax1, theta, "lime")
    # Plot perpendicular line (blue)
    plot_line(ax1, theta_perp, "cyan")
    # Plot primary line profile (left axis)
    ax2.scatter(dist1, values1, color="lime", marker="o", label=f"θ={theta}°", s=2)
    ax2.set_xlabel("Distance from (x_offset, y_offset) along the line")
    ax2.set_ylabel("Histogram Value", color="lime")
    ax2.tick_params(axis="y", labelcolor="lime")
    ax2.grid(True)

    # Create twin axis for perpendicular line profile
    ax2b = ax2.twinx()
    ax2b.scatter(dist2, values2, color="cyan", marker="d", label=f"θ={theta_perp}°", s=2)
    ax2b.set_ylabel("Histogram Value", color="cyan")
    ax2b.tick_params(axis="y", labelcolor="cyan")
    ax2b.set_ylim(0.0, 0.04)

    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    ax2.set_title(f"Line Profiles through ({x_offset}, {y_offset})")

    ax2.set_xlim(-0.1, 0.1)
    ax2.set_ylim(0.0, 0.04)
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
    ax2b.spines["left"].set_color("lime")
    # Set the tick labels color to match the line color
    ax2.tick_params(axis="y", colors="lime")
    ax2b.spines["right"].set_color("cyan")
    plt.tight_layout()

    save_theta = np.round(theta, 1)
    save_theta = str(save_theta).zfill(6)
    save_folder = Path("../figures/line_profiles_v2/")
    save_folder.mkdir(parents=True, exist_ok=True)
    fig_name = f"20250407_sunset_double_linear_line_profile_theta_{save_theta}_offset_{x_offset:0.3f}_{y_offset:0.3f}.png"
    fig.savefig(save_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    # print(f"Line profile plot saved to {save_folder / fig_name}")

    plt.close(fig)
    return theta, sum_values1


input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time": "2025-03-16T19:45:00Z",
    "end_time": "2025-03-16T20:15:00Z",
    "bins": 200,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
}


read_data = True
if "hist" not in locals() or "xedges" not in locals() or "yedges" not in locals() or read_data:
    hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
        **input_dict
    )


theta_list = np.linspace(0, 180, num=180, endpoint=False)

# Find the maximum value in the histogram and its corresponding coordinates
max_index = np.unravel_index(np.argmax(hist, axis=None), hist.shape)
# x_offset = (xedges[max_index[0]] + xedges[max_index[0] + 1]) / 2
# y_offset = (yedges[max_index[1]] + yedges[max_index[1] + 1]) / 2
x_offset = 0.0  # X coordinate of the point the line must pass through
y_offset = 0.0  # Y coordinate of the point the line must pass through
sum_values = []
for theta in theta_list:
    print(f"Processing line profile for theta = {theta:0.2f} degrees", end="\r")
    # Ensure the histogram is loaded
    if "hist" not in locals():
        hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
            **input_dict
        )

    # Generate the line profile and plot it
    theta, sum_values1 = plot_line_profile(hist, xedges, yedges, theta, x_offset, y_offset)
    sum_values.append(sum_values1)


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
