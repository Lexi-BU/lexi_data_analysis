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

    # Calculate line endpoints for plotting (extend to edges of plot)
    theta_rad = np.deg2rad(theta)
    if theta == 90:  # Vertical line
        x_line = [x_offset, x_offset]
        y_line = [yedges[0], yedges[-1]]
    else:
        slope = np.tan(theta_rad)
        # Find intersections with plot boundaries
        x_min, x_max = xedges[0], xedges[-1]
        y_min, y_max = yedges[0], yedges[-1]

        # Calculate possible endpoints
        endpoints = []
        # Intersection with left boundary (x_min)
        y = y_offset + slope * (x_min - x_offset)
        if y_min <= y <= y_max:
            endpoints.append((x_min, y))
        # Intersection with right boundary (x_max)
        y = y_offset + slope * (x_max - x_offset)
        if y_min <= y <= y_max:
            endpoints.append((x_max, y))
        # Intersection with bottom boundary (y_min)
        x = x_offset + (y_min - y_offset) / slope
        if x_min <= x <= x_max:
            endpoints.append((x, y_min))
        # Intersection with top boundary (y_max)
        x = x_offset + (y_max - y_offset) / slope
        if x_min <= x <= x_max:
            endpoints.append((x, y_max))

        # Take the two endpoints farthest apart
        if len(endpoints) >= 2:
            # Find pair with maximum distance
            max_dist = 0
            best_pair = None
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

    # Plot the line on the histogram
    ax1.plot(
        x_line,
        y_line,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax1.plot(x_offset, y_offset, "rd")  # Mark the offset point
    # Put a marker at 0.1 distance from the line in the direction of the line
    line_length = np.sqrt((x_line[1] - x_line[0]) ** 2 + (y_line[1] - y_line[0]) ** 2)
    if line_length > 0:
        # Normalize the line vector
        dx = (x_line[1] - x_line[0]) / line_length
        dy = (y_line[1] - y_line[0]) / line_length
        # Point 0.1 units away from (x_offset, y_offset) along the line
        x_marker = x_offset + 0.1 * dx
        y_marker = y_offset + 0.1 * dy
        # Add a marker in both directions
        ax1.plot(x_marker, y_marker, "go", markersize=6, label="0.1 units from line")
        # Also add a marker in the opposite direction
        x_marker_neg = x_offset - 0.1 * dx
        y_marker_neg = y_offset - 0.1 * dy
        ax1.plot(x_marker_neg, y_marker_neg, "go", markersize=6)
        ax1.legend(loc="upper right", fontsize="small", frameon=False)
        ax1.set_xlim(xedges[0], xedges[-1])
        ax1.set_ylim(yedges[0], yedges[-1])

    # Get and plot the line profile
    distances, values = get_line_profile(hist, xedges, yedges, theta, x_offset, y_offset)
    ax2.scatter(distances, values, color="c", s=5, label="Line Profile")
    ax2.set_xlabel("Distance from (x_offset, y_offset) along the line")
    ax2.set_ylabel("Histogram Value")
    ax2.set_title(f"Line Profile at θ={theta:0.1f}° through ({x_offset:0.3f}, {y_offset:0.3f})")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.2, color="aqua")

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
    plt.tight_layout()

    save_theta = np.round(theta, 1)
    save_theta = str(save_theta).zfill(6)
    save_folder = Path("../figures/line_profiles/")
    save_folder.mkdir(parents=True, exist_ok=True)
    fig_name = f"sunset_single_linear_line_profile_theta_{save_theta}_offset_{x_offset:0.3f}_{y_offset:0.3f}.png"
    fig.savefig(save_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    # print(f"Line profile plot saved to {save_folder / fig_name}")

    plt.close(fig)
    return None


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


theta_list = np.linspace(0, 180, num=180, endpoint=True)

# Find the maximum value in the histogram and its corresponding coordinates
max_index = np.unravel_index(np.argmax(hist, axis=None), hist.shape)
# x_offset = (xedges[max_index[0]] + xedges[max_index[0] + 1]) / 2
# y_offset = (yedges[max_index[1]] + yedges[max_index[1] + 1]) / 2
x_offset = 0  # X coordinate of the point the line must pass through
y_offset = 0  # Y coordinate of the point the line must pass through
for theta in theta_list:
    print(f"Processing line profile for theta = {theta:0.2f} degrees", end="\r")
    # Ensure the histogram is loaded
    if "hist" not in locals():
        hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
            **input_dict
        )

    # Generate the line profile and plot it
    plot_line_profile(hist, xedges, yedges, theta, x_offset, y_offset)
# Generate the plot
# plot_line_profile(hist, xedges, yedges, theta, x_offset, y_offset)
# plt.show()
