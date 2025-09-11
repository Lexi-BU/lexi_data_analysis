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
from matplotlib.cm import get_cmap

importlib.reload(lexi_functions)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Suppress dvision by zero warnings in numpy
np.seterr(divide="ignore", invalid="ignore")


rot_angle = 13.7
input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time": "2025-03-16T19:45:00Z",
    "end_time": "2025-03-16T21:15:00Z",
    # "start_time": "2024-05-23T22:45:00Z", "end_time": "2024-05-30T02:45:00Z", "start_time":
    # "2025-03-06T15:05:00Z", "end_time": "2025-03-06T16:35:00Z",
    "bins": 200,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
    "rotate_data": True,
    "rotation_angle": rot_angle,
}

read_data = False
normalize_against_ground = True
if "hist" not in locals() or "xedges" not in locals() or "yedges" not in locals() or read_data:
    org_hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
        **input_dict
    )
    # Save the histogram data to a pickle file along with start and end time
    save_folder = Path("../data/")
    save_folder.mkdir(parents=True, exist_ok=True)
    file_name = f"line_profile_histogram_data_moon_coordinate_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}.pkl"
    save_file = save_folder / file_name
    with open(save_file, "wb") as f:
        pickle.dump(
            {
                "hist": org_hist,
                "xedges": xedges,
                "yedges": yedges,
                "ra_median": ra_median,
                "dec_median": dec_median,
                "start_time": input_dict["start_time"],
                "end_time": input_dict["end_time"],
                "bins": input_dict["bins"],
                "bin_range": input_dict["bin_range"],
                "time_normalization": input_dict["time_normalization"],
                "x_key": input_dict["x_key"],
                "y_key": input_dict["y_key"],
            },
            f,
        )

if normalize_against_ground:
    ground_file_name = (
        "../data/ground_histogram_data_20240523_224500Z_20240530_024500Z_rot_angle_13.7.pkl"
    )
    with open(ground_file_name, "rb") as f:
        ground_data = pickle.load(f)

    ground_hist = ground_data["hist"]
    # Replace 0 values in ground_hist with nan
    ground_hist = np.where(ground_hist == 0, np.nan, ground_hist)
    ground_xedges = ground_data["xedges"]
    ground_yedges = ground_data["yedges"]

    hist = org_hist / ground_hist
else:
    hist = org_hist

# Select only the part of histogram that is within a radius of 0.04 from the center
# Compute bin centers
xcenters = 0.5 * (ground_xedges[:-1] + ground_xedges[1:])
ycenters = 0.5 * (ground_yedges[:-1] + ground_yedges[1:])

# Create meshgrid of bin centers
X, Y = np.meshgrid(xcenters, ycenters)

# Compute radius from center
radius = np.sqrt(X**2 + Y**2)

# Create mask for radius <= 0.04
mask = radius <= 0.04

# Apply mask to histogram (set outside-circle values to NaN)
hist_masked = hist.copy()
hist_masked[~mask] = np.nan

# New edges (0 to 9 degrees, same number of bins)
new_xedges = (
    ground_xedges * 112.5 + 4.5
)  # Scale to 0-9 degrees (4.5 is the center of the original range)
new_yedges = ground_yedges * 112.5 + 4.5  # Scale to 0-9 degrees

scaled_radius = 0.04 * 112.5
scaled_x_centers = xcenters * 112.5 + 4.5
scaled_y_centers = ycenters * 112.5 + 4.5
y_list = np.linspace(-0.025, 0.025, 3)
scaled_y_list = y_list * 112.5 + 4.5  # Scale to 0-9 range
m_rows = 6
offset = m_rows // 2

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# First subplot: 2D Histogram
mpl.style.use("dark_background")  # Use dark background style
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    }
)
plt.rcParams.update
cmap = mpl.colormaps["plasma"]  # Use the plasma colormap
cmap.set_bad(color="white")  # Set NaN values to white
# norm = mpl.colors.LogNorm(vmin=np.nanmin(hist_masked[hist_masked > 0]), vmax=np.nanmax(hist_masked))
norm = mpl.colors.LogNorm(vmin=1, vmax=1e4)

# Plot the 2D histogram
c = ax1.pcolormesh(
    new_xedges,
    new_yedges,
    hist_masked.T,
    cmap=cmap,
    shading="auto",
    norm=norm,
)

# Add colorbar
cbar = fig.colorbar(c, ax=ax1, orientation="vertical", pad=0.01, shrink=0.9, fraction=0.1)
cbar.set_label("Normalized Counts", fontsize=14)

# Set labels and title
ax1.set_xlabel("Altitude (degrees)", fontsize=14)
ax1.set_ylabel("Azimuth (degrees)", fontsize=14)
ax1.set_title(
    f"Histogram of {input_dict['x_key']} and {input_dict['y_key']}",
    fontsize=16,
)

# Set aspect ratio
ax1.set_aspect("equal", adjustable="box")

# Add grid
# ax1.grid(True, linestyle="--", alpha=0.5)

# Add circle to indicate radius
circle = patches.Circle(
    (4.5, 4.5), scaled_radius, color="red", fill=False, linestyle="--", linewidth=2
)
ax1.add_patch(circle)

# Set the limits of the axes
ax1.set_xlim(0, 9)
ax1.set_ylim(0, 9)
center_x, center_y = 4.5, 4.5
scaled_y_list = scaled_y_list.tolist()
# Add horizontal lines for each y in y_list
# for y in scaled_y_list:
#     # Check if this y-coordinate is within the circle's bounds
#     if abs(y - center_y) <= scaled_radius:
#         dx = np.sqrt(scaled_radius**2 - (y - center_y) ** 2)  # Half-length of chord
#         x_start = center_x - dx
#         x_end = center_x + dx
#         ax1.plot([x_start, x_end], [y, y], color="k", linestyle=":", alpha=1)

# Second subplot: Line profiles
scaled_x_centers = xcenters * 112.5 + 4.5

cmap = mpl.colormaps["jet"]  # Use the plasma colormap
colors = [cmap(i / (len(y_list) - 1)) for i in range(len(y_list))]

for i, (y, color) in enumerate(zip(y_list, colors)):
    y_bin_index = np.argmin(np.abs(ycenters - y))
    y_start = max(0, y_bin_index - offset)
    y_end = min(len(ycenters), y_bin_index + offset + 1)

    hist_slice = hist_masked.T[y_start:y_end, :].mean(axis=0)
    # Normalize the histogram slice
    hist_slice = hist_slice  #  / np.nanmax(hist_slice)
    hist_slice = (
        pd.Series(hist_slice).rolling(window=6, center=True, min_periods=1).mean().to_numpy()
    )

    # Plot line profile on ax2
    ax2.plot(scaled_x_centers, hist_slice, label=f"{scaled_y_list[i]:.1f}°", color=color)
    # Add the colorbar corresponding to the line profile
    ax2.scatter(
        scaled_x_centers,
        hist_slice,
        color=color,
        s=10,
        alpha=0.5,
        edgecolor="none",
    )
    # Plot vertical line at the center of the histogram
    ax2.axvline(center_x, color="k", linestyle="--", linewidth=1)
    # Plot horizontal line at the center of the histogram
    ax2.axhline(0, color="k", linestyle="--", linewidth=1)
    # Plot vertical line at the center of the histogram
    ax1.axvline(center_x, color="k", linestyle="--", linewidth=1)

    # Plot corresponding horizontal line on ax1
    if abs(scaled_y_list[i] - center_y) <= scaled_radius:
        dx = np.sqrt(scaled_radius**2 - (scaled_y_list[i] - center_y) ** 2)
        x_start = center_x - dx
        x_end = center_x + dx
        ax1.plot([x_start, x_end], [scaled_y_list[i]] * 2, color=color, linestyle=":", alpha=1)

    # Optional: highlight the region (m_rows bins)
    ax1.fill_betweenx(
        [scaled_y_centers[y_start], scaled_y_centers[y_end - 1]],
        scaled_x_centers[0],
        scaled_x_centers[-1],
        color="k",
        alpha=0.1,
    )

# Add the colorbar for the line profiles
# cbar2 = fig.colorbar(
#     mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#     ax=ax2,
#     orientation="vertical",
#     pad=0.01,
#     shrink=0.9,
#     fraction=0.1,
# )
# cbar2.set_label("Normalized Counts", fontsize=14)

# Set labels and title
ax2.set_xlabel("Altitude (degrees)", fontsize=14)
ax2.set_ylabel("Normalized Counts", fontsize=14)
ax2.set_yscale("log")
ax2.set_title(
    "Line Profiles at Different Azimuth Angles",
    fontsize=16,
)

# Add legend
ax2.legend(title="Azimuth Angle", fontsize=10, title_fontsize=12)

# Add grid
# ax2.grid(True, linestyle="--", alpha=0.5)

# Adjust layout
plt.tight_layout()

# Save figure
save_fig_folder = Path("../figures/az_hist/band/")
save_fig_folder.mkdir(parents=True, exist_ok=True)
fig_file_name = (
    f"combined_plot_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_"
    f"{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}_{scaled_y_list[0]:.0f}_{scaled_y_list[-1]:.0f}_{len(scaled_y_list)}_band_{m_rows}.png"
)
fig_file_path = save_fig_folder / fig_file_name
fig.savefig(fig_file_path, bbox_inches="tight", dpi=300)

print(f"Figure saved to {fig_file_path}")

# Close figure
plt.close(fig)
