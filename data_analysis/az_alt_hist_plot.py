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
from scipy.ndimage import map_coordinates

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

# Plotting (use original edges, not masked edges)
fig, ax = plt.subplots(figsize=(10, 8))
cmap = mpl.colormaps["viridis"]
cmap.set_bad(color="white")  # Set NaN values to white
# Set the norm to logarithmic scale for better visibility
norm = mpl.colors.LogNorm(vmin=np.nanmin(hist_masked[hist_masked > 0]), vmax=np.nanmax(hist_masked))
# Use original edges for pcolormesh
c = ax.pcolormesh(
    new_xedges,  # Use original x edges
    new_yedges,  # Use original y edges
    hist_masked.T,  # Transpose if needed
    cmap=cmap,
    shading="auto",
    norm=norm,
)

# Add colorbar
cbar = fig.colorbar(c, ax=ax, orientation="vertical")
cbar.set_label("Normalized Counts", fontsize=14)

# Set labels and title
ax.set_xlabel("Altitude (degrees)", fontsize=14)
ax.set_ylabel("Azimuth (degrees)", fontsize=14)
ax.set_title(
    f"Histogram of {input_dict['x_key']} and {input_dict['y_key']} from {input_dict['start_time']} to {input_dict['end_time']}",
    fontsize=16,
)

# Set aspect ratio
ax.set_aspect("equal", adjustable="box")

# Add grid
ax.grid(True, linestyle="--", alpha=0.5)

# Add circle to indicate radius
scaled_radius = 0.04 * 112.5  # Scale the radius to new axis range
circle = patches.Circle(
    (4.5, 4.5), scaled_radius, color="red", fill=False, linestyle="--", linewidth=2
)  # Center at (4.5, 4.5)
ax.add_patch(circle)

# Set the limits of the axes to 0.04
# ax.set_xlim(-0.04, 0.04)
# ax.set_ylim(-0.04, 0.04)
# Set the limits of the axes to 9 degrees
ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
# Save figure
save_fig_folder = Path("../figures/alt/az_hist/")
save_fig_folder.mkdir(parents=True, exist_ok=True)
fig_file_name = (
    f"az_alt_hist_plot_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_"
    f"{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}.png"
)
fig_file_path = save_fig_folder / fig_file_name
fig.savefig(fig_file_path, bbox_inches="tight", dpi=300)

# Close figure
plt.close(fig)

scaled_x_centers = xcenters * 112.5 + 4.5
fig, ax = plt.subplots(figsize=(10, 6))
y_list = np.linspace(-0.04, 0.04, 5)
scaled_y_list = y_list * 112.5 + 4.5  # Scale to 0-9 range
for i, y in enumerate(y_list[0:1]):
    # Find the closest y-bin-index
    y_bin_index = np.argmin(np.abs(ycenters - y))

    # Extract the histogram slice for the given y-bin
    hist_slice = hist_masked[y_bin_index, :]

    # Plot the x vs histogram counts
    ax.plot(scaled_x_centers, hist_slice, label=f"{y*112.5 + 4.5:.1f}", marker=".")
# Set labels and title
ax.set_xlabel("Altitude (degrees)", fontsize=14)
ax.set_ylabel("Normalized Counts", fontsize=14)
# Set yscale to logarithmic for better visibility
ax.set_yscale("log")
ax.set_title(
    f"Line Profile Histogram from {input_dict['start_time']} to {input_dict['end_time']}",
    fontsize=16,
)
# Add legend
ax.legend(title="Azimuth", fontsize=12, title_fontsize=14)
# Set grid
ax.grid(True, linestyle="--", alpha=0.5)
# Save figure
save_fig_file_name = (
    f"line_profile_histogram_plot_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_"
    f"{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}_{scaled_y_list[0]:.0f}_{scaled_y_list[-1]:.0f}.png"
)
save_fig_file_path = save_fig_folder / save_fig_file_name
fig.savefig(save_fig_file_path, bbox_inches="tight", dpi=300)
# Close figure
plt.close(fig)
