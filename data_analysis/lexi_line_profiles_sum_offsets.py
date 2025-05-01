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
        perp_line_length = 0.2
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 2D histogram
    im = ax1.imshow(
        hist.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        norm=mpl.colors.Normalize(vmin=10, vmax=50),
        # norm=mpl.colors.Normalize(vmin=0, vmax=np.nanmax(hist)),
    )
    fig.colorbar(im, ax=ax1, label="Counts/s", pad=0.01, shrink=0.9, fraction=0.1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Normalized 2D Histogram with Line\n" "histogram/ground histogram")

    ax1.set_xlim(-0.1, 0.1)
    ax1.set_ylim(-0.1, 0.1)

    # Add a  circle at the center of the histogram of radius 0.04
    circle = patches.Circle(
        (x_offset, y_offset), 0.04, color="lime", fill=False, linestyle="--", linewidth=1, zorder=20
    )
    # ax1.add_patch(circle)
    # x_centers = (xedges[:-1] + xedges[1:]) / 2
    # y_centers = (yedges[:-1] + yedges[1:]) / 2

    # for i in range(len(x_centers)):
    #     for j in range(len(y_centers)):
    #         value = hist[i, j]
    #         if value > 0:  # Only display text for non-zero bins
    #             ax1.text(
    #                 x_centers[i],
    #                 y_centers[j],
    #                 f"{value:.1f}",
    #                 color="green",
    #                 ha="center",
    #                 va="center",
    #                 fontsize=5,
    #             )
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
        # Find the index of the closest point
        closest_index = np.argmin(closest_dist)
        x_line_closest = perp_line_x[closest_index]
        y_line_closest = perp_line_y[closest_index]
        dx = x_line_closest - x_offset
        dy = y_line_closest - y_offset
        tolerance_value = 1e-3
        if theta <= 180:
            if np.isclose(dy, 0, atol=tolerance_value):
                if np.isclose(dx, 0, atol=tolerance_value):
                    closest_sign = 1
                else:
                    closest_sign = np.sign(dx)
            else:
                closest_sign = np.sign(dy)
        else:
            if np.isclose(dy, 0, atol=tolerance_value):
                if np.isclose(dx, 0, atol=tolerance_value):
                    closest_sign = 1
                else:
                    closest_sign = -np.sign(dx)
            else:
                closest_sign = -np.sign(dy)

        dist2_list = []
        values2_list = []
        if min_dist <= 0.1:
            # Select all the values where the distance is within 0.1
            for dist, value in zip(perp_dist_list, perp_value[xi, yi]):
                if abs(dist) <= 0.1:
                    dist2_list.append(dist)
                    values2_list.append(value)

            dist2.append(closest_sign * min_dist)
            # Find the number of points where values2_list is nan
            non_nan_indices_length = len(np.where(~np.isnan(values2_list))[0])
            # nan_indices = len(values2_list)
            values2.append(np.nansum(values2_list) / non_nan_indices_length)

    # Sum values along the primary line profile (where dist1 is between -0.1 and 0.1)
    mask = (np.array(dist1) >= -0.1) & (np.array(dist1) <= 0.1)
    sum_values1 = np.nansum(np.array(values1)[mask])

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
                color="red",
                linestyle="--",
                linewidth=0.5,
                alpha=0.1,
            )
            # Add the perp_value beside the perpendicular line profile
            # ax1.text(
            #     x_perp_line[xi, yi][0],
            #     y_perp_line[xi, yi][0],
            #     f"{sum(perp_value[xi, yi]):.1f}",
            #     color="cyan",
            #     fontsize=5,
            #     ha="center",
            #     va="bottom",
            # )

    # Plot primary line profile (left axis)
    ax2.scatter(
        np.array(dist1)[mask],
        np.array(values1)[mask],
        color="lime",
        marker="d",
        label=f"θ={theta}°",
        alpha=1,
        s=5,
    )
    # At each point, write down the hist value beside it
    # for dist, value in zip(
    #     np.array(dist1)[mask],
    #     np.array(values1)[mask],
    # ):
    #     ax2.text(
    #         dist,
    #         value,
    #         f"{value:.1f}",
    #         color="lime",
    #         fontsize=5,
    #         ha="left",
    #         va="bottom",
    #     )
    ax2b = ax2.twinx()
    ax2b.scatter(
        dist2,
        values2,
        color="red",
        marker=".",
        label="Perpendicular Values",
        alpha=1,
        s=2,
    )
    # ax2b.set_ylim(100, 800)
    ax2b.set_ylim(0.002, 0.013)
    # Select the values2 that are within 0.04 distance from the x_offset, y_offset
    values2_selected = []
    dist2_selected = []
    for dist, value in zip(dist2, values2):
        if abs(dist) <= 0.04:
            dist2_selected.append(dist)
            values2_selected.append(value)
    # Get a best fit line for the selected values2
    if len(dist2_selected) > 0:
        """
        # Fit a 1st-degree polynomial (line)
        coeffs, residuals, _, _, _ = np.polyfit(dist2_selected, values2_selected, 1, full=True)
        poly_fit = np.poly1d(coeffs)
        x_fit = np.linspace(-0.04, 0.04, 100)
        y_fit = poly_fit(x_fit)

        # In a csv file, save the slope, coeffs and residuals along with theta
        best_fit_file = Path(f"../data/line_profile_best_fit_theta_offset_2042_2102_2d.csv")
        best_fit_file.parent.mkdir(parents=True, exist_ok=True)
        # Check if the file already exists
        if best_fit_file.exists():
            # Read the existing file
            df = pd.read_csv(best_fit_file)
            # Append the new data to the existing file
            new_data = {
                "theta": theta,
                "slope": coeffs[0],
                "intercept": coeffs[1],
                "residuals": residuals[0],
            }
            # Add the new data to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(best_fit_file, index=False)
        else:
            # Create a new DataFrame and save it to a CSV file
            df = pd.DataFrame(
                {
                    "theta": [theta],
                    "slope": [coeffs[0]],
                    "intercept": [coeffs[1]],
                    "residuals": [residuals[0]],
                }
            )
            df.to_csv(best_fit_file, index=False)

        ax2b.plot(x_fit, y_fit, color="w", linestyle="--", linewidth=2)
        # Add the equation of the line to the plot right above the line
        ax2b.text(
            0.02,
            0.37,
            f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f} \n residuals = {residuals[0]:.3f}",
            transform=ax2b.transAxes,
            fontsize=12,
            # Rotate the text by the slope of the line
            rotation=8,
            horizontalalignment="left",
            verticalalignment="top",
            bbox=dict(facecolor="k", alpha=0.5, edgecolor="none"),
        )
        """
        # Fit a 2nd-degree polynomial (parabola)
        coeffs, residuals, _, _, _ = np.polyfit(dist2_selected, values2_selected, 2, full=True)
        poly_fit = np.poly1d(coeffs)
        x_fit = np.linspace(-0.04, 0.04, 100)
        y_fit = poly_fit(x_fit)

        best_fit_file = Path(f"../data/line_profile_best_fit_theta_offset_2042_2102_2d.csv")
        best_fit_file.parent.mkdir(parents=True, exist_ok=True)

        if best_fit_file.exists():
            df = pd.read_csv(best_fit_file)
            new_data = {
                "theta": theta,
                "quadratic": coeffs[0],
                "linear": coeffs[1],
                "intercept": coeffs[2],
                "residuals": residuals[0] if len(residuals) > 0 else np.nan,
            }
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(best_fit_file, index=False)
            # Drop the duplicate rows
            df = df.drop_duplicates(subset=["theta"], keep="last")
        else:
            df = pd.DataFrame(
                {
                    "theta": [theta],
                    "quadratic": [coeffs[0]],
                    "linear": [coeffs[1]],
                    "intercept": [coeffs[2]],
                    "residuals": [residuals[0] if len(residuals) > 0 else np.nan],
                }
            )
            df.to_csv(best_fit_file, index=False)

        ax2b.plot(x_fit, y_fit, color="w", linestyle="--", linewidth=2)
        ax2b.text(
            0.99,
            0.99,
            (
                f"y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}\nresiduals = {residuals[0]:.2e}"
                if len(residuals) > 0
                else f"y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}"
            ),
            transform=ax2b.transAxes,
            fontsize=12,
            rotation=0,
            horizontalalignment="right",
            verticalalignment="top",
            bbox=dict(facecolor="k", alpha=0.5, edgecolor="none"),
        )

    # for dist, value in zip(dist2, values2):
    #     ax2b.text(
    #         dist,
    #         value,
    #         f"{value:.1f}",
    #         color="red",
    #         fontsize=5,
    #         ha="left",
    #         va="bottom",
    #     )
    ax2.set_xlabel("Distance from (x_offset, y_offset) along the line")
    ax2.set_ylabel("Histogram Value", color="lime")
    ax2.tick_params(axis="y", labelcolor="lime")

    # At top left of the plot, display the sum_values1 and the theta
    # ax2.text(
    #     0.02,
    #     0.98,
    #     f"Sum counts: {sum_values1:.3f}\nθ={theta:.1f}°",
    #     transform=ax2.transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     bbox=dict(facecolor="k", alpha=0.5, edgecolor="none"),
    # )
    # ax2.minor_ticks_on(a)
    ax2.grid(axis="both", which="major", linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.grid(axis="both", which="minor", linestyle=":", linewidth=0.1, alpha=0.5)
    ax2b.grid(axis="both", which="major", linestyle=":", linewidth=0.1, alpha=0.5)
    ax2b.grid(axis="both", which="minor", linestyle=":", linewidth=0.1, alpha=0.5)
    ax2b.set_ylabel("Perpendicular Histogram Value", color="red")
    ax2b.tick_params(axis="y", labelcolor="red")
    # Set the number of minor ticks
    ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

    # Set the number of major ticks
    ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)

    ax2.set_title(f"Line Profiles through ({x_offset}, {y_offset})")

    ax2.set_xlim(-0.04, 0.04)
    # ax2.set_ylim(0, 30)
    ax2.set_ylim(0.01, 0.05)
    ax2.set_yscale("linear")

    # ax2b.set_ylim(0, 0.014)
    # ax2.set_yscale("linear")
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
            right=False,
            top=True,
            bottom=True,
        )
    ax2b.tick_params(
        which="both",
        direction="in",
        length=6,
        width=0.5,
        colors="red",
        grid_color="c",
        left=False,
        right=True,
        top=True,
        bottom=True,
    )
    # Set the spine color to match the line color
    ax2b.spines["left"].set_color("lime")
    # Set the tick labels color to match the line color on the left axis, for ax2
    ax2.tick_params(axis="y", colors="lime")

    # ax2b.spines["bottom"].set_color("lime")
    # Set the tick labels color to match the line color
    # ax2b.tick_params(axis="y", colors="lime")
    ax2b.spines["right"].set_color("red")
    plt.tight_layout()

    save_theta = np.round(theta, 1)
    save_theta = str(save_theta).zfill(6)
    start_date_str = start_date.replace(":", "").replace("-", "").replace("T", "_")
    end_date_str = end_data.replace(":", "").replace("-", "").replace("T", "_")
    save_folder = Path(f"../figures/line_profiles_v2/{start_date_str}_{end_date_str}/")
    save_folder.mkdir(parents=True, exist_ok=True)
    fig_name = f"scaled_shifted_non_normalized_sunset_single_linear_line_profile_theta_{save_theta}_offset_{x_offset:0.3f}_{y_offset:0.3f}.png"
    # if theta >= 180:
    #     fig_name = "test2.png"
    # else:
    #     fig_name = "test.png"
    fig.savefig(save_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    # print(f"Line profile plot saved to {save_folder / fig_name}")

    plt.close(fig)
    return theta, sum_values1, perp_value, perp_dist


input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time": "2025-03-16T20:42:00Z",
    "end_time": "2025-03-16T21:02:00Z",
    # "start_time": "2024-05-23T22:45:00Z",
    # "end_time": "2024-05-30T02:45:00Z",
    # "start_time": "2025-03-06T15:05:00Z",
    # "end_time": "2025-03-06T16:35:00Z",
    "bins": 200,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
    "rotate_data": False,
    "rotation_angle": -18.6,
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
    file_name = f"line_profile_histogram_data_moon_coordinate_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}.pkl"
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

n_shift_bin_x = 0
n_shift_bin_y = 0
print("Histogram data loaded successfully.")
if normalize_against_ground:
    ground_file_name = "../data/ground_test_histogram_data_20240523_224500Z_20240530_024500Z.pkl"
    with open(ground_file_name, "rb") as f:
        ground_data = pickle.load(f)
    ground_hist = ground_data["hist"]
    # Replace 0 values in ground_hist with nan
    ground_hist = np.where(ground_hist == 0, np.nan, ground_hist)

    # Shift ground_hist n_shift_bin bins to the left
    ground_hist = np.roll(ground_hist, shift=n_shift_bin_x, axis=0)  # y-direction
    ground_hist = np.roll(ground_hist, shift=n_shift_bin_y, axis=1)  # x-direction

    if n_shift_bin_y > 0:
        ground_hist[:n_shift_bin_y, :] = np.nan
    elif n_shift_bin_y < 0:
        ground_hist[n_shift_bin_y:, :] = np.nan  # This works since negative indices wrap correctly

    # Invalidate wrapped-around columns (x-direction)
    if n_shift_bin_x > 0:
        ground_hist[:, :n_shift_bin_x] = np.nan
    elif n_shift_bin_x < 0:
        ground_hist[:, n_shift_bin_x:] = np.nan

    ny, nx = ground_hist.shape
    xedges = ground_data["xedges"]
    yedges = ground_data["yedges"]
    x_grid, y_grid = np.meshgrid(xedges[:-1], yedges[:-1])

    # Apply scaling to bin coordinates (rescale the coordinates by a factor of alpha)
    alpha = 1.1
    x_scaled = (x_grid - xedges[0]) * alpha + xedges[0]
    y_scaled = (y_grid - yedges[0]) * alpha + yedges[0]

    # Interpolate the histogram data to the scaled coordinates
    rescaled_ground_hist = map_coordinates(
        ground_hist,
        [y_scaled.ravel(), x_scaled.ravel()],
        order=1,
        mode="nearest",
        # cval=np.nan,
    ).reshape(ny, nx)
    # Normalize the histogram data against the ground data
    hist = org_hist / ground_hist
    rescaled_hist = org_hist / rescaled_ground_hist

    # Replace inf values with nan
    hist = np.where(np.isinf(hist), np.nan, hist)

    rescaled_hist = np.where(np.isinf(rescaled_hist), np.nan, rescaled_hist)
else:
    hist = org_hist
    # Replace 0 values in hist with nan
    # hist = np.where(hist == 0, np.nan, hist)
    # Replace inf values with nan
    hist = np.where(np.isinf(hist), np.nan, hist)
print("Histogram data normalized against ground data.")

theta_list = np.linspace(0, 180, num=181, endpoint=True)
theta_index = 3

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
    # for theta in theta_list:
    print(f"Processing line profile for theta = {theta:0.2f} degrees", end="\r")
    # Ensure the histogram is loaded
    if "hist" not in locals():
        hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
            **input_dict
        )

    # Generate the line profile and plot it
    theta, sum_values1, perp_sum_value, perp_dist = plot_line_profile(
        hist, xedges, yedges, theta, x_offset, y_offset, start_date, end_date
    )
    sum_values.append(sum_values1)


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
            "perp_value": perp_sum_value,
            "perp_dist": perp_dist,
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

"""
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
fig.colorbar(im, ax=ax1, label="counts/s", pad=0.01, shrink=0.9, fraction=0.1)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("2D Histogram")

ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)

# Plot the sum of values for each theta
ax2.plot(theta_list, sum_values, color="lime", linewidth=0.5, ls="--", alpha=0.3)
ax2.scatter(
    theta_list,
    sum_values,
    color="lime",
    marker="o",
    alpha=1,
    s=5,
)
ax2.set_xlabel("θ [°]")
ax2.set_ylabel("Sum of Histogram Values")
ax2.set_title("Sum of Histogram Values vs. Theta")
ax2.set_xlim(0, 180)
# ax2.set_ylim(0, 0.03)
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
