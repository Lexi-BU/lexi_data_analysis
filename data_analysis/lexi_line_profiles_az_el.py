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

radius_val = 5  # Radius of the circle in degrees


def get_histogram_values_along_line_both_directions(
    hist, xedges, yedges, theta, x_offset, y_offset
):
    """
    Extract values from a 2D histogram along a specified line in both directions. Parameters: - hist:
    2D numpy array of histogram values - xedges, yedges: Bin edges for x and y dimensions - theta:
    Angle in ° (0 is horizontal, 90 is vertical) - x_offset, y_offset: Point that the line must
    pass through Returns: - x_line: X coordinates of the line - y_line: Y coordinates of the line -
    distance: Distances from (x_offset, y_offset) along the line - value: Histogram values along the
    line
    """
    # Convert angle to radians
    theta = np.deg2rad(theta)
    # Convert theta to a direction vector
    dx = np.cos(theta)
    dy = np.sin(theta)

    # Define line endpoints (adjust range if needed)
    line_length = max(xedges[-1] - xedges[0], yedges[-1] - yedges[0])
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

    # Print values and indices print("Bins and values the line passes through:")
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
        perp_line_length = 9.2
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
            # print( f"Perpendicular Bin: ({x_center_perp:.3f}, {y_center_perp:.3f}), " f"Distance:
            #     {directed_distance_perp:.3f}, Value: {hist[xi_perp, yi_perp]}" )
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


def plot_line_profile(
    hist,
    xedges,
    yedges,
    theta,
    x_offset,
    y_offset,
    start_date,
    end_date,
    alpha,
    normalize_against_ground,
    rot_angle,
    delta_time,
    version_number,
):
    """
    Plot the histogram and the line profile.
    """
    # Use the dark background for better visibility
    plt.style.use("dark_background")
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if normalize_against_ground:
        norm = mpl.colors.LogNorm(vmin=1, vmax=np.nanmax(hist))
    else:
        norm = mpl.colors.Normalize(vmin=0, vmax=np.nanmax(hist))
    # Plot 2D histogram
    im = ax1.imshow(
        hist.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="equal",
        cmap="inferno",
        norm=norm,
    )
    fig.colorbar(im, ax=ax1, label="Counts/s", pad=0.01, shrink=0.9, fraction=0.1)
    ax1.set_xlabel("Azimuth [°]", fontsize=14)
    ax1.set_ylabel("Elevation [°]", fontsize=14)

    # ax1.set_xlim(265.5, 276.5)
    # ax1.set_ylim(19.5, 30.5)
    ax1.set_xlim(263, 280)
    ax1.set_ylim(15, 32)
    # tick_positions = np.linspace(263, 281, 5)

    # Corresponding labels from 0 to 9
    # tick_labels = np.linspace(0, 9, 5).astype(int).astype(str)

    # Apply to both axes
    # ax1.set_xticks(tick_positions)
    # ax1.set_xticklabels(tick_labels)

    # ax1.set_yticks(tick_positions)
    # ax1.set_yticklabels(tick_labels)

    # Add a  circle at the center of the histogram of radius 9.1
    circle = patches.Circle(
        (x_offset, y_offset),
        radius_val,
        color="lime",
        fill=False,
        linestyle="--",
        linewidth=1,
        zorder=20,
    )
    ax1.add_patch(circle)
    # x_centers = (xedges[:-1] + xedges[1:]) / 2 y_centers = (yedges[:-1] + yedges[1:]) / 2

    # for i in range(len(x_centers)): for j in range(len(y_centers)): value = hist[i, j] if value >
    #     0:  # Only display text for non-zero bins ax1.text( x_centers[i], y_centers[j],
    #         f"{value:.1f}", color="green", ha="center", va="center", fontsize=5, ) Get primary line
    #         profile and its perpendicular slope
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
        tolerance_value = 1e-1
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
        if min_dist <= radius_val:
            # Select all the values where the distance is within radius_val
            for dist, value in zip(perp_dist_list, perp_value[xi, yi]):
                if abs(dist) <= radius_val:
                    dist2_list.append(dist)
                    values2_list.append(value)

            dist2.append(closest_sign * min_dist)
            # Find the number of points where values2_list is nan
            non_nan_indices_length = len(np.where(~np.isnan(values2_list))[0])
            # nan_indices = len(values2_list)
            values2.append(np.nansum(values2_list) / non_nan_indices_length)

    # Sum values along the primary line profile (where dist1 is between -radius_val and radius_val)
    mask = (np.array(dist1) >= -radius_val) & (np.array(dist1) <= radius_val)
    sum_values1 = np.nansum(np.array(values1)[mask])

    ax1.plot(x_line, y_line, color="white", linestyle="--", linewidth=3)
    ax1.plot(x_offset, y_offset, "wo", markersize=9)  # Mark the offset point

    # Add the value of rotation angle to the plot
    ax1.text(
        0.01,
        0.99,
        f"θ={theta:.1f}°\nSum counts: {sum_values1:.3f}",
        transform=ax1.transAxes,
        fontsize=12,
        rotation=0,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=dict(facecolor="k", alpha=0.5, edgecolor="none"),
    )
    # Put a marker at radius_val distance from the line in the direction of the line
    line_length = np.sqrt((x_line[1] - x_line[0]) ** 2 + (y_line[1] - y_line[0]) ** 2)
    if line_length > 0:
        # Normalize the direction vector
        dx = (x_line[1] - x_line[0]) / line_length
        dy = (y_line[1] - y_line[0]) / line_length
        # Point radius_val units away from (x_offset, y_offset) in the line direction
        x_marker = x_offset + radius_val * dx
        y_marker = y_offset + radius_val * dy
        ax1.plot(x_marker, y_marker, "wd", markersize=10, zorder=25)  # Mark the radius_val point
        x_marker_neg = x_offset - radius_val * dx
        y_marker_neg = y_offset - radius_val * dy
        ax1.plot(
            x_marker_neg, y_marker_neg, "wd", markersize=10, zorder=25
        )  # Mark the -radius_val point

    # Select perp_dist list and perp_value only if that line is withing radius_val distance from the
    # x_offset, y_offset
    perp_dist_list = []
    for (xi, yi), perp_dist_list in perp_dist.items():
        if len(perp_dist_list) > 0:
            perp_dist_list = [dist for dist in perp_dist_list if abs(dist) <= radius_val]
            perp_dist[(xi, yi)] = perp_dist_list
    perp_value_list = []
    for (xi, yi), perp_value_list in perp_value.items():
        if len(perp_value_list) > 0:
            perp_value_list = [
                value
                for dist, value in zip(perp_dist[(xi, yi)], perp_value_list)
                if abs(dist) <= radius_val
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
            # Add the perp_value beside the perpendicular line profile ax1.text( x_perp_line[xi,
            # yi][0], y_perp_line[xi, yi][0], f"{sum(perp_value[xi, yi]):.1f}", color="cyan",
            #     fontsize=5, ha="center", va="bottom", )

    # At each point, write down the hist value beside it for dist, value in zip(
    # np.array(dist1)[mask], np.array(values1)[mask], ): ax2.text( dist, value, f"{value:.1f}",
    #     color="lime", fontsize=5, ha="left", va="bottom", )

    # ax2.set_ylim(0.002, 0.013) Select the values2 that are within 0.04 distance from the x_offset,
    # y_offset
    values2_selected = []
    dist2_selected = []
    for dist, value in zip(dist2, values2):
        if abs(dist) <= radius_val:
            dist2_selected.append(dist)
            values2_selected.append(value)

    # Scale the dist2_selected from -0.04 to 0.04 to 0 to 9.1
    dist2_selected = np.array(dist2_selected)
    # print(
    #     f"Max and min of dist2_selected: {np.nanmax(dist2_selected)}, {np.nanmin(dist2_selected)}"
    # )
    # dist2_scaled = (dist2_selected + 0.04) / (0.08) * 9.1
    # dist2_selected = dist2_scaled

    # Scale the values2_selected depending on the maximum value of values2_selected
    max_value = np.nanmax(values2_selected)
    if max_value > 0:
        values2_selected_scaled = np.array(values2_selected) / max_value
    else:
        values2_selected_scaled = np.array(values2_selected)

    # values2_selected = values2_selected_scaled
    # Shift the dist2_selected to start at the x_offset - radius_val
    dist2_selected = y_offset + dist2_selected
    ax2.scatter(
        values2_selected,
        dist2_selected,
        color="lime",
        marker=".",
        label="Perpendicular Values",
        alpha=1,
        s=2,
    )

    # ax2.set_ylim(0, 500) Get a best fit line for the selected values2
    if len(dist2_selected) > 0:
        normalize_against_ground = True
        if normalize_against_ground:
            # Fit a 1st-degree polynomial (line)
            coeffs, residuals, _, _, _ = np.polyfit(dist2_selected, values2_selected, 1, full=True)
            poly_fit = np.poly1d(coeffs)
            x_fit = np.linspace(
                y_offset - radius_val, y_offset + radius_val, 100
            )  # Updated x_fit range
            y_fit = poly_fit(x_fit)

            # In a csv file, save the slope, coeffs and residuals along with theta
            best_fit_file = Path("../data/line_profile_best_fit_theta_offset_2042_2102_2d.csv")
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

            ax2.plot(y_fit, x_fit, color="w", linestyle="--", linewidth=2)
            # Add the equation of the line to the plot right above the line
            ax2.text(
                0.99,
                0.99,
                f"x = {coeffs[0]:.3f}y + {coeffs[1]:.3f} \n residuals = {residuals[0]:.3f}",
                transform=ax2.transAxes,
                fontsize=12,
                # Rotate the text by the slope of the line
                rotation=0,
                horizontalalignment="right",
                verticalalignment="top",
                bbox=dict(facecolor="k", alpha=0.5, edgecolor="none"),
            )
        else:
            # Fit a 2nd-degree polynomial (parabola)
            coeffs, residuals, _, _, _ = np.polyfit(dist2_selected, values2_selected, 2, full=True)
            poly_fit = np.poly1d(coeffs)
            # x_fit = np.linspace(-0.04, 0.04, 100)
            x_fit = np.linspace(0, 9.1, 100)
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

            ax2.plot(x_fit, y_fit, color="w", linestyle="--", linewidth=2)
            ax2.text(
                0.99,
                0.99,
                (
                    f"y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}\nresiduals = {residuals[0]:.2e}"
                    if len(residuals) > 0
                    else f"y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}"
                ),
                transform=ax2.transAxes,
                fontsize=12,
                rotation=0,
                horizontalalignment="right",
                verticalalignment="top",
                bbox=dict(facecolor="k", alpha=0.5, edgecolor="none"),
            )

    # for dist, value in zip(dist2, values2): ax2.text( dist, value, f"{value:.1f}", color="red",
    #     fontsize=5, ha="left", va="bottom", ) ax2.set_xlabel("Distance from (x_offset, y_offset)
    # along the line")
    ax2.set_ylabel("Elevation from the lunar surface[°]", fontsize=14)
    # Set the y-label on the right side of the plot
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # At top left of the plot, display the sum_values1 and the theta ax2.text( 0.02, 0.98, f"Sum
    # counts: {sum_values1:.3f}\nθ={theta:.1f}°", transform=ax2.transAxes, fontsize=10,
    #     verticalalignment="top", bbox=dict(facecolor="k", alpha=0.5, edgecolor="none"), )
    #     ax2.minor_ticks_on(a)
    ax2.grid(axis="both", which="major", linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.grid(axis="both", which="minor", linestyle=":", linewidth=0.1, alpha=0.5)
    ax2.set_xlabel("Scaled x-ray counts", color="lime")
    ax2.tick_params(axis="x", labelcolor="lime")
    # Set the number of minor ticks
    ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

    # Set the number of major ticks
    ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines, labels, loc="upper left", fontsize=8)

    ax2.set_title(f"Line Profiles through ({x_offset}, {y_offset})")

    # delta time = 5 min
    # ax2.set_xlim(1.8, 2.6)
    # delta time = 45 min
    # ax2.set_xlim(1.8, 2.6)
    # ax2.set_xlim(0, 9.1)
    # ax2.set_ylim(0, 600)
    # ax2.set_ylim(0, 1)
    #  ax2.set_ylim(0.01, 0.05)
    ax2.set_yscale("linear")

    # ax2.set_ylim(0, 0.014) ax2.set_yscale("linear") Set the maximum number of ticks for both axes
    # to avoid clutter
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
    ax2.spines["bottom"].set_color("lime")
    # Set the tick labels color to match the line color on the left axis, for ax2
    ax2.tick_params(axis="x", colors="lime")

    # ax2.spines["bottom"].set_color("lime") Set the tick labels color to match the line color
    # ax2.tick_params(axis="y", colors="lime")
    ax2.spines["right"].set_color("white")
    plt.tight_layout()

    save_theta = np.round(theta, 1)
    save_theta = str(save_theta).zfill(6)
    start_date_str = start_date.replace(":", "").replace("-", "").replace("T", "_")
    end_date_str = end_date.replace(":", "").replace("-", "").replace("T", "_")

    # Set the title of the first plot
    ax1.set_title(
        f"Line Profile through ({x_offset:.2f}, {y_offset:.2f})\n"
        f"Time: {start_date_str[-7:-3]} to {end_date_str[-7:-3]}\n",
        fontsize=12,
    )
    delta_minutes = int(delta_time.total_seconds() / 60)
    save_folder = Path(f"../figures/line_profiles/el_az_{version_number}/{delta_minutes}min/")
    save_folder.mkdir(parents=True, exist_ok=True)
    # print(f"Start date : {start_date}, End date: {end_date}")
    fig_name = f"{start_date_str[-7:-3]}_{end_date_str[-7:-3]}_sunset_single_linear_line_profile_theta_{save_theta}_offset_{x_offset:0.3f}_{y_offset:0.3f}.png"
    # fig.savefig(save_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    # print(f"Line profile plot saved to {save_folder / fig_name}")

    # print(f"Line profile plot saved to {save_folder / fig_name}")
    # print(f"Line profile plot saved to {save_folder / fig_name}")

    # plt.close(fig)

    # Open a csv file and save the values of theta, sum_values1, perp_value and perp_dist, slope,
    # intercept and residuals Save the values to a csv file
    # Define the file path
    best_fit_file = Path(
        f"../data/line_profile_data/line_profile_best_fit_theta_offset_{x_offset:0.3f}_{y_offset:0.3f}_{delta_minutes}_minutes_{version_number}.csv"
    )
    best_fit_file.parent.mkdir(parents=True, exist_ok=True)

    # New data to be inserted
    new_data = {
        "start_time": start_time,
        "end_time": end_time,
        "theta": theta,
        "sum_values1": sum_values1,
        "slope": np.round(coeffs[0], 3),
        "intercept": np.round(coeffs[1], 3),
        "residuals": np.round(residuals[0], 3) if len(residuals) > 0 else np.nan,
        "min_value": np.round(np.nanmin(values2_selected), 3),
        "max_value": np.round(np.nanmax(values2_selected), 3),
    }
    new_row = pd.DataFrame([new_data])

    # Load existing data or create a new DataFrame
    if best_fit_file.exists():
        df = pd.read_csv(best_fit_file)
        # Ensure proper datetime conversion if necessary
        # df["start_time"] = pd.to_datetime(df["start_time"])
        # df["end_time"] = pd.to_datetime(df["end_time"])
    else:
        df = pd.DataFrame(columns=new_data.keys())

    # Drop existing row with the same start and end time if it exists
    mask = (df["start_time"] == start_time) & (df["end_time"] == end_time)
    df = df[~mask]

    # Append new data
    df = pd.concat([df, new_row], ignore_index=True)

    # Optional: sort by start_time and end_time
    df = df.sort_values(by=["start_time", "end_time"]).reset_index(drop=True)

    # Save back to file
    df.to_csv(best_fit_file, index=False)

    # Find the minimum and maximum values of min_value and max_value
    min_of_min_value = np.nanmin(df["min_value"])
    max_of_max_value = np.nanmax(df["max_value"])

    # Set the x-limit for the second plot based on the min and max values
    ax2.set_xlim(min_of_min_value, max_of_max_value)

    # Save the figure
    fig.savefig(save_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    # Close the figure to free up memory
    plt.close(fig)
    return theta, sum_values1, perp_value, perp_dist


global_start_time = "2025-03-16T19:00:00Z"
global_end_time = "2025-03-16T22:00:00Z"
delta_time = pd.Timedelta(minutes=400)  # 30 seconds
version_number = "v0.1"
start_date = pd.to_datetime(global_start_time)
end_date = pd.to_datetime(global_end_time)
# Create a list of times from start_date to end_date with a step of delta_time
time_list = pd.date_range(start=start_date, end=end_date, freq=delta_time)
# Convert the time_list to a list of strings in the format "YYYY-MM-DDTHH:MM:SSZ"
time_list_str = time_list.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()

for i, start_time in enumerate(time_list_str):
    print(
        f"Processing time interval {i + 1}/{len(time_list_str)}: {start_time} to {time_list_str[i + 1] if i + 1 < len(time_list_str) else global_end_time}",
        end="\r",
    )
    # for kkk in range(1):
    try:
        end_time = time_list_str[i + 1] if i + 1 < len(time_list_str) else global_end_time
        rot_angle = 0
        input_dict = {
            "x_key": "photon_az",
            "y_key": "photon_el",
            "start_time": start_time,
            "end_time": end_time,
            "bins": 200,
            "bin_range": [263, 281, 15, 33],
            "time_normalization": True,
            "rotate_data": False,
            "rotation_angle": rot_angle,
            "version_number": version_number,
        }

        read_data = True
        normalize_against_ground = True
        if (
            "hist" not in locals()
            or "xedges" not in locals()
            or "yedges" not in locals()
            or read_data
        ):
            org_hist, xedges, yedges, ra_median, dec_median = (
                lexi_functions.get_single_histogram_array_l1c_files(**input_dict)
            )
            # Save the histogram data to a pickle file along with start and end time
            # save_folder = Path("../data/")
            # save_folder.mkdir(parents=True, exist_ok=True)
            # file_name = f"line_profile_histogram_data_az_el_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}_xkey_{input_dict['x_key']}_ykey_{input_dict['y_key']}.pkl"
            # save_file = save_folder / file_name
            # with open(save_file, "wb") as f:
            #     pickle.dump(
            #         {
            #             "hist": org_hist,
            #             "xedges": xedges,
            #             "yedges": yedges,
            #             "ra_median": ra_median,
            #             "dec_median": dec_median,
            #             "start_time": input_dict["start_time"],
            #             "end_time": input_dict["end_time"],
            #             "bins": input_dict["bins"],
            #             "bin_range": input_dict["bin_range"],
            #             "time_normalization": input_dict["time_normalization"],
            #             "x_key": input_dict["x_key"],
            #             "y_key": input_dict["y_key"],
            #         },
            #         f,
            #     )

        n_shift_bin_x = 0
        n_shift_bin_y = 0
        # alpha_list = [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25]
        alpha_list = [1.0]

        if normalize_against_ground:
            alpha_list = alpha_list
        else:
            alpha_list = [1.0]
        # print("Histogram data loaded successfully.")
        for alpha in alpha_list:
            if normalize_against_ground:
                if input_dict["rotate_data"]:
                    ground_file_name = "../data/ground_histogram_data_20240523_224500Z_20240530_024500Z_rot_angle_0.pkl"
                else:
                    ground_file_name = "../data/ground_histogram_data_20250316_163000Z_20250316_213000Z_rot_angle_0_xkey_photon_az_ykey_photon_el_l1c.pkl"
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
                    ground_hist[n_shift_bin_y:, :] = (
                        np.nan
                    )  # This works since negative indices wrap correctly

                # Invalidate wrapped-around columns (x-direction)
                if n_shift_bin_x > 0:
                    ground_hist[:, :n_shift_bin_x] = np.nan
                elif n_shift_bin_x < 0:
                    ground_hist[:, n_shift_bin_x:] = np.nan

                ny, nx = ground_hist.shape
                y_orig = np.linspace(0, ny - 1, ny)
                x_orig = np.linspace(0, nx - 1, nx)

                # Create coordinate arrays for the scaled grid Center of scaling is assumed to be the center
                # of the image
                y_center, x_center = ny // 2, nx // 2
                y_scaled = (y_orig - y_center) / alpha + y_center
                x_scaled = (x_orig - x_center) / alpha + x_center

                # Create meshgrid for interpolation
                X, Y = np.meshgrid(x_scaled, y_scaled)
                coordinates = np.array([Y.ravel(), X.ravel()])

                # Interpolate the ground histogram
                ground_hist_scaled = map_coordinates(
                    ground_hist, coordinates, order=1, mode="constant", cval=np.nan
                )
                ground_hist_scaled = ground_hist_scaled.reshape(ground_hist.shape)

                # Normalize the histogram data against the scaled ground data
                hist = org_hist / ground_hist_scaled

                # Replace inf values with nan
                hist = np.where(np.isinf(hist), np.nan, hist)
            else:
                hist = org_hist
                # Replace inf values with nan
                hist = np.where(np.isinf(hist), np.nan, hist)

            # print("Histogram data normalized against ground data.")

            theta_list = np.linspace(0, 180, num=181, endpoint=True)
            theta_index = 90

            # Find the maximum value in the histogram and its corresponding coordinates
            max_index = np.unravel_index(np.argmax(hist, axis=None), hist.shape)
            # x_offset = (xedges[max_index[0]] + xedges[max_index[0] + 1]) / 2 y_offset =
            # (yedges[max_index[1]] + yedges[max_index[1] + 1]) / 2
            x_offset = 271  # X coordinate of the point the line must pass through
            y_offset = 24.8  # Y coordinate of the point the line must pass through

            # x_offset_list = np.linspace(-0.1, 0.1, num=10) y_offset_list = np.linspace(-0.1, 0.1, num=10)
            sum_values = []
            start_date = input_dict["start_time"]
            end_date = input_dict["end_time"]
            for theta in theta_list[theta_index : theta_index + 1]:
                # for theta in theta_list:
                # print(f"Processing line profile for theta = {theta:0.2f} °", end="\r")
                # Ensure the histogram is loaded
                if "hist" not in locals():
                    hist, xedges, yedges, ra_median, dec_median = (
                        lexi_functions.get_single_histogram_array(**input_dict)
                    )

                # Generate the line profile and plot it
                theta, sum_values1, perp_sum_value, perp_dist = plot_line_profile(
                    hist,
                    xedges,
                    yedges,
                    theta,
                    x_offset,
                    y_offset,
                    start_date,
                    end_date,
                    alpha,
                    normalize_against_ground,
                    rot_angle if input_dict["rotate_data"] else None,
                    delta_time,
                    version_number=version_number,
                )
                sum_values.append(sum_values1)

            # Save, theta, sum_values, x_offset, y_offset, hist, xedges, yedges, ra_median, dec_median to a
            # pickle file
            # save_folder = Path("../data/")
            # save_folder.mkdir(parents=True, exist_ok=True)
            # file_name = f"line_profile_data_{theta_list[0]}_{theta_list[-1]}_{len(theta_list)}.pkl"
            # save_file = save_folder / file_name
            # with open(save_file, "wb") as f:
            #     pickle.dump(
            #         {
            #             "theta_list": theta_list,
            #             "sum_values": sum_values,
            #             "perp_value": perp_sum_value,
            #             "perp_dist": perp_dist,
            #             "x_offset": x_offset,
            #             "y_offset": y_offset,
            #             "hist": hist,
            #             "xedges": xedges,
            #             "yedges": yedges,
            #             "ra_median": ra_median,
            #             "dec_median": dec_median,
            #         },
            #         f,
            #     )
    except Exception:
        pass
