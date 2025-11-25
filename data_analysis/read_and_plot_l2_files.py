import datetime
import glob
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from spacepy.pycdf import CDF as cdf


def keep_highest_versions(paths):
    best = {}

    for p in map(Path, paths):
        stem = p.stem
        try:
            base, vstr = stem.rsplit("_v", 1)
        except ValueError:
            # If no _V part, treat version as 0
            base, vstr = stem, "0"
        vtup = tuple(int(x) for x in vstr.split("."))  # e.g. (0, 1)

        if base not in best or vtup > best[base][0]:
            best[base] = (vtup, str(p))

    # Return in chronological/base order
    return [best[k][1] for k in sorted(best)]


def centers_to_corners_2d(ra_c, dec_c):
    """
    ra_c, dec_c: (H, W) center maps in degrees.
    Returns: RAcorn, DECcorn with shape (H+1, W+1) for pcolormesh.
    Uses midpoint edges and linear extrapolation at boundaries.
    """
    ra_c = np.asarray(ra_c, float)
    dec_c = np.asarray(dec_c, float)
    H, W = ra_c.shape

    # Midpoints between adjacent centers (internal edges)
    ra_i = 0.5 * (ra_c[1:, :] + ra_c[:-1, :])  # (H-1, W)
    ra_j = 0.5 * (ra_c[:, 1:] + ra_c[:, :-1])  # (H, W-1)
    dec_i = 0.5 * (dec_c[1:, :] + dec_c[:-1, :])
    dec_j = 0.5 * (dec_c[:, 1:] + dec_c[:, :-1])

    # Extrapolate outer edges along i (rows)
    ra_top = ra_c[0, :] - (ra_i[0, :] - ra_c[0, :])
    ra_bot = ra_c[-1, :] + (ra_c[-1, :] - ra_i[-1, :])
    dec_top = dec_c[0, :] - (dec_i[0, :] - dec_c[0, :])
    dec_bot = dec_c[-1, :] + (dec_c[-1, :] - dec_i[-1, :])

    # Extrapolate outer edges along j (cols)
    ra_left = ra_c[:, 0] - (ra_j[:, 0] - ra_c[:, 0])
    ra_right = ra_c[:, -1] + (ra_c[:, -1] - ra_j[:, -1])
    dec_left = dec_c[:, 0] - (dec_j[:, 0] - dec_c[:, 0])
    dec_right = dec_c[:, -1] + (dec_c[:, -1] - dec_j[:, -1])

    # Build (H+1, W+1) corners by averaging edges appropriately
    RAcorn = np.empty((H + 1, W + 1), float)
    DECcorn = np.empty((H + 1, W + 1), float)

    # Internal corners
    RAcorn[1:H, 1:W] = 0.25 * (ra_c[:-1, :-1] + ra_c[1:, :-1] + ra_c[:-1, 1:] + ra_c[1:, 1:])
    DECcorn[1:H, 1:W] = 0.25 * (dec_c[:-1, :-1] + dec_c[1:, :-1] + dec_c[:-1, 1:] + dec_c[1:, 1:])

    # Edges (average adjacent edge lines with neighbors)
    RAcorn[0, 1:W] = 0.5 * (ra_top[:-1] + ra_top[1:])
    RAcorn[-1, 1:W] = 0.5 * (ra_bot[:-1] + ra_bot[1:])
    RAcorn[1:H, 0] = 0.5 * (ra_left[:-1] + ra_left[1:])
    RAcorn[1:H, -1] = 0.5 * (ra_right[:-1] + ra_right[1:])

    DECcorn[0, 1:W] = 0.5 * (dec_top[:-1] + dec_top[1:])
    DECcorn[-1, 1:W] = 0.5 * (dec_bot[:-1] + dec_bot[1:])
    DECcorn[1:H, 0] = 0.5 * (dec_left[:-1] + dec_left[1:])
    DECcorn[1:H, -1] = 0.5 * (dec_right[:-1] + dec_right[1:])

    # Four corners (average of adjacent edges)
    RAcorn[0, 0] = 0.5 * (ra_top[0] + ra_left[0])
    RAcorn[0, -1] = 0.5 * (ra_top[-1] + ra_right[0])
    RAcorn[-1, 0] = 0.5 * (ra_bot[0] + ra_left[-1])
    RAcorn[-1, -1] = 0.5 * (ra_bot[-1] + ra_right[-1])

    DECcorn[0, 0] = 0.5 * (dec_top[0] + dec_left[0])
    DECcorn[0, -1] = 0.5 * (dec_top[-1] + dec_right[0])
    DECcorn[-1, 0] = 0.5 * (dec_bot[0] + dec_left[-1])
    DECcorn[-1, -1] = 0.5 * (dec_bot[-1] + dec_right[-1])

    return RAcorn, DECcorn


def plot_on_ra_dec(
    ax,
    ra_corners,
    dec_corners,
    data,
    title=None,
    cbar_title=None,
    time_range=None,
    vmin=None,
    vmax=None,
    norm="linear",
):
    if norm == "linear":
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == "log":
        if vmin is None:
            vmin = np.nanmin(data) if np.nanmin(data) > 0 else 1e-3 * np.nanmax(data)
        if vmax is None:
            vmax = np.nanmax(data)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    pm = ax.pcolormesh(
        ra_corners,
        dec_corners,
        data,
        shading="auto",
        cmap="plasma",
        norm=norm,
    )
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(title)

    spc_df = pd.read_csv(
        "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_pipeline/data/pointing/lexi_look_direction_data_resampled_interpolated_2025-03-02_00-00-00_to_2025-03-16_23-59-59_v0.0.csv"
    )
    spc_df["RA"] = spc_df["ra_lexi"]
    spc_df["DEC"] = spc_df["dec_lexi"]

    spc_df["Epoch"] = pd.to_datetime(spc_df["Epoch"], utc=True)
    spc_df.set_index("Epoch", inplace=True)
    time_range = pd.to_datetime(time_range, utc=True)
    spc_df = spc_df.loc[time_range[0] : time_range[1]]

    ra_center = spc_df["RA"].median()
    dec_center = spc_df["DEC"].median()
    ax.set_aspect("equal", adjustable="box")

    circle = plt.Circle((ra_center, dec_center), 4.55, color="white", fill=False)
    ax.plot(ra_center, dec_center, marker="o", color="k", markersize=5)
    ax.annotate(
        f"({ra_center:.2f}, {dec_center:.2f})",
        (ra_center + 0.1, dec_center + 0.1),
        color="white",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="black", alpha=0.5, pad=1),
    )
    ax.add_artist(circle)
    ax.set_aspect("equal")

    # Make the colorbar (no 'label=' here)
    cbar = plt.colorbar(pm, ax=ax, orientation="vertical", fraction=0.046, pad=0.00)

    # Force scientific notation with offset at the top
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())
    cbar.ax.yaxis.get_offset_text().set(size=14)
    cbar.ax.yaxis.get_offset_text().set_position((1.15, 1))  # move ×10^n to the top-right

    if cbar_title:
        cbar.ax.text(
            0.5,
            0.5,
            cbar_title,
            transform=cbar.ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.25, pad=2, edgecolor="none"),
        )

    # Set the x and y-axis limits
    ax.set_xlim(9, 22)
    ax.set_ylim(9, 22)
    return pm


def plot_on_az_el(
    ax,
    az_corners,
    el_corners,
    data,
    title=None,
    cbar_title=None,
    time_range=None,
    vmin=None,
    vmax=None,
    norm="linear",
):
    if norm == "linear":
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == "log":
        if vmin is None:
            vmin = np.nanmin(data) if np.nanmin(data) > 0 else 1e-3 * np.nanmax(data)
        if vmax is None:
            vmax = np.nanmax(data)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    pm = ax.pcolormesh(
        az_corners,
        el_corners,
        data,
        shading="auto",
        cmap="plasma",
        norm=norm,
    )
    ax.set_xlabel("Az [deg]")
    ax.set_ylabel("El [deg]")
    ax.set_title(title)
    time_range = pd.to_datetime(time_range, utc=True)

    ax.set_aspect("equal", adjustable="box")

    # Make the colorbar (no 'label=' here)
    cbar = plt.colorbar(pm, ax=ax, orientation="vertical", fraction=0.046, pad=0.00)
    # Force scientific notation with offset at the top
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())
    cbar.ax.yaxis.get_offset_text().set(size=14)
    cbar.ax.yaxis.get_offset_text().set_position((1.15, 1))  # move ×10^n to the top-right

    if cbar_title:
        cbar.ax.text(
            0.5,
            0.5,
            cbar_title,
            transform=cbar.ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.25, pad=2, edgecolor="none"),
        )

    # Set the x and y-axis limits
    ax.set_xlim(264.75, 275)
    ax.set_ylim(20, 29.5)
    return pm


def overlay_ra_dec_contours(ax, ra_c, dec_c, n_ra=7, n_dec=7, **kw):
    ra_levels = np.linspace(np.nanmin(ra_c), np.nanmax(ra_c), n_ra)
    dec_levels = np.linspace(np.nanmin(dec_c), np.nanmax(dec_c), n_dec)
    ax.contour(
        ra_c, levels=ra_levels, colors="k", linewidths=0.5, alpha=0.4, extent=None
    )  # drawn in pixel coords for quick look
    ax.contour(dec_c, levels=dec_levels, colors="w", linewidths=0.5, alpha=0.4, extent=None)


def overlay_az_el_contours(ax, az_c, el_c, n_az=7, n_el=7, **kw):
    az_levels = np.linspace(np.nanmin(az_c), np.nanmax(az_c), n_az)
    el_levels = np.linspace(np.nanmin(el_c), np.nanmax(el_c), n_el)
    ax.contour(
        az_c, levels=az_levels, colors="k", linewidths=0.5, alpha=0.4, extent=None
    )  # drawn in pixel coords for quick look
    ax.contour(el_c, levels=el_levels, colors="w", linewidths=0.5, alpha=0.4, extent=None)


def elevation_profile_normalized(ELcorn: np.ndarray, H: np.ndarray):
    """
    Compute normalized azimuth-summed profile vs elevation.

    Parameters
    ----------
    ELcorn : (M+1, N+1) array
        Elevation bin *corners*.
    H : (M, N) array
        Histogram values per cell; may contain NaNs.

    Returns
    -------
    elev_row_mean : (M,) array
        Representative elevation for each row (degrees).
    norm_profile : (M,) array
        Row-wise sum over azimuth normalized by count of non-NaN cells.
    row_sum : (M,) array
        Row-wise sum over azimuth (ignoring NaNs).
    row_counts : (M,) array
        Number of non-NaN cells per row.
    """
    H = np.asarray(H, dtype=float)
    ELcorn = np.asarray(ELcorn, dtype=float)

    # Cell-centered elevations from corners
    ELc = 0.25 * (ELcorn[:-1, :-1] + ELcorn[1:, :-1] + ELcorn[:-1, 1:] + ELcorn[1:, 1:])

    mask = ~np.isnan(H)
    # Row-wise representative elevation, masking where H is NaN
    with np.errstate(invalid="ignore"):
        elev_row_mean = np.nanmean(np.where(mask, ELc, np.nan), axis=1)

    # Row-wise sum and counts over azimuth
    row_sum = np.nansum(H, axis=1)
    row_counts = np.sum(mask, axis=1)

    # Normalized profile
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_profile = np.where(row_counts > 0, row_sum / row_counts, np.nan)

    # Get the total counts in the histogram (ignoring NaNs)
    total_counts = np.nansum(H)
    return elev_row_mean, norm_profile, row_sum, row_counts, total_counts


def plot_line_profile(
    data_df,
    ax,
    input_histogram,
    azimuth_corners,
    elevation_corners,
    title=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    logy=False,
    key=None,
):
    """
    Starting from the bottom of the plot and going up, at each y-bin, find the sum of histogram
    values along the x-axis and save it as the y-value for that bin.
    """
    # Set anything less than zero to NaN
    input_histogram = np.where(input_histogram <= 0, np.nan, input_histogram)
    elev_row_mean, norm_profile, row_sum, row_counts, total_counts = elevation_profile_normalized(
        elevation_corners, input_histogram
    )
    # print(f"key is {key} and total counts is {total_counts}")
    # print(f"Row counts (non-NaN cells): {row_counts}\n Row sums: {row_sum}\n\n")
    el_centers = elev_row_mean
    profile = norm_profile
    if profile is None or np.all(np.isnan(profile)):
        # print("Profile is empty or all NaNs, skipping plot.")
        return ax, data_df

    # Get the best fit line (ignoring NaNs)
    valid = ~np.isnan(el_centers) & ~np.isnan(profile)
    # Do not use first 3 and last 3 valid points for the fit
    valid_indices = np.where(valid)[0]
    if len(valid_indices) > 6:
        valid[valid_indices[:3]] = False
        valid[valid_indices[-3:]] = False
    if np.sum(valid) >= 2:
        coeffs = np.polyfit(el_centers[valid], profile[valid], deg=1)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(np.nanmin(el_centers), np.nanmax(el_centers), 100)
        y_fit = poly(x_fit)
        # Get the residuals and R^2 value
        residuals = profile[valid] - poly(el_centers[valid])
        ss_res = np.nansum(residuals**2)
        ss_tot = np.nansum((profile[valid] - np.nanmean(profile[valid])) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        ax.plot(y_fit, x_fit, color="cyan", linestyle="--", label="Best fit line")
        # ax.legend()
        # Display the equation on the plot
        ax.text(
            0.01,
            0.01,
            f"y = {coeffs[0]:.3e}x + {coeffs[1]:.3e}\nR² = {r_squared:.3e}, resid = {ss_tot:.3e}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="k", alpha=0.5),
        )
    # Add the total counts to the bottom right corner
    ax.text(
        0.99,
        0.99,
        f"Total Counts = {total_counts:.2e}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="k", alpha=0.5),
    )
    ax.scatter(profile, el_centers, color="magenta", s=10, marker=".")
    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else "Counts")
    ax.set_ylabel(ylabel if ylabel else "Elevation [deg]")
    # Make vertical orange line at the minimum and maximum profile values
    ax.axvline(x=np.nanmin(profile), color="orange", linestyle="--", linewidth=1.5)
    ax.axvline(x=np.nanmax(profile), color="orange", linestyle="--", linewidth=1.5)
    if xlim:
        ax.set_xlim(xlim)
        ax.set_xscale("linear")
    else:
        ax.set_xlim(left=np.nanmin(profile) * 0.9)
        ax.set_xlim(right=np.nanmax(profile) * 1.1)
    if ylim:
        ax.set_ylim(ylim)
    if logy:
        ax.set_yscale("log")
    # ax.set_xscale("log")

    # Update the data_df with new columns (if the columns don't already exist)
    if key:
        if f"{key}_residuals" in data_df.columns:
            data_df.at[len(data_df) - 1, f"{key}_residuals"] = (
                ss_res if np.sum(valid) >= 2 else np.nan
            )
        if f"{key}_r_squared" in data_df.columns:
            data_df.at[len(data_df) - 1, f"{key}_r_squared"] = (
                r_squared if np.sum(valid) >= 2 else np.nan
            )
        if f"{key}_slope" in data_df.columns:
            data_df.at[len(data_df) - 1, f"{key}_slope"] = (
                coeffs[0] if np.sum(valid) >= 2 else np.nan
            )
        if f"{key}_intercept" in data_df.columns:
            data_df.at[len(data_df) - 1, f"{key}_intercept"] = (
                coeffs[1] if np.sum(valid) >= 2 else np.nan
            )
        if f"{key}_total_hist_counts" in data_df.columns:
            data_df.at[len(data_df) - 1, f"{key}_total_hist_counts"] = total_counts
    return ax, data_df


warnings.filterwarnings("ignore")
keys_to_plot = [
    "lexi_image",
    "lexi_image_bgnd_corrected",
]


# Define an empty dataframe
data_df = pd.DataFrame()
keys_to_add = [
    "start_time",
    "end_time",
    "raw_counts_total_hist_counts",
    "raw_counts_residuals",
    "raw_counts_r_squared",
    "raw_counts_slope",
    "raw_counts_intercept",
    "background_corrected_total_hist_counts",
    "background_corrected_residuals",
    "background_corrected_r_squared",
    "background_corrected_slope",
    "background_corrected_intercept",
]
for key in keys_to_add:
    data_df[key] = pd.Series(
        dtype=(
            "float64"
            if "residuals" in key or "r_squared" in key or "slope" in key or "intercept" in key
            else "datetime64[ns, UTC]"
        )
    )

# Example
time_res = "5min"
all_l2_files = sorted(
    glob.glob(f"/mnt/cephadrius/bu_research/lexi_data/l2/{time_res}/clps-bgm1_lexi_l2-images*.cdf")
)
l2_files = keep_highest_versions(all_l2_files)

norm_lexi = "log"
v_min_lexi = 1e-4
v_max_lexi = 5e-2

line_x_min = 10
line_x_max = 30

for i, f in enumerate(l2_files[:]):
    dat = cdf(f)

    print(f"Reading file: {f}, {i + 1} out of {len(l2_files)}", end="\r")
    ra_c = np.asarray(dat["ra_bin_map"][...])[0]
    dec_c = np.asarray(dat["dec_bin_map"][...])[0]
    az_c = np.asarray(dat["az_bin_map"][...])[0]
    el_c = np.asarray(dat["el_bin_map"][...])[0]
    time_range = [dat["epoch_start"][...][0], dat["epoch_end"][...][0]]

    RAcorn, DECcorn = centers_to_corners_2d(ra_c, dec_c)
    AZcorn, ELcorn = centers_to_corners_2d(az_c, el_c)

    # Update the data_df with start_time and end_time columns
    t0 = pd.to_datetime(dat["epoch_start"][...][0], utc=True)
    t1 = pd.to_datetime(dat["epoch_end"][...][0], utc=True)
    data_df.loc[len(data_df)] = {"start_time": t0, "end_time": t1}

    # Set the plot theme to dark
    plt.style.use("dark_background")
    # Plot the exposure maps and counts
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    # Set the hspace and wspace
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    # Set the default font size
    mpl.rcParams.update({"font.size": 16})
    file_name = Path(f).name.split("_")[-2]
    hh = int(file_name[8:10])
    mm = int(file_name[10:12])
    fig.suptitle(f"LEXI L2 Data from {hh:02d}:{mm:02d} for {time_res}", fontsize=20)

    plot_on_az_el(
        axs[0, 0],
        AZcorn,
        ELcorn,
        np.asarray(dat["lexi_image"][...])[0]
        / np.asarray(dat["pixel_area"][...])[0]
        * np.asarray(dat["exposure_map"][...])[0]
        / np.asarray(dat["exposure_map"][...])[0],
        time_range=time_range,
        title="Raw Counts",
        cbar_title="Counts/s/arcmin$^2$",
        norm=norm_lexi,
        vmin=v_min_lexi,
        vmax=v_max_lexi,
    )
    plot_on_az_el(
        axs[0, 1],
        AZcorn,
        ELcorn,
        np.asarray(dat["lexi_image_background_corrected"][...])[0]
        / np.asarray(dat["pixel_area"][...])[0]
        * np.asarray(dat["exposure_map"][...])[0]
        / np.asarray(dat["exposure_map"][...])[0],
        time_range=time_range,
        title="Background-Corrected Counts",
        cbar_title="Counts/s/arcmin$^2$",
        norm=norm_lexi,
        vmin=v_min_lexi,
        vmax=v_max_lexi,
    )
    x_lim = (line_x_min, line_x_max)
    # Plot the line profiles on the bottom row
    _, data_df = plot_line_profile(
        data_df,
        axs[1, 0],
        np.asarray(dat["lexi_image"][...])[0]
        / np.asarray(dat["pixel_area"][...])[0]
        * np.asarray(dat["exposure_map"][...])[0]
        / np.asarray(dat["exposure_map"][...])[0],
        AZcorn,
        ELcorn,
        title="Line Profile",
        xlabel="Counts/s/arcmin$^2$",
        # xlim=x_lim,
        ylim=(20, 29.5),
        key="raw_counts",
    )
    _, data_df = plot_line_profile(
        data_df,
        axs[1, 1],
        np.asarray(dat["lexi_image_background_corrected"][...])[0]
        / np.asarray(dat["pixel_area"][...])[0]
        * np.asarray(dat["exposure_map"][...])[0]
        / np.asarray(dat["exposure_map"][...])[0],
        AZcorn,
        ELcorn,
        title="Line Profile",
        xlabel="Counts/s/arcmin$^2$",
        # xlim=x_lim,
        ylim=(20, 29.5),
        key="background_corrected",
    )
    for ax in axs.flatten()[:2]:
        ax.contour(AZcorn[:-1, :-1], ELcorn[:-1, :-1], az_c, colors="c", linewidths=0.6, alpha=0.5)
        ax.contour(AZcorn[:-1, :-1], ELcorn[:-1, :-1], el_c, colors="k", linewidths=0.6, alpha=0.5)
        ax.set_aspect("equal")
        # Modify the fontsize of tick labels
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=12)
        ax.xaxis.get_offset_text().set(size=12)
        ax.yaxis.get_offset_text().set(size=12)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    for ax in axs.flatten()[2:]:
        # ax.contour(AZcorn[:-1, :-1], ELcorn[:-1, :-1], az_c, colors="c", linewidths=0.6, alpha=0.5)
        # ax.contour(AZcorn[:-1, :-1], ELcorn[:-1, :-1], el_c, colors="k", linewidths=0.6, alpha=0.5)
        # ax.set_aspect("equal")
        # Modify the fontsize of tick labels
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=12)
        ax.xaxis.get_offset_text().set(size=12)
        ax.yaxis.get_offset_text().set(size=12)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.set_xscale("linear")

    figure_path = Path(f"../figures/line_profiles/bg_corrected/from_l2/{time_res}/az_el/")
    figure_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figure_path / (Path(f).stem + f"_exposure_accounted_az_el_{time_res}.png"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # print("Saved figure:", figure_path / (Path(f).stem + "_exposure_and_counts_az_el.png"))
    plt.close(fig)
    # plt.show()


# Save the dataframe to a CSV file
data_folder = Path(f"../data/line_profile_data/bg_corrected/from_l2/{time_res}/")
data_folder.mkdir(parents=True, exist_ok=True)
data_df.to_csv(
    data_folder / f"line_profile_fit_parameters_bg_corrected_no_flat_field_{time_res}.csv",
    index=False,
)
