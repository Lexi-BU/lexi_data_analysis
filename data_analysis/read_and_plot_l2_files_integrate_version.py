import datetime
import glob
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from spacepy.pycdf import CDF as cdf

# --------------------------------------------
# Config
# --------------------------------------------
plt.style.use("default")
mpl.rcParams.update({"font.size": 22})
# Set the fonttype to 42 to avoid Type 3 fonts in the output PDF/PNGs
mpl.rcParams["pdf.fonttype"] = 42
# Set the the font to be arial-like for better readability
mpl.rcParams["font.family"] = "Arial"


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


def read_l2_metadata_and_paths(l2_files):
    """
    Return a DataFrame with file path and its [t0, t1] in UTC.
    """
    rows = []
    for f in l2_files:
        with cdf(f) as dat:
            t0 = pd.to_datetime(dat["epoch_start"][...][0], utc=True)
            t1 = pd.to_datetime(dat["epoch_end"][...][0], utc=True)
        rows.append({"path": f, "t0": t0, "t1": t1})
    df = pd.DataFrame(rows).sort_values("t0").reset_index(drop=True)
    return df


def load_lexi_arrays_one_file(fpath):
    """
    Read a single L2 CDF and return AZ/EL corners and exposure-weighted images:
      - lexi_raw (raw * exposure_map)
      - lexi_bg (bg-corrected * exposure_map)
      - lexi_bgff (bg+flatfield corrected * exposure_map)
    """
    dat = cdf(fpath)
    az_c = np.asarray(dat["az_bin_map"][...])[0]
    el_c = np.asarray(dat["el_bin_map"][...])[0]
    AZcorn, ELcorn = centers_to_corners_2d(az_c, el_c)

    exp = np.asarray(dat["exposure_map"][...])[0]
    lexi_raw = np.asarray(dat["lexi_image"][...])[0] * exp
    lexi_bg = np.asarray(dat["lexi_image_background_corrected"][...])[0] * exp
    lexi_bgff = np.asarray(dat["lexi_image_background_flatfield_corrected"][...])[0] * exp

    # also return az_c, el_c for contour overlays
    return AZcorn, ELcorn, az_c, el_c, lexi_raw, lexi_bg, lexi_bgff


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
        edgecolor="face",
        norm=norm,
    )
    ax.set_xlabel("Az [deg]")
    ax.set_ylabel("El [deg]")
    ax.set_title(title)
    time_range = pd.to_datetime(time_range, utc=True)

    ax.set_aspect("equal", adjustable="box")

    # ---- Custom colorbar tick formatting ----
    cbar = plt.colorbar(pm, ax=ax, orientation="vertical", fraction=0.046, pad=-0.075)

    # Force scientific notation with shared exponent
    # Get the colorbar limits
    if vmin is None or vmax is None:
        vmin, vmax = pm.get_clim()
    else:
        vmin, vmax = vmin, vmax

    # Calculate the order of magnitude
    magnitude = int(np.floor(np.log10(max(abs(vmin), abs(vmax)))))
    scale_factor = 10**magnitude

    # Create custom tick formatter that shows values divided by scale factor
    def format_func(x, pos):
        return f"{x / scale_factor:.1f}"

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    # Add the offset text at the top
    offset_label = rf"$\times 10^{{{magnitude}}}$"
    # cbar.ax.text(
    #     0.5,
    #     1.03,
    #     offset_label,
    #     transform=cbar.ax.transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=0.8 * mpl.rcParams["font.size"],
    # )

    # Add offset_label to cbar title if provided
    if cbar_title:
        cbar_title = f"{cbar_title} [{offset_label}]"
        cbar.ax.text(
            0.5,
            0.5,
            cbar_title,
            transform=cbar.ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            color="white",
            fontsize=0.7 * mpl.rcParams["font.size"],
            fontweight="bold",
            # bbox=dict(facecolor="black", alpha=0.25, pad=2, edgecolor="none"),
        )

    # Set the x and y-axis limits
    ax.set_xlim(264.75, 275)
    ax.set_ylim(20, 29.5)
    return pm


def plot_selected_integrated_hist(
    ax,
    AZcorn,
    ELcorn,
    hist_choice,
    sums_dict,
    *,
    time_range=None,
    norm=None,
    vmin=None,
    vmax=None,
    cbar_title="Counts",
    title_map=None,
):
    """
    Plot one of the integrated histograms (sum_raw / sum_bg / sum_bgff) using the
    exact same style as plot_on_az_el, including AZ/EL corners.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    AZcorn, ELcorn : 2D arrays
        Azimuth/Elevation bin *corners* (from centers_to_corners_2d).
    hist_choice : str
        One of {"sum_raw", "sum_bg", "sum_bgff"}.
    sums_dict : dict
        {"sum_raw": <2D array>, "sum_bg": <2D array>, "sum_bgff": <2D array>}
    time_range : [pd.Timestamp, pd.Timestamp], optional
        [start, end] UTC for annotation (passed to plot_on_az_el).
    norm, vmin, vmax : optional
        Colormap scaling (passed to plot_on_az_el).
    cbar_title : str
        Colorbar label. For integrated windows, “Counts” is typical.
    title_map : dict, optional
        Optional mapping from hist_choice to title string.
    """
    if hist_choice not in sums_dict:
        raise ValueError(
            f"hist_choice must be one of {list(sums_dict.keys())}, got {hist_choice!r}."
        )

    data = sums_dict[hist_choice]

    # Default titles if none provided
    if title_map is None:
        title_map = {
            "sum_raw": "Raw Counts (integrated)",
            "sum_bg": "Background-Corrected Counts (integrated)",
            "sum_bgff": "Bg + Flat-Field Corrected Counts (integrated)",
        }

    title = title_map.get(hist_choice, f"{hist_choice} (integrated)")

    # Delegate to your existing styling function
    return plot_on_az_el(
        ax,
        AZcorn,
        ELcorn,
        data,
        time_range=time_range,
        title=title,
        cbar_title=cbar_title,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )


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
    integration_seconds=None,
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
    el_centers = elev_row_mean
    profile = norm_profile
    if profile is None or np.all(np.isnan(profile)):
        profile_col = f"profile_{key}" if key else "profile"
        el_col = f"el_centers_{key}" if key else "el_centers"
        return ax, data_df, pd.DataFrame(columns=[el_col, profile_col])

    # Get the best fit line (ignoring NaNs)
    valid = ~np.isnan(el_centers) & ~np.isnan(profile)
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
        ax.plot(y_fit, x_fit, color="k", linestyle="--", label="Best fit line")
        # Display the equation on the plot
        ax.text(
            0.01,
            0.01,
            f"y = {coeffs[0]:.3e}x + {coeffs[1]:.3e}",
            # \nR² = {r_squared:.3e}, resid = {ss_tot:.3e}",
            transform=ax.transAxes,
            fontsize=0.8 * mpl.rcParams["font.size"],
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="w", alpha=0.5),
        )
    # Add the total counts to the bottom right corner
    ax.text(
        0.99,
        0.99,
        f"Total Counts = {total_counts * integration_seconds:.2e}",
        transform=ax.transAxes,
        fontsize=0.8 * mpl.rcParams["font.size"],
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="w", alpha=0.5),
    )
    ax.scatter(profile, el_centers, color="magenta", s=10, marker=".")
    ax.set_title(title)
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

    # Set the tick label font size
    ax.tick_params(axis="both", which="both", labelsize=1.8 * mpl.rcParams["font.size"])

    # Apply same tick formatting style as colorbar (scientific notation with shared exponent)
    x_min, x_max = ax.get_xlim()
    magnitude = int(np.floor(np.log10(max(abs(x_min), abs(x_max)))))
    scale_factor = 10**magnitude

    def format_func(x, pos):
        return f"{x / scale_factor:.1f}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    xlabel = xlabel + rf"[$\times 10^{{{magnitude}}}$]"
    ax.set_xlabel(xlabel, fontsize=1.2 * mpl.rcParams["font.size"])

    ax.set_ylabel(ylabel if ylabel else "Elevation [deg]", fontsize=1.2 * mpl.rcParams["font.size"])

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
            
    profile_col = f"counts_{key}" if key else "profile"
    el_col = f"el_centers_{key}" if key else "el_centers"
    
    profile_df = pd.DataFrame({
        profile_col: profile,
        el_col: el_centers
    })
    
    return ax, data_df, profile_df


def plot_selected_integrated_hist_with_marginals(
    AZcorn,
    ELcorn,
    data,
    *,
    title=None,
    cmap="mako",
    log_color=False,
    cbar_label="Counts",
    height=8,
    ratio=2,
    linewidth=1.5,
):
    """
    Seaborn-based joint plot of an integrated 2D histogram (e.g., sum_raw/sum_bg/sum_bgff)
    with marginal distributions on the top (Azimuth) and right (Elevation).

    Parameters
    ----------
    AZcorn, ELcorn : 2D arrays (corners mesh)
        Output from centers_to_corners_2d(...) (same grids used by plot_on_az_el).
    data : 2D array
        Integrated counts array on the same (ny, nx) grid as the corners.
    title : str, optional
        Figure title.
    cmap : str
        Colormap name for the joint image.
    log_color : bool
        If True, apply LogNorm() to color mapping.
    cbar_label : str
        Colorbar label text.
    height : float
        Height of the JointGrid in inches.
    ratio : int
        Joint-to-marginal area ratio (higher => larger center panel).
    linewidth : float
        Line width for marginal plots.
    """
    # --- derive centers from corners (works even if you don't have 1D centers handy) ---
    # X (az) centers are midpoints along x-direction of the top row of AZcorn
    az_centers = 0.5 * (AZcorn[0, 1:] + AZcorn[0, :-1])  # shape (nx,)
    # Y (el) centers are midpoints along y-direction of the left column of ELcorn
    el_centers = 0.5 * (ELcorn[1:, 0] + ELcorn[:-1, 0])  # shape (ny,)

    ny, nx = data.shape
    assert nx == az_centers.size and ny == el_centers.size, "Data and corner grid sizes mismatch."

    # --- build marginals (sums of counts along axes) ---
    az_marg = np.nansum(data, axis=0)  # counts per az bin
    el_marg = np.nansum(data, axis=1)  # counts per el bin

    # --- JointGrid layout ---
    sns.set_theme(style="whitegrid")
    g = sns.JointGrid(marginal_ticks=True, height=height, ratio=ratio)

    # --- center: use imshow so we can respect physical coordinates with extent ---
    # extent = [x_min, x_max, y_min, y_max]
    az_min, az_max = AZcorn[0, 0], AZcorn[0, -1]
    el_min, el_max = ELcorn[0, 0], ELcorn[-1, 0]

    norm = LogNorm() if log_color else None
    im = g.ax_joint.imshow(
        data,
        origin="lower",
        extent=[az_min, az_max, el_min, el_max],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # colorbar on the joint axis
    cbar = g.figure.colorbar(im, ax=g.ax_joint, pad=0.02)
    cbar.set_label(cbar_label)

    # labels & title
    g.ax_joint.set_xlabel("Azimuth [deg]")
    g.ax_joint.set_ylabel("Elevation [deg]")
    if title:
        g.ax_joint.set_title(title, pad=10)

    # --- marginals ---
    # top marginal: counts vs az
    sns.scatterplot(x=az_centers, y=az_marg, ax=g.ax_marg_x, linewidth=linewidth)
    g.ax_marg_x.set_ylabel("Sum counts")
    g.ax_marg_x.grid(True, alpha=0.3)

    # right marginal: counts vs el (horizontal line plot)
    sns.scatterplot(y=el_centers, x=el_marg, ax=g.ax_marg_y, linewidth=linewidth)
    g.ax_marg_y.set_xlabel("Sum counts")
    g.ax_marg_y.grid(True, alpha=0.3)

    # tidy up tick label sizes a bit
    for ax in (g.ax_joint, g.ax_marg_x, g.ax_marg_y):
        ax.tick_params(labelsize=0.8 * mpl.rcParams["font.size"])

    g.fig.tight_layout()
    return g


def plot_integrated_map_with_marginals_seaborn(
    AZcorn,
    ELcorn,
    data,
    *,
    figsize=(8, 6),  # <-- set to EXACT width/height of your original figure
    title=None,
    cmap="mako",
    log_color=False,
    cbar_label="Counts",
    joint_ratio=4,  # joint : marginal size ratio (keeps proportions, not total size)
    scatter_size=10,  # marginal scatter size
    normalize_marginals=True,  # <-- normalize by number of non-NaN bins
    cbar_lims=(None, None),
):
    """
    Seaborn-styled figure with:
      - center: 2D image of 'data' in AZ/EL with true coordinate extents
      - top:  scatter of marginal vs azimuth (sum or mean over elevation)
      - right: scatter of marginal vs elevation (sum or mean over azimuth)

    EXACT figure width/height is controlled by 'figsize' (in inches).
    """
    sns.set_theme(style="whitegrid")

    # ---- derive centers from corners ----
    az_centers = 0.5 * (AZcorn[0, 1:] + AZcorn[0, :-1])  # (nx,)
    el_centers = 0.5 * (ELcorn[1:, 0] + ELcorn[:-1, 0])  # (ny,)
    ny, nx = data.shape
    assert nx == az_centers.size and ny == el_centers.size, "Data/corners size mismatch."

    # ---- build marginals (sum along axes) ----
    with np.errstate(invalid="ignore"):
        az_sum = np.nansum(data, axis=0)  # counts per az bin
        el_sum = np.nansum(data, axis=1)  # counts per el bin

        # non-NaN bin counts along the reduction axes
        az_counts = np.sum(
            np.isfinite(data), axis=0
        )  # how many valid EL bins contribute to each AZ column
        el_counts = np.sum(
            np.isfinite(data), axis=1
        )  # how many valid AZ bins contribute to each EL row
        if normalize_marginals:
            # mean count per non-NaN bin
            az_marg = az_sum / az_counts
            el_marg = el_sum / el_counts
            ylab_top = "Sum counts"
            xlab_right = "Sum counts"
        else:
            az_marg = az_sum
            el_marg = el_sum
            ylab_top = "Sum counts"
            xlab_right = "Sum counts"

    # ---- construct figure with exact size ----
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        height_ratios=[1, joint_ratio],
        width_ratios=[joint_ratio, 1],
        left=0.08,
        right=0.95,
        bottom=0.08,
        top=0.92,
        hspace=0.01,
        wspace=-0.08,  # Reduced to tighten space between ax_right and colorbar
    )

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    # ---- Force same width between ax_top and ax_joint ----
    pos_joint = ax_joint.get_position()
    pos_top = ax_top.get_position()
    ax_top.set_position([pos_joint.x0, pos_top.y0, pos_joint.width * 0.85, pos_top.height])

    # ---- center image with true extents ----
    az_min, az_max = AZcorn[0, 0], AZcorn[0, -1]
    el_min, el_max = ELcorn[0, 0], ELcorn[-1, 0]
    norm = LogNorm() if log_color else None

    im = ax_joint.imshow(
        data,
        origin="lower",
        extent=[az_min, az_max, el_min, el_max],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # ---- Colorbar ----
    cbar = fig.colorbar(im, ax=ax_joint, pad=0.0)
    if cbar_lims != (None, None):
        im.set_clim(vmin=cbar_lims[0], vmax=cbar_lims[1])

    # Force scientific notation with shared exponent
    # Get the colorbar limits
    vmin, vmax = cbar_lims

    # Calculate the order of magnitude
    magnitude = int(np.floor(np.log10(max(abs(vmin), abs(vmax)))))
    scale_factor = 10**magnitude

    # Create custom tick formatter that shows values divided by scale factor
    def format_func(x, pos):
        return f"{x / scale_factor:.1f}"

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    # Set the font size of colorbar tick labels
    cbar.ax.tick_params(labelsize=1.5 * mpl.rcParams["font.size"])

    # Add the offset text at the top
    offset_label = rf"$\times 10^{{{magnitude}}}$"
    # cbar.ax.text(
    #     0.5,
    #     1.03,
    #     offset_label,
    #     transform=cbar.ax.transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=0.8 * mpl.rcParams["font.size"],
    # )

    # Add offset_label to cbar title if provided
    if cbar_label:
        cbar_label = f"{cbar_label} [{offset_label}]"
        cbar.ax.text(
            0.5,
            0.5,
            cbar_label,
            transform=cbar.ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            color="white",
            fontsize=1.2 * mpl.rcParams["font.size"],
            fontweight="bold",
            # bbox=dict(facecolor="black", alpha=0.25, pad=2, edgecolor="none"),
        )
    # Ensure the (now repositioned) offset text is visible
    cbar.ax.yaxis.offsetText.set_visible(True)
    # ---------------------------------

    ax_joint.set_xlabel("Azimuth [deg]", fontsize=1.8 * mpl.rcParams["font.size"])
    ax_joint.set_ylabel("Elevation [deg]", fontsize=1.8 * mpl.rcParams["font.size"])

    if title:
        ax_joint.set_title(title, pad=8)

    # ---- marginals (scatter) ----
    # top: az_marg vs az_centers
    sns.scatterplot(x=az_centers, y=az_marg, s=scatter_size, ax=ax_top)

    ax_top.grid(True, alpha=0.3)
    # Set the y-axis limits to match the joint plot
    ax_top.set_ylim(0.008, 0.02)
    # Explicitly set x-axis limits to match ax_joint
    ax_top.set_xlim(az_min, az_max)
    plt.setp(ax_top.get_xticklabels(), visible=False)  # sharex with joint; hide x labels on top

    # right: el_marg vs el_centers (horizontal scatter)
    sns.scatterplot(y=el_centers, x=el_marg, s=scatter_size, ax=ax_right)
    ax_right.grid(True, alpha=0.3)
    # Set the x-axis limits to match the joint plot
    ax_right.set_xlim(0.005, 0.017)
    # plt.setp(ax_right.get_yticklabels(), visible=False)  # sharey with joint; hide y labels on right

    # tidy ticks
    # ax_top.tick_params(labelsize=0.9 * mpl.rcParams["font.size"])
    # ax_right.tick_params(labelsize=0.9 * mpl.rcParams["font.size"])
    ax_joint.tick_params(labelsize=1.8 * mpl.rcParams["font.size"])

    # Hide all the spines of ax_top and ax_right
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    for spine in ax_right.spines.values():
        spine.set_visible(False)
    # Hide the ticks and labels for top and right axes
    ax_top.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
    )
    ax_right.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
    )

    # Apply same tick formatting style as colorbar (scientific notation with shared exponent)
    # For ax_top (y-axis)
    # y_min_top, y_max_top = ax_top.get_ylim()
    # magnitude_top = int(np.floor(np.log10(max(abs(y_min_top), abs(y_max_top)))))
    # scale_factor_top = 10**magnitude_top

    # def format_func_top(x, pos):
    #     return f"{x / scale_factor_top:.1f}"

    # ax_top.yaxis.set_major_formatter(FuncFormatter(format_func_top))
    # ylab_top = ylab_top + rf"[$\times 10^{{{magnitude_top}}}$]"
    # ax_top.set_ylabel(ylab_top, fontsize=0.7 * mpl.rcParams["font.size"])

    # # For ax_right (x-axis)
    # x_min_right, x_max_right = ax_right.get_xlim()
    # magnitude_right = int(np.floor(np.log10(max(abs(x_min_right), abs(x_max_right)))))
    # scale_factor_right = 10**magnitude_right

    # def format_func_right(x, pos):
    #     return f"{x / scale_factor_right:.1f}"

    # ax_right.xaxis.set_major_formatter(FuncFormatter(format_func_right))

    # xlab_right = xlab_right + rf"[$\times 10^{{{magnitude_right}}}$]"
    # ax_right.set_xlabel(xlab_right, fontsize=0.7 * mpl.rcParams["font.size"])

    return fig, (ax_joint, ax_top, ax_right)


warnings.filterwarnings("ignore")
keys_to_plot = [
    "lexi_image",
    "lexi_image_bgnd_corrected",
    "lexi_image_bgnd_flat_corrected",
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
    "background_flatfield_corrected_total_hist_counts",
    "background_flatfield_corrected_residuals",
    "background_flatfield_corrected_r_squared",
    "background_flatfield_corrected_slope",
    "background_flatfield_corrected_intercept",
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
time_res = "1min"
all_l2_files = sorted(
    # glob.glob(f"/mnt/cephadrius/bu_research/lexi_data/l2/{time_res}/clps-bgm1_lexi_l2-images*.cdf")
    glob.glob(
        "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_analysis/data/1min/clps-bgm1_lexi_l2-images*.cdf"
    )
)
l2_files = keep_highest_versions(all_l2_files)

# ---- Integration configuration ----
integrate_time = 2

if integrate_time == 1:
    integration = "29min"  # e.g., "5min", "10min", "30min", "1H"
    # Optionally restrict the time span (None => use full span of files)
    span_start = "2025-03-16 19:00:00+00:00"
    span_end = "2025-03-16 19:29:00+00:00"
else:
    integration = "105min"  # e.g., "5min", "10min", "30min", "1H"
    # Optionally restrict the time span (None => use full span of files)
    span_start = "2025-03-16 19:30:00+00:00"
    span_end = "2025-03-16 21:15:00+00:00"

fig_format = "pdf"  # "png" or "pdf"

norm_lexi = "linear"
v_min_lexi = 0.02
v_max_lexi = 1e-3

line_x_min = 0.005
line_x_max = 0.017

# Build a table of files with their time coverage
meta = read_l2_metadata_and_paths(l2_files)
if span_start is None:
    span_start = meta["t0"].min()
else:
    span_start = pd.to_datetime(span_start, utc=True)
if span_end is None:
    span_end = meta["t1"].max()
else:
    span_end = pd.to_datetime(span_end, utc=True)

# Build left-closed bins [bin_k, bin_k+Δ)
edges = pd.date_range(span_start, span_end, freq=integration)
# guard: if only one edge, nothing to do
if len(edges) < 2:
    raise RuntimeError("Not enough files/time span to create integration windows.")

for k in range(len(edges) - 1):
    # for k in range(1):
    # Print the progress
    print(f"Processing integration window {k + 1} of {len(edges) - 1}...", end="\r")
    # for k in range(2, 3):  # TEMP: only first 2 windows for testing
    win_start = edges[k]
    win_end = edges[k + 1]

    # Files overlapping the bin if (t1 > win_start) and (t0 < win_end)
    sel = meta[(meta["t1"] > win_start) & (meta["t0"] < win_end)]
    if sel.empty:
        continue  # nothing to integrate in this bin

    # Accumulators (lazy init after reading the first file)
    AZcorn = ELcorn = az_c = el_c = None
    sum_raw = sum_bg = sum_bgff = None

    for f in sel["path"]:
        AZc, ELc, azC, elC, raw, bg, bgff = load_lexi_arrays_one_file(f)
        if sum_raw is None:
            AZcorn, ELcorn, az_c, el_c = AZc, ELc, azC, elC
            sum_raw = np.zeros_like(raw, dtype=float)
            sum_bg = np.zeros_like(bg, dtype=float)
            sum_bgff = np.zeros_like(bgff, dtype=float)

        # Simply sum exposure-weighted counts over the integration period
        sum_raw += np.nan_to_num(raw, nan=0.0)
        sum_bg += np.nan_to_num(bg, nan=0.0)
        sum_bgff += np.nan_to_num(bgff, nan=0.0)

    # Replace all zeroz with NaN for plotting
    sum_raw = np.where(sum_raw == 0, np.nan, sum_raw)
    sum_bg = np.where(sum_bg == 0, np.nan, sum_bg)
    sum_bgff = np.where(sum_bgff == 0, np.nan, sum_bgff)

    # Normalize the sums by the integration duration in seconds
    integration_seconds = (win_end - win_start).total_seconds()
    sum_raw /= integration_seconds
    sum_bg /= integration_seconds
    sum_bgff /= integration_seconds
    # Bookkeeping row for this integrated window
    data_df.loc[len(data_df)] = {"start_time": win_start, "end_time": win_end}

    # ---- Plot the 3 top + 3 bottom subplots for this integrated window ----
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    # fig.suptitle(
    # f"LEXI L2 integrated image  |  {win_start.strftime('%Y-%m-%d %H:%M:%S')}-{win_end.strftime('%H:%M:%S')} UTC",
    # fontsize=1.2 * mpl.rcParams["font.size"],
    # )
    time_range = [win_start, win_end]  # for annotation only

    # --- Top row: integrated maps ---
    plot_on_az_el(
        axs[0, 0],
        AZcorn,
        ELcorn,
        sum_raw,
        time_range=time_range,
        title="Raw Counts",
        cbar_title="Counts/s",  # Sum of counts over the window
        norm=norm_lexi,
        vmin=v_min_lexi,
        vmax=v_max_lexi,
    )
    plot_on_az_el(
        axs[0, 1],
        AZcorn,
        ELcorn,
        sum_bg,
        time_range=time_range,
        title="Background-Corrected Counts",
        cbar_title="Counts/s",
        norm=norm_lexi,
        vmin=v_min_lexi,
        vmax=v_max_lexi,
    )
    plot_on_az_el(
        axs[0, 2],
        AZcorn,
        ELcorn,
        sum_bgff,
        time_range=time_range,
        title="Bg + Flat-Field Corrected Counts",
        cbar_title="Counts/s",
        norm=norm_lexi,
        vmin=v_min_lexi,
        vmax=v_max_lexi,
    )

    # --- Bottom row: integrated line profiles ---
    x_lim = (line_x_min, line_x_max)
    _, data_df, profile_df_raw = plot_line_profile(
        data_df,
        axs[1, 0],
        sum_raw,
        AZcorn,
        ELcorn,
        integration_seconds=integration_seconds,
        title="Line Profile",
        xlabel="Counts/s",
        xlim=x_lim,
        ylim=(20, 29.5),
        key="raw_counts",
    )
    _, data_df, profile_df_bg = plot_line_profile(
        data_df,
        axs[1, 1],
        sum_bg,
        AZcorn,
        ELcorn,
        integration_seconds=integration_seconds,
        title="Line Profile",
        xlabel="Counts/s",
        xlim=x_lim,
        ylim=(20, 29.5),
        key="background_corrected",
    )
    _, data_df, profile_df_bgff = plot_line_profile(
        data_df,
        axs[1, 2],
        sum_bgff,
        AZcorn,
        ELcorn,
        integration_seconds=integration_seconds,
        title="Line Profile",
        xlabel="Counts/s",
        xlim=x_lim,
        ylim=(20, 29.5),
        key="background_flatfield_corrected",
    )
    # Combine the three profile dataframes and save them to a csv file
    combined_profile_df = pd.concat([profile_df_raw, profile_df_bg, profile_df_bgff], axis=1)
    # Rename "el_centers_raw_counts" to "elevation"
    combined_profile_df = combined_profile_df.rename(columns={"el_centers_raw_counts": "elevation"})
    # Remove el_centers_background_corrected and el_centers_background_flatfield_corrected from the combined dataframe
    combined_profile_df = combined_profile_df.drop(columns=["el_centers_background_corrected", "el_centers_background_flatfield_corrected"])

    # Rename counts_raw_counts to raw_counts, counts_background_corrected to background_corrected_counts and counts_background_flatfield_corrected to background_flatfield_corrected_counts
    combined_profile_df = combined_profile_df.rename(columns={"counts_raw_counts": "raw_counts", "counts_background_corrected": "background_corrected_counts", "counts_background_flatfield_corrected": "background_flatfield_corrected_counts"})

    # Round off all values to 5 sig-figs
    combined_profile_df = combined_profile_df.round(5)

    # Set elevation as index
    combined_profile_df = combined_profile_df.set_index("elevation")


    # Save the combined profile dataframe to a csv file
    combined_profile_df.to_csv(f"/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_analysis/data/line_profile_data/bg_corrected/from_l2/combined_profile_df_{win_start.strftime('%Y%m%d_%H%M%S')}_{win_end.strftime('%Y%m%d_%H%M%S')}.csv")

    # Cosmetic overlays + tick formatting (same as your original)
    for ax in axs.flatten()[:3]:
        ax.contour(AZcorn[:-1, :-1], ELcorn[:-1, :-1], az_c, colors="c", linewidths=0.6, alpha=0.5)
        ax.contour(AZcorn[:-1, :-1], ELcorn[:-1, :-1], el_c, colors="k", linewidths=0.6, alpha=0.5)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", which="major", labelsize=1 * mpl.rcParams["font.size"])
        ax.tick_params(axis="both", which="minor", labelsize=1 * mpl.rcParams["font.size"])
        ax.xaxis.get_offset_text().set(size=1 * mpl.rcParams["font.size"])
        ax.yaxis.get_offset_text().set(size=1 * mpl.rcParams["font.size"])
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    for ax in axs.flatten()[3:]:
        ax.tick_params(axis="both", which="major", labelsize=1 * mpl.rcParams["font.size"])
        ax.tick_params(axis="both", which="minor", labelsize=1 * mpl.rcParams["font.size"])
        ax.xaxis.get_offset_text().set(size=1 * mpl.rcParams["font.size"])
        ax.yaxis.get_offset_text().set(size=1 * mpl.rcParams["font.size"])
        # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # Save figure per integrated window
    outdir = Path(f"../figures/line_profiles/bg_corrected/from_l2/az_el_integrated_{integration}/")
    outdir.mkdir(parents=True, exist_ok=True)
    fig_name = (
        f"lexi_l2_integrated_{win_start.strftime('%Y%m%d_%H%M%S')}_{win_end.strftime('%H%M%S')}"
    )
    combined_profile_df.to_csv(outdir / f"{fig_name}.csv", index=False)
    fig.savefig(
        outdir / f"{fig_name}.{fig_format}",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
        format=fig_format,
    )
    print(f"Figure and CSV saved to {outdir / fig_name}.*")
    plt.close(fig)

    sums = {"sum_raw": sum_raw, "sum_bg": sum_bg, "sum_bgff": sum_bgff}
    # hist_choice = "sum_raw"  # or "sum_raw" / "sum_bgff"

    for hist_choice in ["sum_raw", "sum_bg", "sum_bgff"]:

        # fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
        # plot_selected_integrated_hist(
        #     ax2,
        #     AZcorn,
        #     ELcorn,
        #     hist_choice,
        #     sums,
        #     time_range=[win_start, win_end],
        #     norm=norm_lexi,
        #     vmin=v_min_lexi,
        #     vmax=v_max_lexi,
        #     cbar_title="Counts",
        # )
        # fig2.savefig(
        #     outdir
        #     / f"histmap_{hist_choice}_{win_start.strftime('%Y%m%d_%H%M%S')}_{win_end.strftime('%H%M%S')}.png",
        #     dpi=200,
        # )
        # plt.close(fig2)

        # title = (
        #     f"{hist_choice} | "
        #     f"{win_start.strftime('%Y-%m-%d %H:%M:%S')}–{win_end.strftime('%H:%M:%S')} UTC"
        # )

        # g = plot_selected_integrated_hist_with_marginals(
        #     AZcorn,
        #     ELcorn,
        #     sums[hist_choice],
        #     title=title,
        #     cmap="mako",  # any seaborn/mpl cmap
        #     log_color=False,  # set True if dynamic range is huge
        #     cbar_label="Counts",  # integrated counts
        #     height=8,
        #     ratio=4,
        #     linewidth=1.6,
        # )

        # outdir = Path(
        #     f"../figures/line_profiles/bg_corrected/from_l2/az_el_integrated_{integration}/histmaps_with_marginals/"
        # )
        # outdir.mkdir(parents=True, exist_ok=True)
        # fname = f"{hist_choice}_{win_start.strftime('%Y%m%d_%H%M%S')}.png"
        # g.fig.savefig(outdir / fname, dpi=200, bbox_inches="tight")
        # plt.close(g.fig)

        figsize_exact = (10, 9)

        # title = f"{hist_choice} | {win_start:%Y-%m-%d %H:%M:%S}-{win_end:%H:%M:%S} UTC"
        fig, _ = plot_integrated_map_with_marginals_seaborn(
            AZcorn,
            ELcorn,
            sums[hist_choice],
            figsize=figsize_exact,  # <-- exact width/height
            # title=title,
            cmap="plasma",
            log_color=False,
            cbar_label="Counts/s",
            joint_ratio=7,
            scatter_size=10,
            normalize_marginals=True,  # <-- your #1
            cbar_lims=(v_min_lexi, v_max_lexi),
        )

        outdir = Path(
            f"../figures/line_profiles/bg_corrected/from_l2/az_el_integrated_{integration}/histmaps_with_marginals/"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        fname = f"{hist_choice}_normalized_{win_start.strftime('%Y%m%d_%H%M%S')}_{win_end.strftime('%H%M%S')}.{fig_format}"
        fig.savefig(outdir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
