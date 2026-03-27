# slope_v_time_series_v3.py
import datetime
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from spacepy.pycdf import CDF as cdf

# --------------------------------------------
# Config
# --------------------------------------------
plt.style.use("default")
mpl.rcParams.update({"font.size": 25})
# Set the fonttype to 42 to avoid Type 3 fonts in the output PDF/PNGs
mpl.rcParams["pdf.fonttype"] = 42
# Set the the font to be arial-like for better readability
mpl.rcParams["font.family"] = "Arial"
# Set latex-style text rendering
mpl.rcParams["text.usetex"] = False


# --------------------------------------------
# Helpers
# --------------------------------------------


def keep_highest_versions(paths):
    best = {}
    for p in map(Path, paths):
        stem = p.stem
        try:
            base, vstr = stem.rsplit("_v", 1)
        except ValueError:
            base, vstr = stem, "0"
        vtup = tuple(int(x) for x in vstr.split("."))
        if base not in best or vtup > best[base][0]:
            best[base] = (vtup, str(p))
    return [best[k][1] for k in sorted(best)]


def centers_to_corners_2d(ra_c, dec_c):
    ra_c = np.asarray(ra_c, float)
    dec_c = np.asarray(dec_c, float)
    H, W = ra_c.shape
    ra_i = 0.5 * (ra_c[1:, :] + ra_c[:-1, :])
    ra_j = 0.5 * (ra_c[:, 1:] + ra_c[:, :-1])
    dec_i = 0.5 * (dec_c[1:, :] + dec_c[:-1, :])
    dec_j = 0.5 * (dec_c[:, 1:] + dec_c[:, :-1])

    ra_top = ra_c[0, :] - (ra_i[0, :] - ra_c[0, :])
    ra_bot = ra_c[-1, :] + (ra_c[-1, :] - ra_i[-1, :])
    dec_top = dec_c[0, :] - (dec_i[0, :] - dec_c[0, :])
    dec_bot = dec_c[-1, :] + (dec_c[-1, :] - dec_i[-1, :])

    ra_left = ra_c[:, 0] - (ra_j[:, 0] - ra_c[:, 0])
    ra_right = ra_c[:, -1] + (ra_c[:, -1] - ra_j[:, -1])
    dec_left = dec_c[:, 0] - (dec_j[:, 0] - dec_c[:, 0])
    dec_right = dec_c[:, -1] + (dec_c[:, -1] - dec_j[:, -1])

    RAc = np.empty((H + 1, W + 1), float)
    DECc = np.empty((H + 1, W + 1), float)

    RAc[1:H, 1:W] = 0.25 * (ra_c[:-1, :-1] + ra_c[1:, :-1] + ra_c[:-1, 1:] + ra_c[1:, 1:])
    DECc[1:H, 1:W] = 0.25 * (dec_c[:-1, :-1] + dec_c[1:, :-1] + dec_c[:-1, 1:] + dec_c[1:, 1:])

    RAc[0, 1:W] = 0.5 * (ra_top[:-1] + ra_top[1:])
    RAc[-1, 1:W] = 0.5 * (ra_bot[:-1] + ra_bot[1:])
    RAc[1:H, 0] = 0.5 * (ra_left[:-1] + ra_left[1:])
    RAc[1:H, -1] = 0.5 * (ra_right[:-1] + ra_right[1:])

    DECc[0, 1:W] = 0.5 * (dec_top[:-1] + dec_top[1:])
    DECc[-1, 1:W] = 0.5 * (dec_bot[:-1] + dec_bot[1:])
    DECc[1:H, 0] = 0.5 * (dec_left[:-1] + dec_left[1:])
    DECc[1:H, -1] = 0.5 * (dec_right[:-1] + dec_right[1:])

    RAc[0, 0] = 0.5 * (ra_top[0] + ra_left[0])
    RAc[0, -1] = 0.5 * (ra_top[-1] + ra_right[0])
    RAc[-1, 0] = 0.5 * (ra_bot[0] + ra_left[-1])
    RAc[-1, -1] = 0.5 * (ra_bot[-1] + ra_right[-1])

    DECc[0, 0] = 0.5 * (dec_top[0] + dec_left[0])
    DECc[0, -1] = 0.5 * (dec_top[-1] + dec_right[0])
    DECc[-1, 0] = 0.5 * (dec_bot[0] + dec_left[-1])
    DECc[-1, -1] = 0.5 * (dec_bot[-1] + dec_right[-1])

    return RAc, DECc


def elevation_profile_normalized(ELcorn: np.ndarray, H: np.ndarray):
    H = np.asarray(H, dtype=float)
    ELcorn = np.asarray(ELcorn, dtype=float)
    ELc = 0.25 * (ELcorn[:-1, :-1] + ELcorn[1:, :-1] + ELcorn[:-1, 1:] + ELcorn[1:, 1:])
    mask = ~np.isnan(H)
    with np.errstate(invalid="ignore"):
        elev_row_mean = np.nanmean(np.where(mask, ELc, np.nan), axis=1)
    row_sum = np.nansum(H, axis=1)
    row_counts = np.sum(mask, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_profile = np.where(row_counts > 0, row_sum / row_counts, np.nan)
    total_counts = np.nansum(H)
    return elev_row_mean, norm_profile, row_sum, row_counts, total_counts


def list_l2_files():
    all_l2 = sorted(glob.glob(L2_GLOB))
    return keep_highest_versions(all_l2)


def find_file_covering_time(l2_files, when_utc):
    """Return first L2 file whose [epoch_start, epoch_end] covers 'when_utc' (UTC-aware)."""
    for fp in l2_files:
        with cdf(fp) as dat:
            t0 = pd.to_datetime(dat["epoch_start"][...][0], utc=True)
            t1 = pd.to_datetime(dat["epoch_end"][...][0], utc=True)
            if t0 <= when_utc <= t1:
                return fp, t0, t1
    return None, None, None


def load_profile_for_time(l2_file):
    """Load background & flat-field corrected counts * exposure and convert to a line profile."""
    dat = cdf(l2_file)
    try:
        az_c = np.asarray(dat["az_bin_map"][...])[0]
        el_c = np.asarray(dat["el_bin_map"][...])[0]
        AZcorn, ELcorn = centers_to_corners_2d(az_c, el_c)
        img = (
            np.asarray(dat["lexi_image_background_corrected"][...])[0]
            / np.asarray(dat["pixel_area"][...])[0]
            * np.asarray(dat["exposure_map"][...])[0]
            / np.asarray(dat["exposure_map"][...])[0]
        )
        img = np.where(img <= 0, np.nan, img)
        el_centers, profile, *_ = elevation_profile_normalized(ELcorn, img)
        return el_centers, profile
    finally:
        dat.close()


def plot_one_line_profile(
    ax,
    el_centers,
    profile,
    simulated_df=None,
    when_utc=None,
    is_first=False,
    color="magenta",
    plot_simulated=True,
):
    ax.scatter(profile * x_axis_exponent_factor, el_centers, s=4, marker="d", color=color, alpha=1)
    valid = np.isfinite(profile) & np.isfinite(el_centers)
    # Ignore first 3 and last 3 valid points for the fit
    valid_indices = np.where(valid)[0]
    if len(valid_indices) > 6:
        valid[valid_indices[:3]] = False
        valid[valid_indices[-3:]] = False
    if valid.sum() >= 2:
        coeffs = np.polyfit(
            el_centers[valid][3:-3], profile[valid][3:-3] * x_axis_exponent_factor, deg=1
        )
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(np.nanmin(el_centers[valid]), np.nanmax(el_centers[valid]), 100)
        y_fit = poly(x_fit)
        ax.plot(y_fit, x_fit, color="k", linestyle="--", linewidth=1.5)

    # If requested, overplot simulated profile at the same time
    if plot_simulated and simulated_df is not None and when_utc is not None:
        # Find the closest time column in simulated_df
        time_cols = [col for col in simulated_df.columns if col != "Elevation"]
        time_diffs = [
            abs((pd.Timestamp(col) - pd.Timestamp(when_utc)).total_seconds()) for col in time_cols
        ]
        closest_col = time_cols[np.argmin(time_diffs)]
        sim_profile = simulated_df[closest_col].values

        ax.scatter(
            sim_profile * x_axis_exponent_factor,
            simulated_df["Elevation"].values,
            color="green",
            s=4,
            marker="o",
            alpha=1,
        )
        if is_first:
            # Modify the marker size for the legend
            marker_size = 8
            ax.scatter(
                [],
                [],
                color="green",
                s=marker_size,
                marker="o",
                alpha=1,
                label="Simulated",
            )

            ax.legend(loc="upper right", framealpha=1, fontsize=mpl.rcParams["font.size"] * 0.6)
    # Requested limits
    ax.set_xlim(LINE_X_MIN, LINE_X_MAX)
    ax.set_xscale("linear")
    # Modify the x-axis so that the ticks are in scientific notation
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0001e"))

    ax.set_ylim(EL_Y_LIM)
    ax.set_xlabel("")  # no axis label
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    ax.spines["bottom"].set_visible(True)

    ax.set_ylabel("")
    ax.set_title("")

    # Add horizontal gridlines
    ax.grid(axis="y", alpha=0.5, linestyle="-", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.set_axisbelow(True)

    # --- always show bottom axis and labels ---
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    if is_first:
        ax.set_ylabel("Elevation [deg]")
        ax.set_yticks(np.arange(EL_Y_LIM[0], EL_Y_LIM[1] + 1, 2.0))
        ax.spines["left"].set_visible(True)
        # Set the x-axis label only for the first plot
        ax.set_xlabel(
            "Counts [s$^{-1}$ arcmin$^{-2}$] \n ($\\times 10^{-4}$)",
            labelpad=-30,
            fontsize=0.85 * mpl.rcParams["font.size"],
            rotation=0,
            va="bottom",
            ha="left",
            # Set the x position to be at the center of the entire row of subplots
            x=0.01,
        )
    else:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    # Time as title
    if when_utc is not None:
        ax.set_title(pd.Timestamp(when_utc).strftime("%H:%M"))


# --------------------------------------------
# Load the simulated data
# --------------------------------------------
time_res = "5min"  # must match how the CSV & L2 path were generated
simulated_data_folder = Path(f"../data/line_profile_data/bg_corrected/from_l2/{time_res}/")
file_name = "simulation_results_lexi_gonzalo.csv"
data_path = simulated_data_folder / file_name
simulated_df = pd.read_csv(data_path)
elevation_offset = 20  # degrees
profile_offset = 4  # counts/s/arcmin^2
deg_sequare_to_arcmin_square = 60.0 * 60.0

# For keys other than "Elevation", modify it to full datetime
t0 = pd.Timestamp("2025-03-16", tz="UTC")

# Keep "Elevation" unchanged; convert the rest
new_cols = []
for c in simulated_df.columns:
    if c == "Elevation":
        new_cols.append(c)
    else:
        # Parse the HH:MM string and add to the base date
        dt = pd.to_datetime(f"{t0.date()} {c}", utc=True)
        new_cols.append(dt)

simulated_df.columns = new_cols

# Add the elevation offset
simulated_df["Elevation"] += elevation_offset

# Modify each profile to be in counts/s/arcmin^2
for c in simulated_df.columns:
    if c != "Elevation":
        simulated_df[c] = simulated_df[c] / deg_sequare_to_arcmin_square * profile_offset


# --------------------------------------------
# Load slope time series
# --------------------------------------------

data_folder = Path(f"../data/line_profile_data/bg_corrected/from_l2/{time_res}/")
file_name = f"line_profile_fit_parameters_bg_corrected_no_flat_field_{time_res}.csv"
data_path = data_folder / file_name

# L2 file pattern (same style as used to generate the CSV)
L2_GLOB = f"/mnt/cephadrius/bu_research/lexi_data/l2/{time_res}/clps-bgm1_lexi_l2-images*.cdf"

# Limits for line profiles
x_axis_exponent_factor = 1e4  # to plot x-axis in units of 1e-4
LINE_X_MIN = 1e-4 * x_axis_exponent_factor
LINE_X_MAX = 7.50e-4 * x_axis_exponent_factor
EL_Y_LIM = (20.0, 29.5)

series_to_plot = 1

if series_to_plot == 1:
    # Times to showcase in the top row
    times_to_plot = [
        datetime.datetime(2025, 3, 16, 19, 12, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 19, 22, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 19, 42, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 20, 2, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 20, 22, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 20, 42, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 21, 2, 30, tzinfo=datetime.timezone.utc),
    ]
else:
    times_to_plot = [
        datetime.datetime(2025, 3, 16, 19, 12, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 19, 22, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 19, 37, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 20, 2, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 20, 27, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2025, 3, 16, 20, 52, 30, tzinfo=datetime.timezone.utc),
        # datetime.datetime(2025, 3, 16, 21, 2, 30, tzinfo=datetime.timezone.utc),
    ]
# len_times_to_plot = 6
# start_time = pd.Timestamp("2025-03-16 19:05:00", tz="UTC")
# end_time = pd.Timestamp("2025-03-16 21:05:00", tz="UTC")
# total_duration_td = end_time - start_time
# time_res_td = total_duration_td / (len_times_to_plot - 1)
# times_to_plot = [start_time + i * time_res_td for i in range(len_times_to_plot)]

df = pd.read_csv(data_path)

df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
df["middle_time"] = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
df.set_index("middle_time", inplace=True)
df.sort_index(inplace=True)

if "background_corrected_slope" not in df.columns:
    raise RuntimeError("Expected column 'background_corrected_slope' not found in CSV.")

# --------------------------------------------
# Build the figure: top = profiles, bottom = slope vs time
# --------------------------------------------
n_top = len(times_to_plot)
# Define 6 specific colors for the time points
colors = ["#1f77b4", "#ff7f0e", "#0591ef", "#d62728", "#9467bd", "#8c564b", "#033b15"]

fig = plt.figure(figsize=(3.0 * n_top, 10), constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.2])
gs_top = gs[0].subgridspec(1, n_top, wspace=0.05)  # tight spacing since axes are invisible
ax_top = [fig.add_subplot(gs_top[0, i]) for i in range(n_top)]
ax_bottom = fig.add_subplot(gs[1, 0])

# Prepare L2 inventory once
l2_files = list_l2_files()

# Top row: draw profiles (no labels/axes/titles; fixed x limits)
for i, tsel in enumerate(times_to_plot):
    fp, t0, t1 = find_file_covering_time(l2_files, tsel)
    # print(f"Time {tsel} covered by file: {fp} (from {t0} to {t1})")
    if fp is not None:
        elc, prof = load_profile_for_time(fp)
        plot_one_line_profile(
            ax_top[i],
            elc,
            prof,
            simulated_df=simulated_df,
            when_utc=tsel,
            is_first=(i == 0),
            color=colors[i],
            plot_simulated=True,
        )
    else:
        ax_top[i].set_xlim(LINE_X_MIN, LINE_X_MAX)
        ax_top[i].set_ylim(EL_Y_LIM)
        ax_top[i].grid(axis="y", alpha=0.5, linestyle="-", linewidth=0.5)
        ax_top[i].spines["top"].set_visible(False)
        ax_top[i].spines["right"].set_visible(False)
        ax_top[i].spines["bottom"].set_visible(True)

        # show x-axis ticks and labels
        ax_top[i].tick_params(axis="x", which="both", bottom=True, labelbottom=True)

for i, ax in enumerate(ax_top):
    ax.set_ylim(EL_Y_LIM)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2.0))  # grid every 2 deg
    ax.set_axisbelow(True)  # grid behind data
    ax.grid(axis="y", which="major", alpha=1, linestyle="-", linewidth=0.8, zorder=0)

    if i == 0:
        ax.set_ylabel("Elevation [deg]")
    else:
        # Hide labels, not the ticks themselves
        ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=False)
        # (optionally still hide spines)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # Ensure all subplots have horizontal gridlines
    # for ax in ax_top:
    # ax.grid(axis="y", alpha=1, linestyle="-", linewidth=3, color="lightgray", zorder=10)

# Bottom row: slope vs time (NO title); remove top/right spines
# ax_bottom.plot(df.index, df["background_corrected_slope"], ls="--", lw=1.5, marker="o")
ax_bottom.step(
    df.index,
    df["background_corrected_slope"],
    where="mid",
    ls="--",
    lw=1.5,
    marker="o",
    label="Measured Slope",
)
ax_bottom.set_xlabel("Time [UTC]")
# Format x-axis tick labels to only show hours and minutes
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax_bottom.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_bottom.set_ylabel("Slope")
ax_bottom.set_title("")

# Remove the top and right spines as requested
ax_bottom.spines["top"].set_visible(False)
ax_bottom.spines["right"].set_visible(False)

# ax_bottom.legend(loc="best", framealpha=0.3)
ax_bottom.grid(alpha=0.2)

# ----- Vertical dashed lines + labels at selected times -----
ymin, ymax = ax_bottom.get_ylim()
for idx, tsel in enumerate(times_to_plot):
    # draw a vertical dashed line with matching color
    # Get the delta time from time_res
    delta_time = pd.to_timedelta(time_res) / 2
    ax_bottom.axvline(tsel, linestyle="--", linewidth=1.2, alpha=0.8, zorder=2, color=colors[idx])
    ax_bottom.axvspan(
        tsel - delta_time, tsel + delta_time, color=colors[idx], alpha=0.1, zorder=1
    )

    # label right beside the line near the top of the axes
    label = pd.Timestamp(tsel).strftime("%H:%M")
    # Place slightly to the right of the line with an offset, anchored at the top
    ax_bottom.annotate(
        label,
        xy=(tsel - delta_time, (ymin + ymax) / 2),
        xycoords=("data", "data"),
        xytext=(3, -4),  # small offset in points (right, down)
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=0.9 * mpl.rcParams["font.size"],
        bbox=dict(facecolor="white", alpha=0.3, pad=1.5, edgecolor="None"),
    )

sunset_time = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)
ax_bottom.axvline(sunset_time, color="red", linestyle="--", linewidth=1.5)
ax_bottom.axvspan(
    sunset_time - pd.Timedelta(minutes=29),
    sunset_time,
    color="k",
    alpha=0.1,
    zorder=1,
)
ax_bottom.annotate(
    "Sunset Ends",
    xy=(sunset_time, 0.8 * (ymin + ymax)),
    xycoords=("data", "data"),
    xytext=(5, -5),  # small offset in points (right, down)
    textcoords="offset points",
    ha="left",
    va="top",
    fontsize=0.9 * mpl.rcParams["font.size"],
    color="red",
    bbox=dict(facecolor="white", alpha=0.3, pad=1.5, edgecolor="None"),
)
ax_bottom.set_xlim(
    df.index.min() - pd.Timedelta(minutes=1.5),
    df.index.max() + pd.Timedelta(minutes=1.5),
)


# --------------------------------------------
# Get the best fit line slope and intercept between Elevation and values of each column in
# df_simulated
time_cols = [col for col in simulated_df.columns if col != "Elevation"]
slope_list = []
for col in time_cols:
    valid = np.isfinite(simulated_df["Elevation"].values) & np.isfinite(simulated_df[col].values)
    if valid.sum() >= 2:
        coeffs = np.polyfit(
            simulated_df["Elevation"].values[valid],
            simulated_df[col].values[valid],
            deg=1,
        )
        slope, intercept = coeffs
        slope_list.append(slope)
        # print(
        #     f"Simulated data column {col}: slope = {slope:.6e} [counts/s/arcmin^2]/deg, intercept = {intercept:.6e} [counts/s/arcmin^2]"
        # )
slope_array = np.array(slope_list)
# Plot the slope array using twin axes on ax_bottom
ax_slope = ax_bottom
# Shift time_cols by 2.5 minutes to align with the center of the time bins\
shifted_time_cols = [col + pd.Timedelta(minutes=2.5) for col in time_cols]
# plot using step plot
ax_slope.step(
    pd.to_datetime(shifted_time_cols, utc=True),
    slope_array,
    where="mid",
    color="green",
    linestyle="--",
    linewidth=1.5,
    marker="s",
    label="Simulated Slope",
)
# ax_slope.plot(
#     pd.to_datetime(time_cols, utc=True),
#     slope_array,
#     color="green",
#     linestyle="--",
#     linewidth=1.5,
#     marker="s",
#     label="Simulated Slope",
# )
# ax_slope.set_ylabel("Simulated Slope", color="green")
# ax_slope.tick_params(axis="y", labelcolor="green")
# ax_slope.spines["top"].set_visible(False)
# ax_slope.spines["right"].set_visible(True)
# ax_slope.grid(alpha=0.0)  # no grid for twin axis
# --------------------------------------------

# Add annotation at the bottom right corner with green square and label "Simulated Slope", similarly
# for measured slope with dashed line and circle marker
# Combined annotation box
ax_bottom.annotate(
    "Simulated\nMeasured",
    xy=(0.9, 0.20),
    xycoords="axes fraction",
    ha="left",
    va="top",
    fontsize=0.75 * mpl.rcParams["font.size"],
    bbox=dict(facecolor="white", alpha=0.3, pad=3, edgecolor="None"),
)

# Simulated slope sample (top)
ax_bottom.plot(
    [0.87, 0.89],
    [0.165, 0.165],
    transform=ax_bottom.transAxes,
    color="green",
    linestyle="--",
    linewidth=0.5,
    marker="s",
    markersize=5,
)

# Measured slope sample (bottom)
ax_bottom.plot(
    [0.87, 0.89],
    [0.085, 0.085],
    transform=ax_bottom.transAxes,
    color=colors[0],
    linestyle="--",
    linewidth=0.5,
    marker="o",
    markersize=5,
)

# Save
# outdir = Path("../figures/slope_time_series/")
outdir = Path("/home/cephadrius/Desktop/git/overleaf_projects/lexi_draft/figures/")
outdir.mkdir(parents=True, exist_ok=True)
figure_format = "pdf"  # "pdf" or "png"
outfile = (
    outdir
    / f"slope_v_time_{time_res}_profiles_plus_series_vlines_v2_no_flat_field_{len(times_to_plot)}_v2.{figure_format}"
)
fig.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"Saved figure: {outfile}")
# plt.show()
