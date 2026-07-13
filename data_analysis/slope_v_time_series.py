# slope_v_time_series.py
import datetime
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from spacepy.pycdf import CDF as cdf

# --------------------------------------------
# Config
# --------------------------------------------
plt.style.use("dark_background")
mpl.rcParams.update({"font.size": 14})

time_res = "5min"  # must match how the CSV & L2 path were generated
data_folder = Path(f"../data/line_profile_data/bg_corrected/from_l2/{time_res}/")
file_name = f"line_profile_fit_parameters_bg_corrected_{time_res}_flux_avg_.csv"
data_path = data_folder / file_name

# L2 file pattern (same style as used to generate the CSV)
L2_GLOB = f"/mnt/cephandrius/bu_research/lexi_data/l2/{time_res}/clps-bgm1_lexi_l2-images*.cdf"

# Color/limits for line profiles
LINE_X_MIN = 0.0
LINE_X_MAX = 3.0
EL_Y_LIM = (20.0, 29.5)

# “Sunset” (used for optional normalization below; keep if you need it)
SUNSET_TIME = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)

# Times to showcase in the top row (choose 5-ish)
times_to_plot = [
    datetime.datetime(2025, 3, 16, 19, 10, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 3, 16, 19, 20, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 3, 16, 19, 40, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 3, 16, 20, 0, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 3, 16, 21, 0, 0, tzinfo=datetime.timezone.utc),
]


# --------------------------------------------
# Helpers copied/adapted from your L2 plotting code
# --------------------------------------------
def keep_highest_versions(paths):
    best = {}
    for p in map(Path, paths):
        stem = p.stem
        try:
            base, vstr = stem.rsplit("_V", 1)
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
        # Use the same product as in your generator script
        img = (
            np.asarray(dat["lexi_image_background_flatfield_corrected"][...])[0]
            * np.asarray(dat["exposure_map"][...])[0]
        )
        # No <=0 values
        img = np.where(img <= 0, np.nan, img)
        el_centers, profile, *_ = elevation_profile_normalized(ELcorn, img)
        return el_centers, profile
    finally:
        dat.close()


def plot_one_line_profile(ax, el_centers, profile, title):
    ax.scatter(profile, el_centers, s=10, marker=".", color="magenta")
    # Fit (ignore NaNs)
    valid = np.isfinite(profile) & np.isfinite(el_centers)
    if valid.sum() >= 2:
        coeffs = np.polyfit(el_centers[valid], profile[valid], deg=1)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(np.nanmin(el_centers[valid]), np.nanmax(el_centers[valid]), 100)
        y_fit = poly(x_fit)
        ax.plot(y_fit, x_fit, color="cyan", linestyle="--", linewidth=1.5, label="Fit")
        # R^2
        resid = profile[valid] - poly(el_centers[valid])
        ss_res = np.nansum(resid**2)
        ss_tot = np.nansum((profile[valid] - np.nanmean(profile[valid])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        ax.text(
            0.02,
            0.02,
            f"slope={coeffs[0]:.3e}\nR²={r2:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="k", alpha=0.5, pad=2),
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Counts")
    ax.set_ylabel("Elevation [deg]")
    ax.set_xlim(LINE_X_MIN, LINE_X_MAX)
    ax.set_ylim(EL_Y_LIM)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))


# --------------------------------------------
# Load slope time series
# --------------------------------------------
df = pd.read_csv(data_path)

# Compute middle time (UTC-aware)
df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
df["middle_time"] = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
df.set_index("middle_time", inplace=True)
df.sort_index(inplace=True)

# Optional: normalized slope (post-sunset baseline)
if "background_flatfield_corrected_slope" in df.columns:
    df["normalized_slope"] = (
        df["background_flatfield_corrected_slope"]
        .where(df.index > SUNSET_TIME)
        .transform(lambda x: (x - x.mean()) / x.std())
    )

# --------------------------------------------
# Build the figure: top = profiles, bottom = slope vs time
# --------------------------------------------
n_top = len(times_to_plot)
fig = plt.figure(figsize=(4.0 * n_top, 10), constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.2])
gs_top = gs[0].subgridspec(1, n_top, wspace=0.25)
ax_top = [fig.add_subplot(gs_top[0, i]) for i in range(n_top)]
ax_bottom = fig.add_subplot(gs[1, 0])

# Prepare L2 inventory once
l2_files = list_l2_files()

# Top row: line profiles at specific times
for i, tsel in enumerate(times_to_plot):
    fp, t0, t1 = find_file_covering_time(l2_files, tsel)
    if fp is None:
        ax_top[i].text(0.5, 0.5, "No L2 file found", ha="center", va="center")
        ax_top[i].set_axis_off()
        continue
    elc, prof = load_profile_for_time(fp)
    title = f"{tsel.strftime('%H:%M:%S UTC')}\n({Path(fp).stem})"
    plot_one_line_profile(ax_top[i], elc, prof, title)

# Bottom row: slope vs time
if "background_flatfield_corrected_slope" not in df.columns:
    raise RuntimeError("Expected column 'background_flatfield_corrected_slope' not found in CSV.")

ax_bottom.plot(
    df.index, df["background_flatfield_corrected_slope"], lw=1.5, label="Slope (bg+FF corrected)"
)
ax_bottom.set_xlabel("Time (UTC)")
ax_bottom.set_ylabel("Slope")
ax_bottom.set_title("Slope vs Time (middle of interval)")

# Vertical markers for showcased times
for tsel in times_to_plot:
    ax_bottom.axvline(tsel, color="cyan", linestyle="--", alpha=0.6, linewidth=1.0)

ax_bottom.legend(loc="best", framealpha=0.3)
ax_bottom.grid(alpha=0.2)

# Overall title
fig.suptitle(f"Line Profiles (Top) and Slope Time Series (Bottom)  —  {time_res}", fontsize=18)

# Save + show
outdir = Path("../figures/slope_time_series/")
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / f"slope_v_time_{time_res}_profiles_plus_series.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.1)
print(f"Saved figure: {outfile}")
# plt.show()
