"""
THEMIS-LEXI Planar Propagation Analysis

This script performs the following:
1. Loads THEMIS-C magnetic field, plasma (MOM), and orbital position data 
   from local ISTP-compliant JSON files for 2025-03-16 19:30:00 to 2025-03-16 21:15:00.
2. Converts orbital distance to km and computes flux from density and velocity.
3. Loads LEXI count data from CSV and converts to counts per second.
4. Loads LEXI spacecraft position data.
5. Performs Mean Variance Analysis (MVA) on the THEMIS-C magnetic field to
   determine the normal direction of the planar solar-wind structure.
6. Calculates the propagation delay between THEMIS-C and LEXI.
7. Plots original and time-aligned LEXI data alongside THEMIS-C flux,
   with delay color-coding, and saves to disk.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
TIME_RANGE = ["2025-03-16 19:30:00", "2025-03-16 21:15:00"]
SPACECRAFT = "c"
RE_TO_KM = 6371.2

THEMIS_DATA_DIR = Path("/home/cephandrius/Desktop/git/Lexi-BU/lexi_data_analysis/data/downloaded_themis_c_data")
MOM_JSON = THEMIS_DATA_DIR / "THC_L2_MOM_3764009.json"
FGM_JSON = THEMIS_DATA_DIR / "THC_L2_FGM_3764009.json"
SSC_JSON = THEMIS_DATA_DIR / "THC_OR_SSC_3764009.json"

LEXI_DATA_FILE = Path(
    "../data/line_profile_data/bg_corrected/from_l2/"
    "line_profile_fit_parameters_bg_corrected_1min.csv"
)
LEXI_POSITION_FILE = Path(
    "../data/lexi_themis_analysis/lexi_themis_c_analysis_lexi_spacecraft.csv"
)
FIGURE_SAVE_DIR = Path("../figures/lexi_themis_c_analysis/planar_propagation/")


# =============================================================================
# 1. Load THEMIS-C Data
# =============================================================================
def _parse_istp_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    time_col = next((c for c in df.columns if "epoch" in c.lower() or "time" in c.lower()), None)
    if time_col:
        df["Epoch"] = pd.to_datetime(df[time_col], utc=True)
        df.set_index("Epoch", inplace=True)
        if time_col != "Epoch":
            df = df.drop(columns=[time_col])

    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].dropna().iloc[0], list):
            expanded = pd.DataFrame(df[col].tolist(), index=df.index)
            if expanded.shape[1] == 3:
                expanded.columns = [f"{col}_x", f"{col}_y", f"{col}_z"]
            df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    df = df[~df.index.duplicated(keep="first")]
    return df


def load_themis_data(time_range, sc="c"):
    df_fgm = _parse_istp_json(FGM_JSON)
    b_cols = [c for c in df_fgm.columns if "fgs" in c.lower() and c[-1] in ("x", "y", "z")]
    if not b_cols:
        b_cols = [c for c in df_fgm.columns if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")]
    df_fgm = df_fgm.rename(columns={
        b_cols[0]: f"th{sc}_fgs_gse_x",
        b_cols[1]: f"th{sc}_fgs_gse_y",
        b_cols[2]: f"th{sc}_fgs_gse_z"
    })

    df_mom = _parse_istp_json(MOM_JSON)
    dens_col = next(c for c in df_mom.columns if "density" in c.lower())
    vel_cols = [c for c in df_mom.columns if "velocity" in c.lower() and c[-1] in ("x", "y", "z")]
    df_mom = df_mom.rename(columns={
        dens_col: f"th{sc}_mom_density",
        vel_cols[0]: f"th{sc}_velocity_gse_x",
        vel_cols[1]: f"th{sc}_velocity_gse_y",
        vel_cols[2]: f"th{sc}_velocity_gse_z"
    })

    for c in ("x", "y", "z"):
        df_mom[f"th{sc}_ion_flux_{c}"] = df_mom[f"th{sc}_mom_density"] * df_mom[f"th{sc}_velocity_gse_{c}"] * 1e5
    df_mom[f"th{sc}_ion_flux_mag"] = np.sqrt(
        df_mom[f"th{sc}_ion_flux_x"]**2 + df_mom[f"th{sc}_ion_flux_y"]**2 + df_mom[f"th{sc}_ion_flux_z"]**2
    )

    df_ssc = _parse_istp_json(SSC_JSON)
    pos_cols = [c for c in df_ssc.columns if "xyz" in c.lower() or ("gse" in c.lower() and c[-1] in ("x", "y", "z"))]
    if not pos_cols:
        pos_cols = [c for c in df_ssc.columns if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")]
    df_ssc = df_ssc.rename(columns={
        pos_cols[0]: f"th{sc}_pos_gse_x",
        pos_cols[1]: f"th{sc}_pos_gse_y",
        pos_cols[2]: f"th{sc}_pos_gse_z"
    })

    for c in ("x", "y", "z"):
        df_ssc[f"th{sc}_pos_gse_{c}"] *= RE_TO_KM

    df = pd.concat([df_fgm, df_mom, df_ssc], axis=1).sort_index()

    t0 = df.index.min().ceil("min")
    t1 = df.index.max().floor("min")
    if t0.tz is None:
        target_idx = pd.date_range(t0, t1, freq="1min", tz="UTC")
    else:
        target_idx = pd.date_range(t0, t1, freq="1min", tz=df.index.tz)

    df = df.reindex(target_idx, method="nearest", tolerance=pd.Timedelta("30s"))
    df = df.interpolate(method="time", limit=5)

    t_start = pd.to_datetime(time_range[0], utc=True)
    t_end = pd.to_datetime(time_range[1], utc=True)
    df = df[(df.index >= t_start) & (df.index <= t_end)]

    return df


# =============================================================================
# 2. Load LEXI Data
# =============================================================================
def load_lexi_counts(time_range):
    df = pd.read_csv(LEXI_DATA_FILE)

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    df["mid_time"] = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
    df.set_index("mid_time", inplace=True)
    df.sort_index(inplace=True)

    df["counts_per_second"] = df["background_corrected_total_hist_counts"] / 60.0

    t0 = pd.to_datetime(time_range[0], utc=True)
    t1 = pd.to_datetime(time_range[1], utc=True)
    df = df[(df.index >= t0) & (df.index <= t1)]

    return df


# =============================================================================
# 3. Load LEXI Spacecraft Position
# =============================================================================
def load_lexi_position(time_range):
    df = pd.read_csv(LEXI_POSITION_FILE)
    df["Epoch"] = pd.to_datetime(df["Epoch"], utc=True)
    df.set_index("Epoch", inplace=True)
    df.sort_index(inplace=True)

    t0 = pd.to_datetime(time_range[0], utc=True)
    t1 = pd.to_datetime(time_range[1], utc=True)
    df = df[(df.index >= t0) & (df.index <= t1)]

    return df


# =============================================================================
# 4. Mean Variance Analysis
# =============================================================================
def perform_mva(B_data):
    valid = ~np.isnan(B_data).any(axis=1)
    B = B_data[valid]
    if len(B) < 3:
        return None

    B_mean = B.mean(axis=0)
    cov = np.cov((B - B_mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    normal = eigenvectors[:, 0]
    ratio = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else np.inf

    return {
        "normal": normal,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "variance_ratio": ratio,
        "B_mean": B_mean,
    }


# =============================================================================
# 5. Propagation Delay
# =============================================================================
def propagation_delay(pos_themis, pos_lexi, velocity, normal):
    dr = pos_lexi - pos_themis
    v_n = np.dot(velocity, normal)
    if abs(v_n) < 1e-6:
        return np.nan
    return np.dot(dr, normal) / v_n


# =============================================================================
# 6. Main
# =============================================================================
def main():
    themis_df = load_themis_data(TIME_RANGE, sc=SPACECRAFT)
    sc = SPACECRAFT

    lexi_df = load_lexi_counts(TIME_RANGE)
    pos_df = load_lexi_position(TIME_RANGE)

    t0 = pd.to_datetime(TIME_RANGE[0], utc=True)
    t1 = pd.to_datetime(TIME_RANGE[1], utc=True)
    themis_df = themis_df[(themis_df.index >= t0) & (themis_df.index <= t1)]
    lexi_df = lexi_df[(lexi_df.index >= t0) & (lexi_df.index <= t1)]
    pos_df = pos_df[(pos_df.index >= t0) & (pos_df.index <= t1)]

    b_cols = [f"th{sc}_fgs_gse_{c}" for c in "xyz"]
    B = themis_df[b_cols].dropna().values
    mva = perform_mva(B)

    if mva is None:
        return

    vel_cols = [f"th{sc}_velocity_gse_{c}" for c in "xyz"]
    pos_themis_cols = [f"th{sc}_pos_gse_{c}" for c in "xyz"]
    pos_lexi_cols = ["lexi_sc_pos_gse_x", "lexi_sc_pos_gse_y", "lexi_sc_pos_gse_z"]

    common_times = themis_df.index.intersection(pos_df.index)
    delays, delay_times = [], []

    for t in common_times:
        try:
            p_th = themis_df.loc[t, pos_themis_cols].values.astype(float)
            v = themis_df.loc[t, vel_cols].values.astype(float)
            p_lx = pos_df.loc[t, pos_lexi_cols].values.astype(float)
        except (KeyError, TypeError):
            continue

        if np.isnan(p_th).any() or np.isnan(v).any() or np.isnan(p_lx).any():
            continue

        dt = propagation_delay(p_th, p_lx, v, mva["normal"])
        delays.append(dt)
        delay_times.append(t)

    delay_df = pd.DataFrame({"delay_seconds": delays}, index=delay_times)

    assigned_delays = []
    for lexi_t in lexi_df.index:
        if lexi_t in delay_df.index:
            assigned_delays.append(delay_df.loc[lexi_t, "delay_seconds"])
        else:
            idx = delay_df.index.get_indexer(
                [lexi_t], method="nearest", tolerance=pd.Timedelta("2min")
            )
            if idx[0] != -1:
                assigned_delays.append(delay_df.iloc[idx[0]]["delay_seconds"])
            else:
                assigned_delays.append(np.nan)
    lexi_df["propagation_delay_s"] = assigned_delays

    flux_col = f"th{sc}_ion_flux_mag"

    plt.style.use("dark_background")

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 1, hspace=0.3)

    t_start = pd.to_datetime(TIME_RANGE[0], utc=True)
    t_end = pd.to_datetime(TIME_RANGE[1], utc=True)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        themis_df.index, themis_df[flux_col],
        s=40, alpha=0.9, color="#00BFFF", edgecolors="white",
        linewidths=0.3, label=f"THEMIS-C Ion Flux ({flux_col})",
    )
    ax1.set_ylabel("THEMIS-C Flux\n[cm⁻²·s⁻¹]", fontsize=10)
    ax1.set_title(
        "THEMIS-C Flux & LEXI Counts/s – MVA Planar Propagation Analysis",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(t_start, t_end)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.scatter(
        lexi_df.index, lexi_df["counts_per_second"],
        s=40, alpha=0.9, color="#00FF7F", edgecolors="white",
        linewidths=0.3, label="LEXI Counts/s (Original)",
    )
    ax2.set_ylabel("LEXI Counts/s", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    valid = ~lexi_df["propagation_delay_s"].isna()
    shifted_times = (
        lexi_df.index[valid]
        - pd.to_timedelta(lexi_df["propagation_delay_s"][valid], unit="s")
    )
    shifted_vals = lexi_df["counts_per_second"][valid]
    delay_colors = lexi_df["propagation_delay_s"][valid]

    scatter = ax3.scatter(
        shifted_times, shifted_vals,
        c=delay_colors, cmap="plasma", s=50, edgecolors="white",
        linewidths=0.3, label="LEXI Counts/s (Time-aligned)",
    )
    plt.colorbar(scatter, ax=ax3, label="Propagation Delay [s]")
    ax3.set_ylabel("LEXI Counts/s\n(Time-aligned)", fontsize=10)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.2)

    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.scatter(
        delay_df.index, delay_df["delay_seconds"],
        s=40, alpha=0.9, color="#FF6347", edgecolors="white",
        linewidths=0.3, label="Propagation Delay",
    )
    ax4.axhline(y=0, color="white", linestyle="--", alpha=0.3)
    ax4.set_ylabel("Time Delay [s]", fontsize=10)
    ax4.set_xlabel("Time [UTC]", fontsize=10)
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.2)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    fig.autofmt_xdate()

    FIGURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_SAVE_DIR / "themis_lexi_planar_propagation_analysis_v2.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    output_dir = Path("../data/mva_aligned_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "lexi_planar_propagation_aligned.csv"
    lexi_df.to_csv(output_file)

    return lexi_df, delay_df, themis_df, pos_df, mva, fig


if __name__ == "__main__":
    lexi_df, delay_df, themis_df, pos_df, mva, fig = main()