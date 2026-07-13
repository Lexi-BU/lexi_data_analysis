"""
THEMIS-LEXI Planar Propagation Analysis

This script performs the following:
1. Downloads THEMIS-C magnetic field and plasma (MOM) data for
   2025-03-16 19:30:00 to 2025-03-16 21:15:00.
2. Loads LEXI count data from CSV and converts to counts per second.
3. Loads LEXI spacecraft position data.
4. Performs Mean Variance Analysis (MVA) on the THEMIS-C magnetic field to
   determine the normal direction of the planar solar-wind structure.
5. Calculates the propagation delay between THEMIS-C and LEXI.
6. Plots original and time-aligned LEXI data alongside THEMIS-C flux,
   with delay color-coding, and saves to disk.

NOTE: This script does NOT modify any existing files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cdflib


# =============================================================================
# Configuration
# =============================================================================
TIME_RANGE = ["2025-03-16 19:30:00", "2025-03-16 21:15:00"]
SPACECRAFT = "c"

LEXI_DATA_FILE = Path(
    "../data/line_profile_data/bg_corrected/from_l2/"
    "line_profile_fit_parameters_bg_corrected_1min.csv"
)
LEXI_HOUSEKEEPING_FILE = Path(
    "../data/lexi_hk_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00_v0.0.csv"
)
LEXI_POSITION_FILE = Path(
    "../data/lexi_themis_analysis/lexi_themis_c_analysis_lexi_spacecraft.csv"
)
THEMIS_SAVE_DIR = Path("../data/downloaded_themis_c_data/")
FIGURE_SAVE_DIR = Path("../figures/lexi_themis_c_analysis/planar_propagation/")


# =============================================================================
# 1. Download / Load THEMIS-C Data
# =============================================================================
def load_local_themis_data(time_range):
    """
    Loads THEMIS-C data from local ISTP CDF files.
    - MOM: thc_peem_epoch, thc_peem_flux, thc_peim_flux, thc_peem_velocity_gse, thc_peim_velocity_gse
    - FGM: thc_fgs_epoch, thc_fgs_gse
    - SSC: Epoch, XYZ_GSE
    """
    fgm_file = THEMIS_SAVE_DIR / "thc_l2s_fgm_20250316193001_20250316211457_cdaweb.cdf"
    mom_file = THEMIS_SAVE_DIR / "thc_l2s_mom_20250316193000_20250316211459_cdaweb.cdf"
    ssc_file = THEMIS_SAVE_DIR / "thc_ors_ssc_20250316193000_20250316211500_cdaweb.cdf"

    if not (fgm_file.exists() and mom_file.exists() and ssc_file.exists()):
        raise FileNotFoundError(f"Missing CDF files in {THEMIS_SAVE_DIR}")

    print("Loading THEMIS-C FGM CDF data...")
    with cdflib.CDF(str(fgm_file)) as fgm:
        fgm_epochs = fgm.varget("thc_fgs_epoch")
        fgm_bfield = fgm.varget("thc_fgs_gse")
    
    fgm_times = pd.to_datetime(cdflib.cdfepoch.to_datetime(fgm_epochs), utc=True)
    fgm_df = pd.DataFrame(
        fgm_bfield, 
        index=fgm_times,
        columns=["thc_fgs_gse_x", "thc_fgs_gse_y", "thc_fgs_gse_z"]
    )
    fgm_df["thc_fgs_gse_mag"] = np.linalg.norm(fgm_df.values, axis=1)

    print("Loading THEMIS-C MOM CDF data...")
    with cdflib.CDF(str(mom_file)) as mom:
        # Ions
        try:
            ion_epochs = mom.varget("thc_peim_epoch")
            ion_flux = mom.varget("thc_peim_flux")
            ion_vel = mom.varget("thc_peim_velocity_gse")
        except:
            ion_epochs, ion_flux, ion_vel = None, None, None

        # Electrons
        try:
            elec_epochs = mom.varget("thc_peem_epoch")
            elec_flux = mom.varget("thc_peem_flux")
            elec_vel = mom.varget("thc_peem_velocity_gse")
        except:
            elec_epochs, elec_flux, elec_vel = None, None, None

    mom_dfs = []
    if ion_epochs is not None and ion_flux is not None and ion_vel is not None:
        ion_times = pd.to_datetime(cdflib.cdfepoch.to_datetime(ion_epochs), utc=True)
        ion_df = pd.DataFrame(index=ion_times)
        ion_df["thc_peim_flux_x"] = ion_flux[:, 0]
        ion_df["thc_peim_flux_y"] = ion_flux[:, 1]
        ion_df["thc_peim_flux_z"] = ion_flux[:, 2]
        ion_df["thc_ion_flux_mag"] = np.linalg.norm(ion_flux, axis=1)
        ion_df["thc_velocity_gse_x"] = ion_vel[:, 0]
        ion_df["thc_velocity_gse_y"] = ion_vel[:, 1]
        ion_df["thc_velocity_gse_z"] = ion_vel[:, 2]
        mom_dfs.append(ion_df)
    
    if elec_epochs is not None and elec_flux is not None and elec_vel is not None:
        elec_times = pd.to_datetime(cdflib.cdfepoch.to_datetime(elec_epochs), utc=True)
        elec_df = pd.DataFrame(index=elec_times)
        elec_df["thc_peef_flux_x"] = elec_flux[:, 0]
        elec_df["thc_peef_flux_y"] = elec_flux[:, 1]
        elec_df["thc_peef_flux_z"] = elec_flux[:, 2]
        elec_df["thc_elec_flux_mag"] = np.linalg.norm(elec_flux, axis=1)
        elec_df["thc_elec_velocity_gse_x"] = elec_vel[:, 0]
        elec_df["thc_elec_velocity_gse_y"] = elec_vel[:, 1]
        elec_df["thc_elec_velocity_gse_z"] = elec_vel[:, 2]
        # Only add velocity from electrons if we don't already have it
        if "thc_velocity_gse_x" not in (mom_dfs[0].columns if len(mom_dfs)>0 else []):
            elec_df["thc_velocity_gse_x"] = elec_vel[:, 0]
            elec_df["thc_velocity_gse_y"] = elec_vel[:, 1]
            elec_df["thc_velocity_gse_z"] = elec_vel[:, 2]
        mom_dfs.append(elec_df)

    print("Loading THEMIS-C SSC Position CDF data...")
    with cdflib.CDF(str(ssc_file)) as ssc:
        ssc_epochs = ssc.varget("Epoch")
        ssc_pos = ssc.varget("XYZ_GSE")
        
    ssc_times = pd.to_datetime(cdflib.cdfepoch.to_datetime(ssc_epochs), utc=True)
    ssc_pos_km = ssc_pos * 6371.2 # Convert Earth Radii to km
    pos_df = pd.DataFrame(
        ssc_pos_km,
        index=ssc_times,
        columns=["thc_pos_gse_x", "thc_pos_gse_y", "thc_pos_gse_z"]
    )

    mom_df = pd.concat(mom_dfs, axis=1).sort_index() if len(mom_dfs) > 0 else pd.DataFrame()

    fgm_df = fgm_df[~fgm_df.index.duplicated(keep="first")]
    mom_df = mom_df[~mom_df.index.duplicated(keep="first")]
    pos_df = pos_df[~pos_df.index.duplicated(keep="first")]

    t0 = pd.to_datetime(time_range[0], utc=True)
    t1 = pd.to_datetime(time_range[1], utc=True)

    mom_df = mom_df[(mom_df.index >= t0) & (mom_df.index <= t1)]
    fgm_df = fgm_df[(fgm_df.index >= t0) & (fgm_df.index <= t1)]
    pos_df = pos_df[(pos_df.index >= t0) & (pos_df.index <= t1)]

    return mom_df, fgm_df, pos_df


# =============================================================================
# 2. Load LEXI Data
# =============================================================================
def load_lexi_counts(time_range):
    """
    Load LEXI background-corrected counts and convert to counts/second.

    The CSV has 1-minute bins (start_time → end_time).  We use the midpoint
    as the timestamp and divide counts by 60 to get counts per second.
    """
    print(f"Loading LEXI count data from {LEXI_DATA_FILE} ...")
    df = pd.read_csv(LEXI_DATA_FILE)

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    df["mid_time"] = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
    df.set_index("mid_time", inplace=True)
    df.sort_index(inplace=True)

    # Convert counts to counts per second (1-minute bins → divide by 60)
    df["counts_per_second"] = (
        df["background_corrected_total_hist_counts"] / 60.0
    )

    # Filter to time range
    t0 = pd.to_datetime(time_range[0], utc=True)
    t1 = pd.to_datetime(time_range[1], utc=True)
    df = df[(df.index >= t0) & (df.index <= t1)]

    print(f"  {len(df)} LEXI observations in time window")
    return df


def load_lexi_housekeeping(time_range):
    print(f"Loading LEXI housekeeping data from {LEXI_HOUSEKEEPING_FILE} ...")
    df = pd.read_csv(LEXI_HOUSEKEEPING_FILE)
    df["Epoch"] = pd.to_datetime(df["Epoch"], utc=True, format="mixed")
    df.set_index("Epoch", inplace=True)
    df.sort_index(inplace=True)

    t0 = pd.to_datetime(time_range[0], utc=True)
    t1 = pd.to_datetime(time_range[1], utc=True)
    df = df[(df.index >= t0) & (df.index <= t1)]

    # Only store the index and "all_counts" columns
    df = df[["DeltaEvntCount"]]
    print(f"  {len(df)} LEXI housekeeping observations in time window")
    return df

# =============================================================================
# 3. Load LEXI Spacecraft Position
# =============================================================================
def load_lexi_position(time_range):
    """Load LEXI spacecraft GSE position from CSV."""
    print(f"Loading LEXI spacecraft position from {LEXI_POSITION_FILE} ...")
    df = pd.read_csv(LEXI_POSITION_FILE)
    df["Epoch"] = pd.to_datetime(df["Epoch"], utc=True)
    df.set_index("Epoch", inplace=True)
    df.sort_index(inplace=True)

    t0 = pd.to_datetime(time_range[0], utc=True)
    t1 = pd.to_datetime(time_range[1], utc=True)
    df = df[(df.index >= t0) & (df.index <= t1)]

    print(f"  {len(df)} position records in time window")
    return df


# =============================================================================
# 4. Mean Variance Analysis
# =============================================================================
def perform_mva(B_data):
    """
    Minimum Variance Analysis on magnetic field vectors.

    Parameters
    ----------
    B_data : np.ndarray, shape (N, 3)
        Bx, By, Bz in GSE.

    Returns
    -------
    dict  with keys: normal, eigenvalues, eigenvectors, variance_ratio, B_mean
    """
    valid = ~np.isnan(B_data).any(axis=1)
    B = B_data[valid]
    if len(B) < 3:
        return None

    B_mean = B.mean(axis=0)
    cov = np.cov((B - B_mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # returns sorted ascending

    normal = eigenvectors[:, 0]  # minimum-variance direction
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
    """
    Time delay for a planar structure travelling from THEMIS to LEXI.

    dt = (r_lexi − r_themis) · n  /  (V · n)

    Positive dt  →  LEXI observes *after* THEMIS.
    """
    dr = pos_lexi - pos_themis
    v_n = np.dot(velocity, normal)
    if abs(v_n) < 1e-6:
        return np.nan
    return np.dot(dr, normal) / v_n


# =============================================================================
# 6. Main
# =============================================================================
def main():
    # ------------------------------------------------------------------
    # Step 1 – THEMIS data
    # ------------------------------------------------------------------
    mom_df, fgm_df, th_pos_df = load_local_themis_data(TIME_RANGE)

    sc = SPACECRAFT

    # ------------------------------------------------------------------
    # Step 2 – LEXI counts (counts / second)
    # ------------------------------------------------------------------
    lexi_hk_df = load_lexi_housekeeping(TIME_RANGE)
    lexi_df = load_lexi_counts(TIME_RANGE)
    # Make a rolling average of the housekeeping data
    lexi_hk_df["all_counts_rolling"] = lexi_hk_df["DeltaEvntCount"].rolling(window="5min", center=True).mean()
    # Merge the two dataframes using lexi_df as the primary dataframe
    # and merge by finding the nearest index in lexi_hk_df
    lexi_df = pd.merge_asof(
        lexi_df, lexi_hk_df,
        left_index=True, right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("2min")
    )
    # ------------------------------------------------------------------
    # Step 3 – LEXI spacecraft position
    # ------------------------------------------------------------------
    lexi_pos_df = load_lexi_position(TIME_RANGE)

    print(f"\nLoaded: MOM={len(mom_df)}, FGM={len(fgm_df)}, THEMIS Pos={len(th_pos_df)}, "
          f"LEXI={len(lexi_df)}, LEXI Position={len(lexi_pos_df)}")

    # ------------------------------------------------------------------
    # Step 4 – MVA
    # ------------------------------------------------------------------
    print("\nPerforming Mean Variance Analysis …")
    b_cols = [f"th{sc}_fgs_gse_{c}" for c in "xyz"]
    # Provide the high-cadence FGM data for accurate MVA
    B = fgm_df[b_cols].dropna().values
    mva = perform_mva(B)

    if mva is None:
        print("  MVA failed – not enough valid B-field data.")
        return

    print(f"  Normal vector     : {mva['normal']}")
    print(f"  Eigenvalues       : {mva['eigenvalues']}")
    print(f"  λ_mid / λ_min     : {mva['variance_ratio']:.2f}")
    print(f"  Mean B            : {mva['B_mean']} nT")

    # ------------------------------------------------------------------
    # Step 5 – Calculate delays
    # ------------------------------------------------------------------
    print("\nCalculating propagation delays …")

    # Determine which velocity columns are available
    vel_cols = [f"th{sc}_velocity_gse_{c}" for c in "xyz"]
    if not all(c in mom_df.columns for c in vel_cols):
        print("  ERROR: No velocity columns found in THEMIS data.")
        print(f"  Available columns: {list(mom_df.columns)}")
        return

    pos_themis_cols = [f"th{sc}_pos_gse_{c}" for c in "xyz"]
    pos_lexi_cols = ["lexi_sc_pos_gse_x", "lexi_sc_pos_gse_y", "lexi_sc_pos_gse_z"]

    common_times = lexi_df.index
    delays, delay_times, angles = [], [], []
    p_sun = np.array([149597870.0, 0.0, 0.0]) # Sun position in GSE (km)

    # Get values interpolated to nearest LEXI timestamp
    v_aligned = mom_df[vel_cols].reindex(common_times, method="nearest", tolerance=pd.Timedelta("1min"))
    thp_aligned = th_pos_df[pos_themis_cols].reindex(common_times, method="nearest", tolerance=pd.Timedelta("1min"))
    lxp_aligned = lexi_pos_df[pos_lexi_cols].reindex(common_times, method="nearest", tolerance=pd.Timedelta("1min"))

    for t in common_times:
        try:
            p_th = thp_aligned.loc[t].values.astype(float)
            v = v_aligned.loc[t].values.astype(float)
            p_lx = lxp_aligned.loc[t].values.astype(float)
        except (KeyError, TypeError) as e:
            continue

        if np.isnan(p_th).any() or np.isnan(v).any() or np.isnan(p_lx).any():
            continue

        dt = propagation_delay(p_th, p_lx, v, mva["normal"])
        delays.append(dt)
        delay_times.append(t)

        # Angle as seen from LEXI (vertex = p_lx)
        v_lx_sun = p_sun - p_lx
        v_lx_th = p_th - p_lx
        norm_lx_sun = np.linalg.norm(v_lx_sun)
        norm_lx_th = np.linalg.norm(v_lx_th)
        if norm_lx_sun > 0 and norm_lx_th > 0:
            cos_theta = np.dot(v_lx_sun, v_lx_th) / (norm_lx_sun * norm_lx_th)
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        else:
            angle_deg = np.nan
        angles.append(angle_deg)

    delay_df = pd.DataFrame({"delay_seconds": delays, "sun_angle_deg": angles}, index=delay_times)
    
    if len(delays) > 0:
        print(f"  {len(delays)} delay values computed")
        print(f"  Mean delay : {np.nanmean(delay_df['delay_seconds']):.1f} s")
        print(f"  Range      : [{np.nanmin(delay_df['delay_seconds']):.1f}, "
              f"{np.nanmax(delay_df['delay_seconds']):.1f}] s")
    else:
        print("  0 delay values computed! Could not align matching timestamps via nearest neighbor.")

    # ------------------------------------------------------------------
    # Step 6 – Assign delay to each LEXI observation
    # ------------------------------------------------------------------
    print("\nAligning LEXI data …")
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

    # ------------------------------------------------------------------
    # Step 7 – Plot
    # ------------------------------------------------------------------
    print("\nGenerating plots …")

    # Determine which flux column to use
    flux_col = f"th{sc}_ion_flux_mag"
    if flux_col not in mom_df.columns:
        flux_col = f"th{sc}_elec_flux_mag"
    if flux_col not in mom_df.columns:
        print("  WARNING: No flux column found – skipping THEMIS flux panel.")
        flux_col = None

    plt.style.use("dark_background")

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 1, hspace=0.3)

    t_start = pd.to_datetime(TIME_RANGE[0], utc=True)
    t_end = pd.to_datetime(TIME_RANGE[1], utc=True)

    # ---- Panel 1: THEMIS-C flux -------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    flux_val_1min = mom_df[flux_col].resample("1min").mean()
    if flux_col is not None:
        ax1.scatter(
            flux_val_1min.index, flux_val_1min,
            s=40, alpha=0.9, color="#00BFFF", edgecolors="white",
            linewidths=0.3, label=f"THEMIS-C Flux ({flux_col})",
        )
    ax1.set_ylabel("THEMIS-C Flux\n[cm⁻²·s⁻¹]", fontsize=10)
    ax1.set_title(
        "THEMIS-C Flux & LEXI Counts/s – MVA Planar Propagation Analysis",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(t_start, t_end)

    # ---- Panel 2: LEXI original counts/s ----------------------------------
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.scatter(
        lexi_df.index, lexi_df["all_counts_rolling"],
        s=100, alpha=0.8, color="#00FF7F", edgecolors="white",
        linewidths=0.3, label="LEXI Counts/s (Original)",
    )
    # ax2.set_ylim(2000, 3400)
    ax2.set_ylim(200, 600)
    ax2.set_ylabel("LEXI Counts/s", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    # ---- Panel 3: LEXI aligned, color-coded by delay ---------------------
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    valid = ~lexi_df["propagation_delay_s"].isna()
    shifted_times = (
        lexi_df.index[valid]
        - pd.to_timedelta(lexi_df["propagation_delay_s"][valid], unit="s")
    )
    shifted_vals = lexi_df["all_counts_rolling"][valid]
    delay_colors = lexi_df["propagation_delay_s"][valid]

    scatter = ax3.scatter(
        shifted_times, shifted_vals,
        c=delay_colors, cmap="plasma", s=100, edgecolors="white", alpha=0.8,
        linewidths=0.3, # label="LEXI Counts/s (Time-aligned)",
    )
    cax3 = ax3.inset_axes([0, 1.05, 1, 0.05])
    plt.colorbar(scatter, cax=cax3, orientation="horizontal", label="Propagation Delay [s]")
    cax3.xaxis.set_ticks_position("top")
    cax3.xaxis.set_label_position("top")
    # ax3.set_ylim(2000, 3400)
    ax3.set_ylim(200, 600)
    ax3.set_ylabel("LEXI Counts/s\n(Time-aligned)", fontsize=10)
    # ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.2)

    # Twin axis for the angle between Sun and THEMIS (as seen from LEXI)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        delay_df.index, delay_df["sun_angle_deg"],
        color="#00BFFF", linewidth=1.5, alpha=0.8, linestyle="--", label="Angle from LEXI"
    )
    ax3_twin.set_ylabel("Sun-LEXI-THEMIS Angle [deg]", fontsize=10, color="#00BFFF")
    ax3_twin.tick_params(axis="y", colors="#00BFFF")
    ax3_twin.legend(loc="lower right")

    # Find the distance between LEXI and THEMIS-C
    distance = np.sqrt(
        (lexi_pos_df["lexi_sc_pos_gse_x"] - th_pos_df["thc_pos_gse_x"])**2 +
        (lexi_pos_df["lexi_sc_pos_gse_y"] - th_pos_df["thc_pos_gse_y"])**2 +
        (lexi_pos_df["lexi_sc_pos_gse_z"] - th_pos_df["thc_pos_gse_z"])**2
    )
    # ---- Panel 4: Time delay over time ------------------------------------
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax5 = ax4.twinx()
    ax4.scatter(
        delay_df.index, delay_df["delay_seconds"],
        s=40, alpha=0.9, color="#FF6347", edgecolors="white",
        linewidths=0.3, label="Propagation Delay",
    )
    ax5.scatter(
        lexi_pos_df.index, distance,
        s=40, alpha=0.9, color="#00BFFF", edgecolors="white",
        linewidths=0.3, label="Distance",
    )
    ax4.axhline(y=0, color="white", linestyle="--", alpha=0.3)
    ax4.set_ylabel("Time Delay [s]", fontsize=10)
    ax4.set_xlabel("Time [UTC]", fontsize=10)
    ax5.set_ylabel("Distance [km]", fontsize=10)
    ax4.legend(loc="lower left")
    ax5.legend(loc="lower right")
    ax4.grid(True, alpha=0.2)

    # Hide intermediate x-tick labels
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    fig.autofmt_xdate()

    # ---- Save & close -----------------------------------------------------
    FIGURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_SAVE_DIR / "themis_lexi_planar_propagation_analysis_all_counts.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figure saved → {fig_path}")
    plt.close(fig)

    # ---- Save aligned data ------------------------------------------------
    output_dir = Path("../data/mva_aligned_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "lexi_planar_propagation_aligned.csv"
    lexi_df.to_csv(output_file)
    print(f"  Aligned data saved → {output_file}")

    print("\nDone.")

    return lexi_df, delay_df, mom_df, fgm_df, th_pos_df, lexi_pos_df, mva, fig


if __name__ == "__main__":
    lexi_df, delay_df, mom_df, fgm_df, th_pos_df, lexi_pos_df, mva, fig = main()
