"""
THEMIS-LEXI Mean Variance Analysis for Planar Propagation

This script performs mean variance analysis (MVA) to determine the time delay
between solar wind observations at THEMIS-C and the corresponding response at LEXI,
assuming planar propagation of solar wind structures.

Task breakdown:
1. Download THEMIS-C magnetic field and plasma data (2025-03-16 19:30 to 21:30)
2. Load LEXI count data from CSV
3. Apply MVA to determine planar propagation normal vector
4. Calculate time delay based on spacecraft positions and propagation velocity
5. Plot original and time-aligned data with color-coded delays
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspedas
from pytplot import get_data


def download_themis_data(time_range, spacecraft="c", save_dir="../data/themis_mva_data/"):
    """
    Download THEMIS magnetic field and plasma data.
    
    Parameters
    ----------
    time_range : list of str
        [start_time, end_time] in format "YYYY-MM-DD HH:MM:SS"
    spacecraft : str, optional
        THEMIS spacecraft identifier ('a', 'b', 'c', 'd', or 'e'), by default "c"
    save_dir : str, optional
        Directory to save downloaded data, by default "../data/themis_mva_data/"
        
    Returns
    -------
    pd.DataFrame
        Combined dataframe with magnetic field, plasma, and position data
    """
    sc = spacecraft
    
    print(f"Downloading THEMIS-{sc.upper()} data for {time_range[0]} to {time_range[1]}...")
    
    # Download magnetic field data (FGM)
    print("  - Magnetic field data (FGM)...")
    fgm = pyspedas.themis.fgm(
        probe=sc,
        level="l2",
        trange=time_range,
        varnames=[f"th{sc}_fgs_gse"],  # Field in GSE coordinates
        no_update=False,
    )
    
    # Download ion moments data (MOM) - for ion particle flux vector
    print("  - Ion moments data (MOM)...")
    mom = pyspedas.themis.mom(
        probe=sc,
        level="l2",
        trange=time_range,
        no_update=False,
    )
    
    # Download plasma data (ESA) - for electron data
    print("  - Plasma data (ESA)...")
    esa = pyspedas.themis.esa(
        probe=sc,
        level="l2",
        trange=time_range,
        varnames=[
            f"th{sc}_peef_density",
            f"th{sc}_peef_avgtemp",
            f"th{sc}_peef_velocity_gse",
        ],
        no_update=False,
    )
    
    # Download spacecraft position
    print("  - Spacecraft position (SSC)...")
    pos = pyspedas.themis.ssc(
        probe=sc,
        trange=time_range,
        level="l2",
        varnames=["XYZ_GSE"],
        no_update=False,
    )
    
    series_list = []
    
    # Extract magnetic field data
    fgm_data = get_data(f"th{sc}_fgs_gse")
    if fgm_data is not None:
        times = pd.to_datetime(fgm_data.times, unit="s", utc=True)
        indices = ~pd.Series(times).duplicated(keep="first")
        for i, comp in enumerate(["x", "y", "z"]):
            col_name = f"th{sc}_fgs_gse_{comp}"
            series = pd.Series(fgm_data.y[indices, i], index=times[indices], name=col_name)
            series_list.append(series)
    
    # Extract ion particle flux vector from MOM
    ion_flux_data = get_data(f"th{sc}_peim_flux")
    if ion_flux_data is not None:
        times = pd.to_datetime(ion_flux_data.times, unit="s", utc=True)
        indices = ~pd.Series(times).duplicated(keep="first")
        for i, comp in enumerate(["x", "y", "z"]):
            col_name = f"th{sc}_peim_flux_{comp}"
            series = pd.Series(ion_flux_data.y[indices, i], index=times[indices], name=col_name)
            series_list.append(series)
    
    # Extract plasma velocity data (electron)
    vel_data = get_data(f"th{sc}_peef_velocity_gse")
    if vel_data is not None:
        times = pd.to_datetime(vel_data.times, unit="s", utc=True)
        indices = ~pd.Series(times).duplicated(keep="first")
        for i, comp in enumerate(["x", "y", "z"]):
            col_name = f"th{sc}_peef_velocity_gse_{comp}"
            series = pd.Series(vel_data.y[indices, i], index=times[indices], name=col_name)
            series_list.append(series)
    
    # Extract plasma density
    dens_data = get_data(f"th{sc}_peef_density")
    if dens_data is not None:
        times = pd.to_datetime(dens_data.times, unit="s", utc=True)
        series = pd.Series(dens_data.y, index=times, name=f"th{sc}_peef_density")
        series = series[~series.index.duplicated(keep="first")]
        series_list.append(series)
    
    # Extract plasma temperature
    temp_data = get_data(f"th{sc}_peef_avgtemp")
    if temp_data is not None:
        times = pd.to_datetime(temp_data.times, unit="s", utc=True)
        series = pd.Series(temp_data.y, index=times, name=f"th{sc}_peef_avgtemp")
        series = series[~series.index.duplicated(keep="first")]
        series_list.append(series)
    
    # Extract spacecraft position
    spc_data = get_data("XYZ_GSE")
    if spc_data is not None:
        times = pd.to_datetime(spc_data.times, unit="s", utc=True)
        indices = ~pd.Series(times).duplicated(keep="first")
        for i, comp in enumerate(["x", "y", "z"]):
            col_name = f"th{sc}_pos_gse_{comp}"
            series = pd.Series(spc_data.y[indices, i], index=times[indices], name=col_name)
            series_list.append(series)
    
    # Combine all data
    df = pd.concat(series_list, axis=1)
    df.sort_index(inplace=True)
    
    # Calculate ion flux magnitude from ion particle flux vector
    if f"th{sc}_peim_flux_x" in df.columns:
        df[f"th{sc}_ion_flux_mag"] = np.sqrt(
            df[f"th{sc}_peim_flux_x"]**2 + 
            df[f"th{sc}_peim_flux_y"]**2 + 
            df[f"th{sc}_peim_flux_z"]**2
        )
        # Use ion flux magnitude as the main flux
        df[f"th{sc}_peef_flux"] = df[f"th{sc}_ion_flux_mag"]
    else:
        # Fallback to electron flux if ion flux not available
        # Calculate velocity magnitude
        df[f"th{sc}_peef_velocity_gse_mag"] = np.sqrt(
            df[f"th{sc}_peef_velocity_gse_x"]**2 + 
            df[f"th{sc}_peef_velocity_gse_y"]**2 + 
            df[f"th{sc}_peef_velocity_gse_z"]**2
        )
        
        # Calculate flux (density * velocity magnitude)
        df[f"th{sc}_peef_flux"] = df[f"th{sc}_peef_density"] * df[f"th{sc}_peef_velocity_gse_mag"]
    
    # Calculate magnetic field magnitude
    df[f"th{sc}_fgs_gse_mag"] = np.sqrt(
        df[f"th{sc}_fgs_gse_x"]**2 + 
        df[f"th{sc}_fgs_gse_y"]**2 + 
        df[f"th{sc}_fgs_gse_z"]**2
    )
    
    # Resample to 1-minute resolution
    tz = df.index.tz
    t0 = df.index.min().ceil("min")
    t1 = df.index.max().floor("min")
    target_idx = pd.date_range(t0, t1, freq="1min", tz=tz)
    
    df = df.reindex(target_idx, method="nearest", tolerance=pd.Timedelta("30s"))
    df = df.interpolate(method="time", limit=5)
    
    # Save data
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = save_path / f"themis_{sc}_mva_data_{time_range[0].replace(' ', 'T').replace(':', '-')}_to_{time_range[1].replace(' ', 'T').replace(':', '-')}.csv"
    df.to_csv(filename)
    print(f"Data saved to {filename}")
    
    return df


def perform_mva(B_data):
    """
    Perform Minimum Variance Analysis on magnetic field data.
    
    MVA finds the direction of minimum variance in the magnetic field,
    which corresponds to the normal direction of a planar structure.
    
    Parameters
    ----------
    B_data : np.ndarray
        Magnetic field data array of shape (N, 3) with [Bx, By, Bz] components
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'normal': Normal vector (minimum variance direction)
        - 'eigenvalues': Eigenvalues [lambda_min, lambda_mid, lambda_max]
        - 'eigenvectors': All three eigenvectors
        - 'variance_ratio': lambda_mid / lambda_min (quality metric)
    """
    # Remove NaN values
    valid_mask = ~np.isnan(B_data).any(axis=1)
    B_clean = B_data[valid_mask]
    
    if len(B_clean) < 3:
        return None
    
    # Calculate mean field
    B_mean = np.mean(B_clean, axis=0)
    
    # Calculate covariance matrix
    B_centered = B_clean - B_mean
    cov_matrix = np.cov(B_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Minimum variance direction is the normal to the structure
    normal = eigenvectors[:, 0]
    
    # Variance ratio as quality metric (should be >> 1 for good planar structure)
    variance_ratio = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else np.inf
    
    return {
        'normal': normal,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'variance_ratio': variance_ratio,
        'B_mean': B_mean,
    }


def calculate_propagation_delay(pos_themis, pos_lexi, velocity, normal):
    """
    Calculate time delay for planar structure propagation between two spacecraft.
    
    Under the assumption of planar propagation with velocity V and normal vector n,
    the delay is: dt = (r_lexi - r_themis) · n / (V · n)
    
    Parameters
    ----------
    pos_themis : np.ndarray
        THEMIS position vector [x, y, z] in GSE (km)
    pos_lexi : np.ndarray
        LEXI position vector [x, y, z] in GSE (km)
    velocity : np.ndarray
        Plasma bulk velocity vector [vx, vy, vz] in GSE (km/s)
    normal : np.ndarray
        Normal vector to the planar structure
        
    Returns
    -------
    float
        Time delay in seconds (positive means LEXI observes after THEMIS)
    """
    # Position difference vector (km)
    dr = pos_lexi - pos_themis
    
    # Normal component of separation
    dr_normal = np.dot(dr, normal)
    
    # Normal component of velocity
    v_normal = np.dot(velocity, normal)
    
    # Avoid division by zero
    if abs(v_normal) < 1e-6:
        return np.nan
    
    # Time delay in seconds
    delay = dr_normal / v_normal
    
    return delay


def main():
    """Main execution function."""
    
    # -------------------------------------------------------------------------
    # 1. DOWNLOAD THEMIS-C DATA
    # -------------------------------------------------------------------------
    time_range = ["2025-03-16 19:30:00", "2025-03-16 20:50:00"]
    
    # Check if data already exists
    data_file = Path(f"../data/themis_mva_data/themis_c_mva_data_2025-03-16T19-30-00_to_2025-03-16T20-50-00.csv")
    
    if data_file.exists():
        print(f"Loading existing THEMIS data from {data_file}")
        themis_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        themis_df.index = pd.to_datetime(themis_df.index, utc=True)
    else:
        themis_df = download_themis_data(time_range, spacecraft="c")
    
    # -------------------------------------------------------------------------
    # 2. LOAD LEXI DATA
    # -------------------------------------------------------------------------
    print("\nLoading LEXI data...")
    lexi_file = Path("../data/line_profile_data/bg_corrected/from_l2/line_profile_fit_parameters_bg_corrected_1min.csv")
    lexi_df = pd.read_csv(lexi_file)
    
    # Process LEXI time columns
    lexi_df["start_time"] = pd.to_datetime(lexi_df["start_time"], utc=True)
    lexi_df["end_time"] = pd.to_datetime(lexi_df["end_time"], utc=True)
    lexi_df["mid_time"] = lexi_df["start_time"] + (lexi_df["end_time"] - lexi_df["start_time"]) / 2
    lexi_df.set_index("mid_time", inplace=True)
    lexi_df.sort_index(inplace=True)
    
    # Filter to time range
    lexi_df = lexi_df[(lexi_df.index >= pd.to_datetime(time_range[0], utc=True)) & 
                      (lexi_df.index <= pd.to_datetime(time_range[1], utc=True))]
    
    # -------------------------------------------------------------------------
    # 3. LOAD LEXI SPACECRAFT POSITION
    # -------------------------------------------------------------------------
    print("Loading LEXI spacecraft position...")
    pos_file = Path("../data/lexi_themis_analysis/lexi_themis_c_analysis_lexi_spacecraft.csv")
    pos_df = pd.read_csv(pos_file)
    pos_df["Epoch"] = pd.to_datetime(pos_df["Epoch"], utc=True)
    pos_df.set_index("Epoch", inplace=True)
    pos_df.sort_index(inplace=True)
    
    # Filter to time range
    pos_df = pos_df[(pos_df.index >= pd.to_datetime(time_range[0], utc=True)) & 
                    (pos_df.index <= pd.to_datetime(time_range[1], utc=True))]
    
    # -------------------------------------------------------------------------
    # 4. PERFORM MEAN VARIANCE ANALYSIS
    # -------------------------------------------------------------------------
    print("\nPerforming Mean Variance Analysis...")
    
    # Extract magnetic field data for MVA
    B_data = themis_df[['thc_fgs_gse_x', 'thc_fgs_gse_y', 'thc_fgs_gse_z']].values
    
    # Perform MVA on the entire interval
    mva_result = perform_mva(B_data)
    
    if mva_result is not None:
        print(f"  Normal vector: {mva_result['normal']}")
        print(f"  Eigenvalues: {mva_result['eigenvalues']}")
        print(f"  Variance ratio (λ_mid/λ_min): {mva_result['variance_ratio']:.2f}")
        print(f"  Mean B field: {mva_result['B_mean']} nT")
    else:
        print("  MVA failed - insufficient data")
        return
    
    # -------------------------------------------------------------------------
    # 5. CALCULATE TIME DELAYS
    # -------------------------------------------------------------------------
    print("\nCalculating propagation delays...")
    
    delays = []
    delay_times = []
    
    # Merge dataframes on common time index
    common_times = themis_df.index.intersection(pos_df.index)
    
    for time in common_times:
        # Get THEMIS position and velocity
        pos_themis = themis_df.loc[time, ['thc_pos_gse_x', 'thc_pos_gse_y', 'thc_pos_gse_z']].values
        velocity = themis_df.loc[time, ['thc_peef_velocity_gse_x', 'thc_peef_velocity_gse_y', 'thc_peef_velocity_gse_z']].values
        
        # Get LEXI position
        pos_lexi = pos_df.loc[time, ['lexi_sc_pos_gse_x', 'lexi_sc_pos_gse_y', 'lexi_sc_pos_gse_z']].values
        
        # Calculate delay
        delay = calculate_propagation_delay(pos_themis, pos_lexi, velocity, mva_result['normal'])
        
        delays.append(delay)
        delay_times.append(time)
    
    # Create delay dataframe
    delay_df = pd.DataFrame({'delay_seconds': delays}, index=delay_times)
    
    # Clip delays to ±5 minutes (±300 seconds)
    delay_df['delay_seconds'] = delay_df['delay_seconds'].clip(lower=-300, upper=300)
    
    print(f"  Calculated {len(delays)} delay values")
    print(f"  Mean delay: {np.nanmean(delay_df['delay_seconds']):.1f} seconds")
    print(f"  Delay range: [{np.nanmin(delay_df['delay_seconds']):.1f}, {np.nanmax(delay_df['delay_seconds']):.1f}] seconds")
    
    # -------------------------------------------------------------------------
    # 6. ALIGN LEXI DATA USING CALCULATED DELAYS
    # -------------------------------------------------------------------------
    print("\nAligning LEXI data...")
    
    # For each LEXI observation, find the corresponding THEMIS data with delay applied
    lexi_aligned = []
    
    for lexi_time in lexi_df.index:
        # Find nearest delay value
        if lexi_time in delay_df.index:
            delay = delay_df.loc[lexi_time, 'delay_seconds']
        else:
            # Use nearest delay within tolerance
            idx = delay_df.index.get_indexer([lexi_time], method='nearest', tolerance=pd.Timedelta('2min'))
            if idx[0] != -1:
                delay = delay_df.iloc[idx[0]]['delay_seconds']
            else:
                delay = np.nan
        
        lexi_aligned.append(delay)
    
    lexi_df['propagation_delay_s'] = lexi_aligned
    
    # -------------------------------------------------------------------------
    # 7. CREATE VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\nCreating visualizations...")
    
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(5, 1, hspace=0.3)
    
    # Plot 1: THEMIS-C flux (scatter)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(themis_df.index, themis_df['thc_peef_flux'], s=15, alpha=0.7, label='THEMIS-C PEEF Flux')
    ax1.set_ylabel('THEMIS-C Flux\n[cm⁻²·s⁻¹·km·s⁻¹]', fontsize=10)
    ax1.set_title('THEMIS-C Flux and LEXI Counts with MVA-based Time Alignment', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: LEXI counts (original) (scatter)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.scatter(lexi_df.index, lexi_df['background_corrected_total_hist_counts'], s=15, 
                alpha=0.7, color='green', label='LEXI Counts (Original)')
    ax2.set_ylabel('LEXI Counts', fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: LEXI counts time-shifted and color-coded by delay (scatter)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    
    # Create time-shifted index for LEXI
    valid_mask = ~lexi_df['propagation_delay_s'].isna()
    lexi_shifted_times = lexi_df.index[valid_mask] - pd.to_timedelta(lexi_df['propagation_delay_s'][valid_mask], unit='s')
    lexi_shifted_values = lexi_df['background_corrected_total_hist_counts'][valid_mask]
    delays_for_color = lexi_df['propagation_delay_s'][valid_mask]
    
    # Create scatter plot with color representing delay
    scatter = ax3.scatter(lexi_shifted_times, lexi_shifted_values, 
                         c=delays_for_color, cmap='viridis', s=20, 
                         label='LEXI Counts (Time-aligned)')
    cbar = plt.colorbar(scatter, ax=ax3, label='Propagation Delay [s]')
    ax3.set_ylabel('LEXI Counts\n(Time-aligned)', fontsize=10)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Propagation delay over time (scatter)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    
    # Filter delays to range -300 to +300 seconds
    delay_filtered = delay_df['delay_seconds'].copy()
    delay_filtered = delay_filtered.clip(lower=-300, upper=300)
    
    ax4.scatter(delay_df.index, delay_filtered, s=15, alpha=0.7, color='red', label='Propagation Delay')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_ylabel('Delay [s]', fontsize=10)
    ax4.set_ylim(-300, 300)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Time delay distribution (new subplot)
    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
    
    # Plot time delay for each LEXI observation (scatter)
    valid_delay_mask = ~lexi_df['propagation_delay_s'].isna()
    lexi_delays_clipped = lexi_df['propagation_delay_s'][valid_delay_mask].clip(lower=-300, upper=300)
    
    ax5.scatter(lexi_df.index[valid_delay_mask], lexi_delays_clipped, s=15, 
                alpha=0.7, color='purple', label='LEXI Time Delay')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_ylabel('Time Delay [s]', fontsize=10)
    ax5.set_xlabel('Time [UTC]', fontsize=10)
    ax5.set_ylim(-300, 300)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    
    # Set x-axis limits to time_range
    t_start = pd.to_datetime(time_range[0], utc=True)
    t_end = pd.to_datetime(time_range[1], utc=True)
    ax1.set_xlim(t_start, t_end)
    ax2.set_xlim(t_start, t_end)
    ax3.set_xlim(t_start, t_end)
    ax4.set_xlim(t_start, t_end)
    ax5.set_xlim(t_start, t_end)
    
    fig.autofmt_xdate()
    
    # Save figure
    fig_dir = Path("../figures/mva_analysis")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "themis_lexi_mva_aligned_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    
    plt.close(fig)
    
    # Save aligned data
    output_dir = Path("../data/mva_aligned_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "lexi_mva_aligned_with_delays.csv"
    lexi_df.to_csv(output_file)
    print(f"Aligned LEXI data saved to {output_file}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
