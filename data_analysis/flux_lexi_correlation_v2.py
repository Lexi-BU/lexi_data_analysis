"""
Flux-LEXI Correlation Analysis (Version 2)

This script analyzes the correlation between THEMIS spacecraft flux measurements
and LEXI (Lunar Environment X-ray Imager) background-corrected histogram counts.
It implements time-lag optimization to find the best temporal offset between
the two datasets, accounting for potential delays in physical responses.

Key features:
- Time-lag correlation analysis using both Pearson and Spearman methods
- Data interpolation onto common time grids for accurate comparison
- Visualization of shifted time series data
- Filtering for specific time windows (e.g., sunset to spacecraft wake)
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def find_best_time_lag(
    df,
    x_col="flux",
    y_col="background_flatfield_corrected_total_hist_counts",
    lag_min_s=-1800,
    lag_max_s=1800,
    lag_step_s=1,
):
    """
    Find the optimal time lag that maximizes correlation between two time series.

    This function systematically shifts one time series relative to another across
    a range of time lags and computes both Pearson and Spearman correlations at
    each lag. It returns the lag values that produce the highest correlation for
    each method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index containing both x and y columns
    x_col : str, optional
        Column name for the first variable (typically flux), by default "flux"
    y_col : str, optional
        Column name for the second variable (typically LEXI counts),
        by default "background_flatfield_corrected_total_hist_counts"
    lag_min_s : int, optional
        Minimum time lag to test in seconds (negative = y leads x), by default -1800
    lag_max_s : int, optional
        Maximum time lag to test in seconds (positive = x leads y), by default 1800
    lag_step_s : int, optional
        Step size between tested lags in seconds, by default 1

    Returns
    -------
    dict
        Dictionary containing:
        - 'pearson_r': Best Pearson correlation coefficient
        - 'pearson_lag_s': Time lag (seconds) for best Pearson correlation
        - 'spearman_rho': Best Spearman rank correlation coefficient
        - 'spearman_lag_s': Time lag (seconds) for best Spearman correlation

    Notes
    -----
    - Requires at least 3 valid data points for correlation calculation
    - Uses linear interpolation to align time series at each lag
    - Positive lag means x_col leads y_col in time
    - Negative lag means y_col leads x_col in time
    """
    # Extract the x and y series and convert to float for numerical operations
    sx = df[x_col].astype(float)
    sy = df[y_col].astype(float)

    # Create a mask for rows where both x and y have valid (non-NaN) values
    mask = sx.notna() & sy.notna()

    # Check if we have enough data points for meaningful correlation
    # Return NaN results if fewer than 3 valid points
    if mask.sum() < 3:
        return {
            "pearson_r": np.nan,
            "pearson_lag_s": np.nan,
            "spearman_rho": np.nan,
            "spearman_lag_s": np.nan,
        }

    # Convert datetime index to Unix timestamps (seconds since epoch)
    # view("int64") gives nanoseconds, divide by 1e9 to get seconds
    t = df.index.view("int64")[mask] / 1e9
    x = sx[mask].to_numpy()
    y = sy[mask].to_numpy()

    # Sort all arrays by time to ensure monotonic time axis for interpolation
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]

    # Create array of time lags to test (in seconds)
    lags = np.arange(lag_min_s, lag_max_s + lag_step_s, lag_step_s, dtype=float)

    def _best_corr(corr_fn):
        """
        Inner function to find best correlation for a given correlation function.

        For each lag value, it:
        1. Shifts the y time series by adding lag to its time axis
        2. Interpolates shifted y onto the original time grid
        3. Computes correlation between x and shifted y
        4. Tracks the lag that produces the maximum correlation
        """
        best_r, best_lag = -np.inf, np.nan

        for lag in lags:
            # Interpolate y values onto shifted time grid (t + lag)
            # left=np.nan, right=np.nan handles extrapolation by inserting NaN
            y_shift = np.interp(t + lag, t, y, left=np.nan, right=np.nan)

            # Create mask for valid (non-NaN) shifted values
            m = ~np.isnan(y_shift)

            # Skip this lag if too few valid overlap points
            if m.sum() < 3:
                continue

            # Compute correlation between x and shifted y (only valid points)
            res = corr_fn(x[m], y_shift[m])

            # Extract correlation coefficient (handle different scipy versions)
            r = res.statistic if hasattr(res, "statistic") else res[0]

            # Update best correlation and lag if this is better
            if r > best_r:
                best_r, best_lag = r, lag

        return best_r, best_lag

    # Find best lag for both Pearson (linear correlation) and Spearman (rank correlation)
    pearson_r, pearson_lag = _best_corr(pearsonr)
    spearman_rho, spearman_lag = _best_corr(spearmanr)

    # Return all results as a dictionary
    return {
        "pearson_r": pearson_r,
        "pearson_lag_s": pearson_lag,
        "spearman_rho": spearman_rho,
        "spearman_lag_s": spearman_lag,
    }


def plot_flux_vs_background_counts_same_grid(
    selected_data: pd.DataFrame,
    best_lag: float,
    x_col: str = "flux",
    y_col: str = "background_flatfield_corrected_total_hist_counts",
    save_dir: Path | str = "../figures/correlation",
    title_suffix: str | None = None,
):
    """
    Plot flux and background counts on the same time grid with time-shifted overlay.

    Creates a dual-axis time series plot showing the original LEXI counts and
    the time-shifted version that maximizes correlation with flux measurements.
    Vertical grid lines are drawn at intervals equal to the lag period to help
    visualize the temporal relationship.

    Parameters
    ----------
    selected_data : pd.DataFrame
        DataFrame with datetime index containing both x and y columns
    best_lag : float
        Optimal time lag in seconds (from find_best_time_lag)
    x_col : str, optional
        Column name for flux data, by default "flux"
    y_col : str, optional
        Column name for LEXI counts,
        by default "background_flatfield_corrected_total_hist_counts"
    save_dir : Path | str, optional
        Directory to save the output figure, by default "../figures/correlation"
    title_suffix : str | None, optional
        Additional text to append to plot title, by default None

    Returns
    -------
    Path
        Path to the saved figure file

    Notes
    -----
    - Uses dark background matplotlib style
    - Left y-axis for flux, right y-axis for LEXI counts
    - Vertical dashed lines mark intervals equal to |best_lag|
    - Figure saved as PNG at 150 DPI
    """
    # Extract x (flux) and y (LEXI counts) series as floats
    x = selected_data[x_col].astype(float)
    y = selected_data[y_col].astype(float)

    # Use x.index as the common target time grid for interpolation
    # Convert datetime index to Unix timestamps (nanoseconds -> seconds)
    t_target = x.index.view("int64") / 1e9
    t_y = y.index.view("int64") / 1e9
    y_vals = y.to_numpy()

    # Interpolate y values onto the x time grid (unshifted)
    # This ensures both series are on exactly the same time points
    y_on_x = pd.Series(
        np.interp(t_target, t_y, y_vals, left=np.nan, right=np.nan),
        index=x.index,
        name=f"{y_col} (on flux grid)",
    )

    # Interpolate y values onto the x time grid with time shift applied
    # Adding best_lag to t_y shifts y backward in time if lag is positive
    y_shifted_on_x = pd.Series(
        np.interp(t_target, t_y + best_lag, y_vals, left=np.nan, right=np.nan),
        index=x.index,
        name=f"{y_col} (shifted {best_lag:.0f}s, on flux grid)",
    )

    # Set up the plot with dark background theme
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a second y-axis for LEXI counts (shares x-axis with flux)
    ax2 = ax.twinx()

    # Plot original LEXI counts (interpolated onto flux grid)
    ax2.plot(
        y_on_x.index,
        y_on_x.values,
        label="Background FF Counts (original, on flux grid)",
        alpha=0.9,
    )

    # Plot time-shifted LEXI counts for comparison
    ax2.plot(
        y_shifted_on_x.index,
        y_shifted_on_x.values,
        label=f"Background FF Counts (shifted by {best_lag:.0f}s)",
        alpha=0.9,
    )

    # Flux plotting is commented out but available if needed
    # ax.plot(x.index, x.values, label="Flux", alpha=0.9)

    # Label the axes
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("Flux [km/s/cm²]")
    ax2.set_ylabel("Total LEXI Counts")

    # Add vertical grid lines at intervals equal to the lag period
    # This helps visualize the periodic relationship between datasets
    if np.isfinite(best_lag) and abs(best_lag) > 0:
        lag_in_seconds = abs(int(best_lag))
        # Convert lag to nanoseconds for pandas Timestamp operations
        step_ns = pd.to_timedelta(lag_in_seconds, unit="s").value
        t0 = x.index.min().value  # Start of time range (nanoseconds)
        t1 = x.index.max().value  # End of time range (nanoseconds)

        # Draw vertical dashed lines at each lag interval
        for xgrid in range(t0, t1, int(step_ns)):
            ax.axvline(pd.Timestamp(xgrid), linestyle="--", alpha=0.25)

    # Add optional title suffix if provided
    if title_suffix:
        ax.set_title(title_suffix)

    # Add legend for LEXI counts (right side)
    # Flux legend commented out but available if flux is plotted
    # ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Set x-axis limits to span the full time range
    ax.set_xlim(x.index.min(), x.index.max())

    # Set y-axis limits for LEXI counts with 10% padding
    # Combine both original and shifted values to determine range
    finite_vals = np.concatenate(
        [np.asarray(y_on_x.values, dtype=float), np.asarray(y_shifted_on_x.values, dtype=float)]
    )
    finite_vals = finite_vals[np.isfinite(finite_vals)]

    if finite_vals.size:
        ymin, ymax = np.nanmin(finite_vals), np.nanmax(finite_vals)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            # Add 10% padding on both sides
            ax2.set_ylim(ymin - 0.1 * abs(ymax - ymin), ymax + 0.1 * abs(ymax - ymin))

    # Optimize layout and save the figure
    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"flux_vs_background_counts_shifted_{best_lag:.0f}_same_grid.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)

    return out


# -------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# -------------------------------------------------------------------------
# This section demonstrates the complete workflow for correlation analysis
# between THEMIS flux measurements and LEXI background counts.
# -------------------------------------------------------------------------

# Control flag: Set to True to recompute from source files, False to use existing data
recompute = True

if recompute:
    # -------------------------------------------------------------------------
    # 1. LOAD LEXI DATA
    # -------------------------------------------------------------------------
    # Load LEXI line profile fit parameters with background corrections
    data_folder = Path("../data/line_profile_data/bg_corrected/from_l2/")
    csv_file = data_folder / "line_profile_fit_parameters_bg_corrected_1min.csv"
    data_df = pd.read_csv(csv_file)

    # Convert time columns to pandas datetime objects with UTC timezone
    data_df["start_time"] = pd.to_datetime(data_df["start_time"], utc=True)
    data_df["end_time"] = pd.to_datetime(data_df["end_time"], utc=True)

    # Calculate midpoint time for each observation bin (1-minute bins)
    data_df["mid_time"] = data_df["start_time"] + (data_df["end_time"] - data_df["start_time"]) / 2

    # Set mid_time as the index for time-series operations
    data_df.set_index("mid_time", inplace=True)
    data_df.index = pd.to_datetime(data_df.index, utc=True)
    data_df.sort_index(inplace=True)

    # -------------------------------------------------------------------------
    # 2. LOAD THEMIS FLUX DATA
    # -------------------------------------------------------------------------
    # Select THEMIS spacecraft ('b' or 'c') for flux measurements
    themis_spc = "c"
    flux_file_name = (
        f"../data/lexi_themis_analysis/lexi_themis_{themis_spc}_analysis_lexi_spacecraft.csv"
    )

    # Define spacecraft wake time (when LEXI exits Earth's shadow)
    # Different times for THEMIS-B and THEMIS-C due to different orbits
    if themis_spc == "b":
        wake_time = "2025-03-16 20:47"
    elif themis_spc == "c":
        wake_time = "2025-03-16 20:55"
    wake_time = pd.to_datetime(wake_time, utc=True)

    # Load flux data from CSV
    flux_df = pd.read_csv(flux_file_name)
    flux_df["Epoch"] = pd.to_datetime(flux_df["Epoch"], utc=True)
    flux_df.set_index("Epoch", inplace=True)
    flux_df.sort_index(inplace=True)

    # Extract the parallel electron energy flux (PEEF) column
    selected_flux_data = flux_df[f"th{themis_spc}_peef_flux"]
    selected_flux_data.name = f"th{themis_spc}_peef_flux"

    # -------------------------------------------------------------------------
    # 3. ALIGN FLUX DATA TO LEXI TIME GRID
    # -------------------------------------------------------------------------
    # Reindex flux data to match LEXI's 1-minute time grid
    # Floor to nearest minute and use nearest-neighbor matching within 31 seconds
    # This handles slight timing mismatches between datasets
    flux_aligned = selected_flux_data.reindex(
        data_df.index.floor("min"),
        method="nearest",
        tolerance=pd.Timedelta("31s"),
    )

    # Add aligned flux as a new column to the LEXI dataframe
    data_df = data_df.copy()
    data_df["flux"] = flux_aligned.values

    # -------------------------------------------------------------------------
    # 4. SELECT TIME WINDOW FOR ANALYSIS
    # -------------------------------------------------------------------------
    # Analyze data from sunset end to spacecraft wake
    # This window captures the magnetotail passage when correlations are strongest
    sunset_end_time = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)
    selected_data = data_df[(data_df.index >= sunset_end_time) & (data_df.index <= wake_time)]

    # -------------------------------------------------------------------------
    # 5. COMPUTE OPTIMAL TIME LAG
    # -------------------------------------------------------------------------
    # Search for time lag that maximizes correlation between flux and LEXI counts
    # Tests lags from -1800s to +1800s (±30 minutes) in 1-second steps
    result = find_best_time_lag(selected_data, y_col="background_corrected_total_hist_counts")
    print(result)
else:
    # If recompute=False, assume data_df, selected_data, and result
    # already exist in the current session (e.g., from previous run)
    pass

# -------------------------------------------------------------------------
# 6. GENERATE VISUALIZATION
# -------------------------------------------------------------------------
# Use the Spearman lag (rank correlation) as it's more robust to outliers
best_lag = result["spearman_lag_s"]

# Create and save the time-shifted comparison plot
outfile = plot_flux_vs_background_counts_same_grid(
    selected_data,
    best_lag=best_lag,
    x_col="flux",
    y_col="background_flatfield_corrected_total_hist_counts",
    save_dir="../figures/correlation",
    title_suffix=f"Best lag (Spearman): {best_lag:.0f} s",
)
print(f"Saved: {outfile}")
