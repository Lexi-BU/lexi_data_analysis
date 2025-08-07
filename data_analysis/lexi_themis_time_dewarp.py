from datetime import timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr


def safe_shifted_correlation(reference: pd.Series, shifted: pd.Series, tolerance="30s"):
    aligned = shifted.reindex(reference.index, method="nearest", tolerance=pd.Timedelta(tolerance))
    df = pd.concat([reference, aligned], axis=1).dropna()
    if len(df) == 0:
        return np.nan, np.nan  # Pearson, Spearman

    pearson = df.iloc[:, 0].corr(df.iloc[:, 1], method="pearson")
    spearman = spearmanr(df.iloc[:, 0], df.iloc[:, 1], nan_policy="omit").correlation
    return pearson, spearman


def find_best_shift(ref_series, target_series, freq_seconds=5, max_lag_minutes=10):
    s1 = ref_series.dropna()
    s2 = target_series.dropna()
    common = s1.index.intersection(s2.index)
    if len(common) == 0:
        return pd.Timedelta(seconds=0)
    s1 = s1.loc[common] - s1.loc[common].mean()
    s2 = s2.loc[common] - s2.loc[common].mean()

    n = len(s1)
    max_lag = int((max_lag_minutes * 60) // freq_seconds)
    corr = correlate(s2.values, s1.values, mode="full")
    lags = np.arange(-n + 1, n)
    lag_range = (lags >= -max_lag) & (lags <= max_lag)
    lags = lags[lag_range]
    corr = corr[lag_range]

    best_lag = lags[np.argmax(corr)]
    return pd.Timedelta(seconds=best_lag * freq_seconds)


def plot_themis_lexi_hist_time_series_shifted(
    time_series,
    sw_np,
    sw_vp,
    sw_flux,
    hist_sum,
    hist_sum_no_mask,
    themis_sc="c",
    freq="5seconds",
    integration_time="5minutes",
    x_limit=None,
    start_time=None,
    end_time=None,
):
    # Infer frequency
    freq_seconds = int((time_series[1] - time_series[0]).total_seconds())

    # Compute best shifts
    shift_np = find_best_shift(hist_sum, sw_np, freq_seconds)
    shift_vp = find_best_shift(hist_sum, sw_vp, freq_seconds)
    shift_flux = find_best_shift(hist_sum, sw_flux, freq_seconds)

    # Redefine shift_vp to be 4 minutes
    shift_vp = pd.Timedelta(minutes=3.6)
    shift_np = pd.Timedelta(minutes=4.5)

    hist_np = hist_sum.copy()
    hist_np.index = hist_np.index + shift_np
    hist_vp = hist_sum.copy()
    hist_vp.index = hist_vp.index + shift_vp
    hist_flux = hist_sum.copy()
    hist_flux.index = hist_flux.index + shift_flux

    print(f"Optimal shift for density: {shift_np}")
    print(f"Optimal shift for velocity: {shift_vp}")
    print(f"Optimal shift for flux: {shift_flux}")

    # Dark mode
    mpl.style.use("dark_background")
    plt.rcParams.update(
        {
            "axes.facecolor": "#000000",
            "axes.edgecolor": "#ffffff",
            "figure.facecolor": "#020202",
            "figure.edgecolor": "#ffffff",
            "grid.color": "#ffffff",
            "text.color": "#ffffff",
            "xtick.color": "#ffffff",
            "ytick.color": "#ffffff",
        }
    )

    fig, ax = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(
        f"THEMIS {themis_sc} ESA Parameters and LEXI Histogram Data\n{start_time} to {end_time}",
        fontsize=16,
    )

    # ----------- DENSITY -----------
    ax[0].plot(time_series, sw_np, label="Density (cm^-3)", color="blue")
    twin_ax0 = ax[0].twinx()
    twin_ax0.plot(
        hist_np.index,
        hist_np,
        label=f"Hist Sum (shift: {shift_np.total_seconds()}s)",
        color="red",
        alpha=0.6,
    )
    twin_ax0.set_yscale("log")
    ax[0].set_ylabel("Density [cm^-3]")
    twin_ax0.set_ylabel("Hist Sum", color="red")
    twin_ax0.tick_params(axis="y", colors="red")
    twin_ax0.spines["right"].set_color("red")
    ax[0].legend(loc="upper left", fontsize=10)
    twin_ax0.legend(loc="upper right", fontsize=10)

    # Compute the correlations
    corr_np_pearson, spearman_corr_np = safe_shifted_correlation(sw_np, hist_np)
    old_corr_np_pearson, old_spearman_corr_np = safe_shifted_correlation(sw_np, hist_sum)
    # Add text annotations to display all 4 correlations
    ax[0].text(
        0.02,
        0.02,
        f"pearson_shifted = {corr_np_pearson:.2f}\n pearson = {old_corr_np_pearson:.2f}",
        transform=ax[0].transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )
    ax[0].text(
        0.52,
        0.02,
        f"spearman_shifted = {spearman_corr_np:.2f}\n spearman = {old_spearman_corr_np:.2f}",
        transform=ax[0].transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )

    # ----------- VELOCITY -----------
    ax[1].plot(time_series, sw_vp, label="Velocity (km/s)", color="orange")
    twin_ax1 = ax[1].twinx()
    twin_ax1.plot(
        hist_vp.index,
        hist_vp,
        label=f"Hist Sum (shift: {shift_vp.total_seconds()}s)",
        color="red",
        alpha=0.6,
    )
    twin_ax1.set_yscale("log")
    ax[1].set_ylabel("Velocity [km/s]")
    twin_ax1.set_ylabel("Hist Sum", color="red")
    twin_ax1.tick_params(axis="y", colors="red")
    twin_ax1.spines["right"].set_color("red")
    ax[1].legend(loc="upper left", fontsize=10)
    twin_ax1.legend(loc="upper right", fontsize=10)

    # Compute the correlations
    corr_vp_pearson, spearman_corr_vp = safe_shifted_correlation(sw_vp, hist_vp)
    old_corr_vp_pearson, old_spearman_corr_vp = safe_shifted_correlation(sw_vp, hist_sum)
    ax[1].text(
        0.02,
        0.02,
        f"pearson_shifted = {corr_vp_pearson:.2f}\n pearson = {old_corr_vp_pearson:.2f}",
        transform=ax[1].transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )
    ax[1].text(
        0.52,
        0.02,
        f"spearman_shifted = {spearman_corr_vp:.2f}\n spearman = {old_spearman_corr_vp:.2f}",
        transform=ax[1].transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )

    # ----------- FLUX -----------
    ax[2].plot(time_series, sw_flux, label="Flux (km cm^-3 s^-1)", color="green")
    twin_ax2 = ax[2].twinx()
    twin_ax2.plot(
        hist_flux.index,
        hist_flux,
        label=f"Hist Sum (shift: {shift_flux.total_seconds()}s)",
        color="red",
        alpha=0.6,
    )
    twin_ax2.set_yscale("log")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("Flux [km cm^-3 s^-1]")
    twin_ax2.set_ylabel("Hist Sum", color="red")
    twin_ax2.tick_params(axis="y", colors="red")
    twin_ax2.spines["right"].set_color("red")
    ax[2].legend(loc="upper left", fontsize=10)
    twin_ax2.legend(loc="upper right", fontsize=10)

    # Compute the correlations
    corr_flux_pearson, spearman_corr_flux = safe_shifted_correlation(sw_flux, hist_flux)
    old_corr_flux_pearson, old_spearman_corr_flux = safe_shifted_correlation(sw_flux, hist_sum)
    ax[2].text(
        0.02,
        0.02,
        f"pearson_shifted = {corr_flux_pearson:.2f}\n pearson = {old_corr_flux_pearson:.2f}",
        transform=ax[2].transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )
    ax[2].text(
        0.52,
        0.02,
        f"spearman_shifted = {spearman_corr_flux:.2f}\n spearman = {old_spearman_corr_flux:.2f}",
        transform=ax[2].transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )

    # ----------- HISTOGRAM SUM -----------
    ax[3].plot(time_series, hist_sum, label="Original", color="orange")
    twin_ax3 = ax[3].twinx()
    twin_ax3.plot(hist_flux.index, hist_flux, label="shifted", color="red", alpha=0.6)
    ax[3].set_ylabel("Histogram Sum")
    twin_ax3.set_ylabel("Shifted", color="red")
    twin_ax3.tick_params(axis="y", colors="red")
    twin_ax3.spines["right"].set_color("red")
    ax[3].legend(loc="upper left", fontsize=10)
    twin_ax3.legend(loc="upper right", fontsize=10)
    ax[3].set_xlabel("Time [UTC]")
    ax[3].set_yscale("log")
    twin_ax3.set_yscale("log")

    # Set gridlines
    for axis in ax:
        axis.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
        axis.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 3)))
        axis.grid(which="major", color="white", alpha=0.25)
        axis.grid(which="minor", color="white", alpha=0.05)

    # X limit
    if x_limit is not None:
        ax[3].set_xlim(pd.to_datetime(x_limit, utc=True))
    elif time_series is not None and len(time_series) > 0:
        ax[3].set_xlim([time_series[0], time_series[-1]])

    # Save figure
    folder_path = Path(f"../figures/lexi_sw/istp/themis_{themis_sc}/")
    folder_path.mkdir(parents=True, exist_ok=True)
    file_name = f"themis_{themis_sc}_esa_parameters_{start_time}_to_{end_time}_shifted_overlay.png"
    file_path = folder_path / file_name
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Figure saved to {file_path}")

    return {
        "hist_np": hist_np,
        "hist_vp": hist_vp,
        "hist_flux": hist_flux,
        "hist_sum": hist_sum,
    }


folder_path = Path("../data/lexi_sw/istp/")
file_name = "istp_themis_c_esa_parameters_2025-03-16T19:30:00_to_2025-03-16T21:15:00_histogram_sum_1min_300.csv"

df = pd.read_csv(folder_path / file_name, index_col=0, parse_dates=True)
results = plot_themis_lexi_hist_time_series_shifted(
    time_series=df.index,
    sw_np=df["sw_np"],
    sw_vp=df["sw_vp"],
    sw_flux=df["sw_flux"],
    hist_sum=df["hist_sum"],
    hist_sum_no_mask=df["hist_sum_no_mask"],
    themis_sc="c",
    freq="5seconds",
    integration_time="300sec",
    x_limit=["2025-03-16T19:30:00Z", "2025-03-16T21:15:00Z"],
    start_time="2025-03-16T19:30:00Z",
    end_time="2025-03-16T21:15:00Z",
)
