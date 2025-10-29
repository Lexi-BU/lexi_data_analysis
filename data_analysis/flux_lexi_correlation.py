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
    Return the lags (in seconds) that maximize Pearson and Spearman correlation between
    x(t) and y(t+lag). Positive lag => y occurs later (x leads y).
    """
    sx = df[x_col].astype(float)
    sy = df[y_col].astype(float)

    mask = sx.notna() & sy.notna()
    if mask.sum() < 3:
        return {
            "pearson_r": np.nan,
            "pearson_lag_s": np.nan,
            "spearman_rho": np.nan,
            "spearman_lag_s": np.nan,
        }

    # time in seconds (monotonic)
    t = df.index.view("int64")[mask] / 1e9
    x = sx[mask].to_numpy()
    y = sy[mask].to_numpy()

    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]

    lags = np.arange(lag_min_s, lag_max_s + lag_step_s, lag_step_s, dtype=float)

    def _best_corr(corr_fn):
        best_r, best_lag = -np.inf, np.nan
        for lag in lags:
            y_shift = np.interp(t + lag, t, y, left=np.nan, right=np.nan)
            m = ~np.isnan(y_shift)
            if m.sum() < 3:
                continue
            res = corr_fn(x[m], y_shift[m])
            r = res.statistic if hasattr(res, "statistic") else res[0]
            if r > best_r:
                best_r, best_lag = r, lag
        return best_r, best_lag

    pearson_r, pearson_lag = _best_corr(pearsonr)
    spearman_rho, spearman_lag = _best_corr(spearmanr)

    return {
        "pearson_r": pearson_r,
        "pearson_lag_s": pearson_lag,
        "spearman_rho": spearman_rho,
        "spearman_lag_s": spearman_lag,
    }


def plot_flux_vs_background_counts(selected_data, data_df, best_lag):
    # Align the second series by that lag
    x = selected_data["flux"]
    y_original = data_df["background_flatfield_corrected_total_hist_counts"]

    # Create shifted index for y
    shifted_index = y_original.index + pd.to_timedelta(best_lag, unit="s")
    y_shifted = pd.Series(
        np.interp(
            x.index.view("int64") / 1e9,
            shifted_index.view("int64") / 1e9,
            y_original.to_numpy(),
            left=np.nan,
            right=np.nan,
        ),
        index=x.index,
    )

    # Plot both
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x.index, x, label="Flux", color="cyan")

    # Plot y_shifted on a twin axis
    ax2 = ax.twinx()
    ax2.plot(
        x.index,
        y_shifted,
        label=f"Background FF Counts (shifted by {best_lag:.0f}s)",
        color="orange",
    )
    # Plot the non-shifted y data for comparison
    # ax2.plot(
    #     data_df.index,
    #     y_original,
    #     label="Background FF Counts (original)",
    #     color="green",
    #     alpha=0.01,
    # )

    # Show legend
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("Flux [km/s/cm²]")
    ax2.set_ylabel("Total LEXI Counts")
    # ax.grid(True, which="both", linestyle="--", alpha=0.3)
    # Add a grid every lag seconds
    lag_in_seconds = abs(int(best_lag))
    for xgrid in np.arange(
        x.index.min().value, x.index.max().value, pd.to_timedelta(lag_in_seconds, unit="s").value
    ):
        ax.axvline(pd.Timestamp(xgrid), color="gray", linestyle="--", alpha=0.3)

    # Add the value of the best lag to the plot
    ax2.text(
        0.05,
        0.95,
        f"Best lag: {best_lag:.0f} s",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        color="orange",
    )

    # # Set the x-axis limits to the range of the data
    ax.set_xlim(x.index.min(), x.index.max())

    # Set the twin axis y limits to match the original y data
    ax2.set_ylim(y_shifted.min() * 0.9, y_shifted.max() * 1.1)

    plt.tight_layout()
    plot_folder = Path("../figures/correlation/")
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_folder / f"flux_vs_background_counts_shifted_{best_lag:.0f}.png")


recompute = False

if recompute:
    # Load the CSV file
    data_folder = Path("../data/line_profile_data/bg_corrected/from_l2/")
    csv_file = data_folder / "line_profile_fit_parameters_bg_corrected_1min.csv"
    data_df = pd.read_csv(csv_file)

    data_df["start_time"] = pd.to_datetime(data_df["start_time"], utc=True)
    data_df["end_time"] = pd.to_datetime(data_df["end_time"], utc=True)
    # Make time series plots of the fit parameters
    data_df["mid_time"] = data_df["start_time"] + (data_df["end_time"] - data_df["start_time"]) / 2
    data_df.set_index("mid_time", inplace=True)
    # Convert the index to datetime if not already
    data_df.index = pd.to_datetime(data_df.index, utc=True)
    # Sort the dataframe by the datetime index
    data_df.sort_index(inplace=True)

    themis_spc = "c"
    flux_file_name = (
        f"../data/lexi_themis_analysis/lexi_themis_{themis_spc}_analysis_lexi_spacecraft.csv"
    )
    if themis_spc == "b":
        wake_time = "2025-03-16 20:47"
    elif themis_spc == "c":
        wake_time = "2025-03-16 20:55"

    wake_time = pd.to_datetime(wake_time, utc=True)

    flux_df = pd.read_csv(flux_file_name)

    # Set Epoch as datetime index
    flux_df["Epoch"] = pd.to_datetime(flux_df["Epoch"], utc=True)
    flux_df.set_index("Epoch", inplace=True)
    flux_df.sort_index(inplace=True)
    selected_flux_data = flux_df[f"th{themis_spc}_peef_flux"]
    # Name the flux column
    selected_flux_data.name = f"th{themis_spc}_peef_flux"

    # reindex the flux to match data_df.index (with tolerance for nearest minute)
    flux_aligned = selected_flux_data.reindex(
        data_df.index.floor("min"),  # floor to minute
        method="nearest",  # nearest match
        tolerance=pd.Timedelta("31s"),  # safe tolerance
    )

    data_df = data_df.copy()
    data_df["flux"] = flux_aligned.values

    sunset_end_time = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)

    selected_data = data_df[(data_df.index >= sunset_end_time) & (data_df.index <= wake_time)]
    result = find_best_time_lag(selected_data)
    print(result)

best_lag = result["pearson_lag_s"]
plot_flux_vs_background_counts(selected_data, data_df, best_lag)
