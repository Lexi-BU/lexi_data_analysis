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


def plot_flux_vs_background_counts_same_grid(
    selected_data: pd.DataFrame,
    best_lag: float,
    x_col: str = "flux",
    y_col: str = "background_flatfield_corrected_total_hist_counts",
    save_dir: Path | str = "../figures/correlation",
    title_suffix: str | None = None,
):
    # Series
    x = selected_data[x_col].astype(float)
    y = selected_data[y_col].astype(float)

    # Common target grid = x.index
    t_target = x.index.view("int64") / 1e9
    t_y = y.index.view("int64") / 1e9
    y_vals = y.to_numpy()

    # Unshifted and shifted onto the same grid
    y_on_x = pd.Series(
        np.interp(t_target, t_y, y_vals, left=np.nan, right=np.nan),
        index=x.index,
        name=f"{y_col} (on flux grid)",
    )
    y_shifted_on_x = pd.Series(
        np.interp(t_target, t_y + best_lag, y_vals, left=np.nan, right=np.nan),
        index=x.index,
        name=f"{y_col} (shifted {best_lag:.0f}s, on flux grid)",
    )

    # Plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax2 = ax.twinx()
    ax2.plot(
        y_on_x.index,
        y_on_x.values,
        label="Background FF Counts (original, on flux grid)",
        alpha=0.9,
    )
    ax2.plot(
        y_shifted_on_x.index,
        y_shifted_on_x.values,
        label=f"Background FF Counts (shifted by {best_lag:.0f}s)",
        alpha=0.9,
    )

    # Optionally plot flux (uncomment if needed)
    # ax.plot(x.index, x.values, label="Flux", alpha=0.9)

    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("Flux [km/s/cm²]")
    ax2.set_ylabel("Total LEXI Counts")

    # Grid every |lag| seconds if finite and > 0
    if np.isfinite(best_lag) and abs(best_lag) > 0:
        lag_in_seconds = abs(int(best_lag))
        step_ns = pd.to_timedelta(lag_in_seconds, unit="s").value
        t0 = x.index.min().value
        t1 = x.index.max().value
        for xgrid in range(t0, t1, int(step_ns)):
            ax.axvline(pd.Timestamp(xgrid), linestyle="--", alpha=0.25)

    if title_suffix:
        ax.set_title(title_suffix)

    # Legends
    # ax.legend(loc="upper left")  # if flux is plotted
    ax2.legend(loc="upper right")

    # Limits
    ax.set_xlim(x.index.min(), x.index.max())
    finite_vals = np.concatenate(
        [np.asarray(y_on_x.values, dtype=float), np.asarray(y_shifted_on_x.values, dtype=float)]
    )
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size:
        ymin, ymax = np.nanmin(finite_vals), np.nanmax(finite_vals)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            ax2.set_ylim(ymin - 0.1 * abs(ymax - ymin), ymax + 0.1 * abs(ymax - ymin))

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"flux_vs_background_counts_shifted_{best_lag:.0f}_same_grid.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -------------------------
# Example pipeline
# -------------------------
recompute = True

if recompute:
    data_folder = Path("../data/line_profile_data/bg_corrected/from_l2/")
    csv_file = data_folder / "line_profile_fit_parameters_bg_corrected_1min.csv"
    data_df = pd.read_csv(csv_file)

    data_df["start_time"] = pd.to_datetime(data_df["start_time"], utc=True)
    data_df["end_time"] = pd.to_datetime(data_df["end_time"], utc=True)
    data_df["mid_time"] = data_df["start_time"] + (data_df["end_time"] - data_df["start_time"]) / 2
    data_df.set_index("mid_time", inplace=True)
    data_df.index = pd.to_datetime(data_df.index, utc=True)
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
    flux_df["Epoch"] = pd.to_datetime(flux_df["Epoch"], utc=True)
    flux_df.set_index("Epoch", inplace=True)
    flux_df.sort_index(inplace=True)

    selected_flux_data = flux_df[f"th{themis_spc}_peef_flux"]
    selected_flux_data.name = f"th{themis_spc}_peef_flux"

    # Align flux to minute grid closest to data_df mid_time (tolerance 31s)
    flux_aligned = selected_flux_data.reindex(
        data_df.index.floor("min"),
        method="nearest",
        tolerance=pd.Timedelta("31s"),
    )

    data_df = data_df.copy()
    data_df["flux"] = flux_aligned.values

    sunset_end_time = datetime.datetime(2025, 3, 16, 19, 29, 0, tzinfo=datetime.timezone.utc)
    selected_data = data_df[(data_df.index >= sunset_end_time) & (data_df.index <= wake_time)]

    result = find_best_time_lag(selected_data)
    print(result)
else:
    # Assume `data_df`, `selected_data`, and `result` already exist in the session
    pass

best_lag = result["spearman_lag_s"]
outfile = plot_flux_vs_background_counts_same_grid(
    selected_data,
    best_lag=best_lag,
    x_col="flux",
    y_col="background_flatfield_corrected_total_hist_counts",
    save_dir="../figures/correlation",
    title_suffix=f"Best lag (Spearman): {best_lag:.0f} s",
)
print(f"Saved: {outfile}")
