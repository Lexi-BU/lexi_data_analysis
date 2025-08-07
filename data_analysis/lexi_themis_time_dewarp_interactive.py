from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dtaidistance import dtw
from plotly.subplots import make_subplots
from scipy.stats import pearsonr


# ---------------------------
# Helper function
# ---------------------------
def minmax_normalize(series):
    return (series - series.min()) / (series.max() - series.min())


# ---------------------------
# Load Data
# ---------------------------
folder_path = Path("../data/lexi_sw/istp/")
file_name = "istp_themis_c_esa_parameters_2025-03-16T19:30:00_to_2025-03-16T21:15:00_histogram_sum_1min_300.csv"
df = pd.read_csv(folder_path / file_name, index_col=0, parse_dates=True)

# Prepare data
sw_flux = df["sw_flux"].dropna()
hist_sum = df["hist_sum"].dropna()
common_index = sw_flux.index.intersection(hist_sum.index)
sw_flux = sw_flux.loc[common_index]
hist_sum = hist_sum.loc[common_index]

# Normalize both series (0 to 1)
sw_flux_norm = minmax_normalize(sw_flux)
hist_sum_norm = minmax_normalize(hist_sum)

# ---------------------------
# SWAP: hist_sum is reference (row); sw_flux is query (column)
# ---------------------------
distance, paths = dtw.warping_paths(hist_sum_norm.values, sw_flux_norm.values)
initial_best_path = np.array(dtw.best_path(paths))

# ---------------------------
# Apply time dewarping cap at ±5 minutes
# ---------------------------
max_warp_minutes = 15
bounded_best_path = []

for idx_hist, idx_sw in initial_best_path:
    t_hist = hist_sum.index[idx_hist]
    t_sw = sw_flux.index[idx_sw]
    delta_t = (t_sw - t_hist).total_seconds() / 60

    if abs(delta_t) <= max_warp_minutes:
        bounded_best_path.append((idx_hist, idx_sw))
    else:
        valid_times = sw_flux.index[
            (sw_flux.index >= t_hist - pd.Timedelta(minutes=max_warp_minutes))
            & (sw_flux.index <= t_hist + pd.Timedelta(minutes=max_warp_minutes))
        ]

        if not valid_times.empty:
            closest_time = valid_times[np.argmin(np.abs(valid_times - t_hist))]
            idx_sw_new = sw_flux.index.get_loc(closest_time)
            bounded_best_path.append((idx_hist, idx_sw_new))
        else:
            bounded_best_path.append((idx_hist, idx_sw))

# Convert to array
best_path = np.array(bounded_best_path)

# ---------------------------
# Aligned data using bounded DTW path
# ---------------------------
hist_sum_aligned = hist_sum.iloc[best_path[:, 0]].values
sw_flux_aligned = sw_flux.iloc[best_path[:, 1]].values
aligned_index = hist_sum.index[best_path[:, 0]]

# ✅ Compute correlation
r, _ = pearsonr(hist_sum_aligned, sw_flux_aligned)

# Time difference in minutes
original_times = pd.Series(hist_sum.index[best_path[:, 0]])
aligned_times = pd.Series(sw_flux.index[best_path[:, 1]])
time_warping_min = (aligned_times - original_times).dt.total_seconds() / 60
# Compute median of time dewarping (in minutes)
median_warping_min = np.median(time_warping_min)
mean_warping_min = np.mean(time_warping_min)
print(f"Median time dewarping applied: {median_warping_min:.2f} minutes")
print(f"Mean time dewarping applied: {mean_warping_min:.2f} minutes")
# ---------------------------
# Create figure
# ---------------------------
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    specs=[[{}], [{"secondary_y": True}], [{}], [{}]],
    subplot_titles=(
        "Original Time Series (MinMax Normalized)",
        "DTW-Aligned Series",
        "Time Warping (minutes)",
        "Warping Path Matrix",
    ),
    row_heights=[0.35, 0.35, 0.15, 0.15],
)

# ---------------------------
# 1. Normalized Original Data (single axis)
# ---------------------------
fig.add_trace(
    go.Scatter(
        x=hist_sum_norm.index,
        y=hist_sum_norm,
        name="hist_sum (norm)",
        line=dict(color="orange", width=1.5),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=sw_flux_norm.index,
        y=sw_flux_norm,
        name="sw_flux (norm)",
        line=dict(color="blue", width=1.5),
    ),
    row=1,
    col=1,
)

# ---------------------------
# Add directional arrows (actual Δt ≤ 5min)
# ---------------------------
warp_arrow_interval = 5

for i in range(0, len(best_path), warp_arrow_interval):
    x_hist = hist_sum_norm.index[best_path[i, 0]]
    y_hist = hist_sum_norm.iloc[best_path[i, 0]]
    x_sw = sw_flux_norm.index[best_path[i, 1]]
    y_sw = sw_flux_norm.iloc[best_path[i, 1]]

    delta_t_min = (x_sw - x_hist).total_seconds() / 60

    if abs(delta_t_min) > max_warp_minutes:
        continue

    arrow_color = "blue" if delta_t_min > 0 else "red"

    fig.add_annotation(
        x=x_sw,
        y=y_sw,
        ax=x_hist,
        ay=y_hist,
        xref="x1",
        yref="y1",
        axref="x1",
        ayref="y1",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        opacity=0.5,
        arrowcolor=arrow_color,
    )

# ---------------------------
# 2. Aligned Data (dual y-axis)
# ---------------------------
fig.add_trace(
    go.Scatter(
        x=aligned_index,
        y=hist_sum_aligned,
        name="hist_sum (aligned)",
        line=dict(color="orange", width=1.5),
    ),
    row=2,
    col=1,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=aligned_index,
        y=sw_flux_aligned,
        name="sw_flux (aligned)",
        line=dict(color="blue", width=1.5),
    ),
    row=2,
    col=1,
    secondary_y=True,
)

# ---------------------------
# 3. Time Warping in Minutes
# ---------------------------
fig.add_trace(
    go.Scatter(
        x=original_times,
        y=time_warping_min,
        name="Time Difference",
        mode="lines+markers",
        line=dict(color="green", width=1),
        marker=dict(size=5, color="darkgreen"),
        opacity=0.8,
    ),
    row=3,
    col=1,
)

fig.add_hline(y=0, line=dict(color="red", dash="dash", width=1), row=3, col=1)

fig.add_trace(
    go.Scatter(
        x=original_times,
        y=[max_warp_minutes] * len(original_times),
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(0,200,0,0.1)",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=original_times,
        y=[-max_warp_minutes] * len(original_times),
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(200,0,0,0.1)",
        showlegend=False,
    ),
    row=3,
    col=1,
)

# ---------------------------
# 4. Warping Path Matrix
# ---------------------------
path_matrix = np.log(paths + 1)
fig.add_trace(
    go.Heatmap(z=path_matrix, colorscale="plasma", showscale=False, hoverinfo="z"), row=4, col=1
)
fig.add_trace(
    go.Scatter(
        x=best_path[:, 1],
        y=best_path[:, 0],
        mode="lines",
        line=dict(color="white", width=2),
        showlegend=False,
    ),
    row=4,
    col=1,
)

# ---------------------------
# Layout & Axis Labels
# ---------------------------
fig.update_layout(
    title=f"DTW Analysis with Time Dewarping Capped at ±5 Minutes | Correlation: r = {r:.3f}",
    height=1200,
    hovermode="x unified",
    template="plotly_white",
    annotations=[
        dict(
            x=0.5,
            y=1.07,
            xref="paper",
            yref="paper",
            text=(
                "Blue arrows: sw_flux lags hist_sum | "
                "Red arrows: sw_flux leads hist_sum | "
                "Alignments capped at ±5 minutes | "
                f"Pearson r = {r:.3f}"
            ),
            showarrow=False,
            font=dict(size=12),
        )
    ]
    + list(fig.layout.annotations),
)

# Y-Axis Labels
fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
fig.update_yaxes(title_text="hist_sum (aligned)", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="sw_flux (aligned)", row=2, col=1, secondary_y=True)
fig.update_yaxes(title_text="Time Difference (minutes)", row=3, col=1)
fig.update_yaxes(title_text="hist_sum index", row=4, col=1)
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_xaxes(title_text="Reference Time (hist_sum)", row=3, col=1)
fig.update_xaxes(title_text="sw_flux index", row=4, col=1)

fig.update_yaxes(range=[-max_warp_minutes, max_warp_minutes], row=3, col=1)

fig.show()
