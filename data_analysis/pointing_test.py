from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

uninterpolated_file_name = "/mnt/cephadrius/bu_research/lexi_data/pointing_data/lexi_look_direction_data_uninterpolated_v0.0.csv"
interpolated_file_name = "/mnt/cephadrius/bu_research/lexi_data/pointing_data/lexi_look_direction_data_resampled_interpolated_v0.0.csv"

uninterpolated_df = pd.read_csv(uninterpolated_file_name)
interpolated_df = pd.read_csv(interpolated_file_name)

# Set the Epoch column as the index
uninterpolated_df["Epoch"] = pd.to_datetime(uninterpolated_df["Epoch"], utc=True)
interpolated_df["Epoch"] = pd.to_datetime(interpolated_df["Epoch"], utc=True)
uninterpolated_df.set_index("Epoch", inplace=True)
interpolated_df.set_index("Epoch", inplace=True)

# Time range filter
start_time = pd.to_datetime("2025-03-16T00:00:00Z", utc=True)
end_time = pd.to_datetime("2025-03-17T00:00:00Z", utc=True)
uninterpolated_df = uninterpolated_df.loc[start_time:end_time]
interpolated_df = interpolated_df.loc[start_time:end_time]

fig = go.Figure()

# Uninterpolated RA (higher zorder, bigger marker)
fig.add_trace(
    go.Scattergl(
        x=uninterpolated_df.index,
        y=uninterpolated_df["ra_lexi"],
        mode="markers",
        name="Uninterpolated RA",
        marker=dict(color="blue", size=6),
        yaxis="y1",
    )
)

# Interpolated RA (lower zorder, smaller marker)
fig.add_trace(
    go.Scattergl(
        x=interpolated_df.index,
        y=interpolated_df["ra_lexi"],
        mode="markers",
        name="Interpolated RA",
        marker=dict(color="orange", size=3),
        yaxis="y1",
    )
)

# Uninterpolated Dec (higher zorder, bigger marker)
fig.add_trace(
    go.Scattergl(
        x=uninterpolated_df.index,
        y=uninterpolated_df["dec_lexi"],
        mode="markers",
        name="Uninterpolated Dec",
        marker=dict(color="blue", size=6, symbol="circle-open"),
        yaxis="y2",
    )
)

# Interpolated Dec (lower zorder, smaller marker)
fig.add_trace(
    go.Scattergl(
        x=interpolated_df.index,
        y=interpolated_df["dec_lexi"],
        mode="markers",
        name="Interpolated Dec",
        marker=dict(color="orange", size=3, symbol="circle-open"),
        yaxis="y2",
    )
)

fig.update_layout(
    title="Right Ascension and Declination Comparison",
    height=800,
    xaxis=dict(domain=[0, 1], title="Epoch (UTC)"),
    yaxis=dict(title="Right Ascension (degrees)", anchor="x", domain=[0.55, 1]),
    yaxis2=dict(title="Declination (degrees)", anchor="x", domain=[0, 0.45]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# Save to HTML
output_path = Path(
    "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_analysis/figures/pointing/pointing_comparison_scatter.html"
)
fig.write_html(str(output_path))

print(f"Interactive plot saved to: {output_path.resolve()}")
