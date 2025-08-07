from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

# Load data
folder_path = Path("../data/lexi_sw/istp/")
file_name = "istp_themis_c_esa_parameters_2025-03-16T19:30:00_to_2025-03-16T21:15:00_histogram_sum_1min_300.csv"
df = pd.read_csv(folder_path / file_name, index_col=0, parse_dates=True)

# Define fixed time range
start_time = pd.Timestamp("2025-03-16T19:30:00Z")
end_time = pd.Timestamp("2025-03-16T20:50:00Z")
df = df[start_time:end_time]
time_series = df.index
hist_sum = df["hist_sum"]
sw_np, sw_vp, sw_flux = df["sw_np"], df["sw_vp"], df["sw_flux"]


# Helper for correlation
def compute_corrs(ref, shifted):
    aligned = shifted.reindex(ref.index, method="nearest", tolerance=pd.Timedelta(seconds=30))
    df_corr = pd.concat([ref, aligned], axis=1).dropna()
    if len(df_corr) == 0:
        return np.nan, np.nan
    pearson = df_corr.iloc[:, 0].corr(df_corr.iloc[:, 1], method="pearson")
    spearman = spearmanr(df_corr.iloc[:, 0], df_corr.iloc[:, 1], nan_policy="omit").correlation
    return pearson, spearman


# App
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "THEMIS-LEXI Histogram Correlation Viewer",
            style={"color": "white", "backgroundColor": "black"},
        ),
        html.Div(
            [
                html.Label("Shift Density (s)", style={"color": "white"}),
                dcc.Input(id="shift-np", type="number", value=300, style={"color": "black"}),
                html.Label("Shift Velocity (s)", style={"color": "white"}),
                dcc.Input(id="shift-vp", type="number", value=300, style={"color": "black"}),
                html.Label("Shift Flux (s)", style={"color": "white"}),
                dcc.Input(id="shift-flux", type="number", value=300, style={"color": "black"}),
                html.Br(),
                html.Label("Log Scale Toggle", style={"color": "white"}),
                dcc.Checklist(
                    id="log-scale-toggle",
                    options=[{"label": "Log scale", "value": "log"}],
                    value=["log"],
                    labelStyle={"color": "white", "backgroundColor": "black"},
                ),
            ],
            style={"padding": 1, "backgroundColor": "black", "borderRadius": "5px"},
        ),
        dcc.Graph(id="main-graph"),
    ],
    style={"backgroundColor": "black", "height": "90vh", "padding": "1px"},
)


@app.callback(
    Output("main-graph", "figure"),
    Input("shift-np", "value"),
    Input("shift-vp", "value"),
    Input("shift-flux", "value"),
    Input("log-scale-toggle", "value"),
)
def update_graph(shift_np, shift_vp, shift_flux, log_toggle):
    shift_td_np = pd.to_timedelta(shift_np, unit="s")
    shift_td_vp = pd.to_timedelta(shift_vp, unit="s")
    shift_td_flux = pd.to_timedelta(shift_flux, unit="s")

    sw_np_range = sw_np
    sw_vp_range = sw_vp
    sw_flux_range = sw_flux
    hist_sum_range = hist_sum

    hist_np = hist_sum_range.copy()
    hist_np.index = hist_np.index + shift_td_np
    hist_vp = hist_sum_range.copy()
    hist_vp.index = hist_vp.index + shift_td_vp
    hist_flux = hist_sum_range.copy()
    hist_flux.index = hist_flux.index + shift_td_flux

    log_y = "log" in log_toggle

    # Compute correlations
    pearson_np, spearman_np = compute_corrs(sw_np_range, hist_np)
    pearson_vp, spearman_vp = compute_corrs(sw_vp_range, hist_vp)
    pearson_flux, spearman_flux = compute_corrs(sw_flux_range, hist_flux)

    # Create figure with subplots and secondary y-axes
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        subplot_titles=("Density", "Velocity", "Flux", "Histogram Overlays"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{}]],
    )

    # Set the background color for the entire figure
    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black")

    # Add traces for each subplot
    # 1. Density plot
    fig.add_trace(
        go.Scatter(x=sw_np_range.index, y=sw_np_range, name="Density", line=dict(color="blue")),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hist_np.index, y=hist_np, name=f"Hist (shift {shift_np}s)", line=dict(color="red")
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # 2. Velocity plot
    fig.add_trace(
        go.Scatter(x=sw_vp_range.index, y=sw_vp_range, name="Velocity", line=dict(color="orange")),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hist_vp.index, y=hist_vp, name=f"Hist (shift {shift_vp}s)", line=dict(color="red")
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # 3. Flux plot
    fig.add_trace(
        go.Scatter(x=sw_flux_range.index, y=sw_flux_range, name="Flux", line=dict(color="green")),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hist_flux.index,
            y=hist_flux,
            name=f"Hist (shift {shift_flux}s)",
            line=dict(color="red"),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    # 4. Histogram overlays
    fig.add_trace(
        go.Scatter(x=hist_sum.index, y=hist_sum, name="Hist (original)", line=dict(color="white")),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=hist_np.index, y=hist_np, name="Hist NP", line=dict(color="blue")),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=hist_vp.index, y=hist_vp, name="Hist VP", line=dict(color="orange")),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=hist_flux.index, y=hist_flux, name="Hist Flux", line=dict(color="green")),
        row=4,
        col=1,
    )

    # Add styled annotations for each subplot
    def add_corr_annotation(fig, row, pearson, spearman):
        # Get the y-axis range for proper positioning
        y_range = (
            fig.layout[f"yaxis{row}"].range if row == 1 else fig.layout[f"yaxis{row*2-1}"].range
        )
        y_pos = y_range

        fig.add_annotation(
            x=start_time,
            y=y_pos,
            text=f"Pearson: {pearson:.2f}<br>Spearman: {spearman:.2f}",
            showarrow=False,
            font=dict(color="white", size=10),
            xanchor="left",
            yanchor="bottom",
            bgcolor="black",
            bordercolor="magenta",
            borderwidth=1,
            borderpad=4,
            row=row,
            col=1,
        )

    # Add annotations to each subplot
    add_corr_annotation(fig, 1, pearson_np, spearman_np)
    add_corr_annotation(fig, 2, pearson_vp, spearman_vp)
    add_corr_annotation(fig, 3, pearson_flux, spearman_flux)

    # Adjust ranges if needed
    max_density = sw_np_range.max()
    min_density = sw_np_range.min()
    max_hist = hist_np.max()
    min_hist = hist_np.min()
    max_velocity = sw_vp_range.max()
    min_velocity = sw_vp_range.min()
    max_flux = sw_flux_range.max()
    min_flux = sw_flux_range.min()

    if log_y:
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=2, col=1)
        fig.update_yaxes(type="log", row=3, col=1)
        fig.update_yaxes(type="log", row=4, col=1)

        fig.update_yaxes(
            range=[np.log10(min_density), np.log10(max_density)], row=1, col=1, secondary_y=False
        )
        fig.update_yaxes(
            range=[np.log10(min_hist), np.log10(max_hist)], row=1, col=1, secondary_y=True
        )

        fig.update_yaxes(
            range=[np.log10(min_velocity), np.log10(max_velocity)], row=2, col=1, secondary_y=False
        )
        fig.update_yaxes(
            range=[np.log10(min_hist), np.log10(max_hist)], row=2, col=1, secondary_y=True
        )

        fig.update_yaxes(
            range=[np.log10(min_flux), np.log10(max_flux)], row=3, col=1, secondary_y=False
        )
        fig.update_yaxes(
            range=[np.log10(min_hist), np.log10(max_hist)], row=3, col=1, secondary_y=True
        )

        fig.update_yaxes(range=[np.log10(min_hist), np.log10(max_hist)], row=4, col=1)
    else:
        fig.update_yaxes(range=[min_density, max_density], row=1, col=1, secondary_y=False)
        fig.update_yaxes(range=[min_hist, max_hist], row=1, col=1, secondary_y=True)

        fig.update_yaxes(range=[min_velocity, max_velocity], row=2, col=1, secondary_y=False)
        fig.update_yaxes(range=[min_hist, max_hist], row=2, col=1, secondary_y=True)

        fig.update_yaxes(range=[min_flux, max_flux], row=3, col=1, secondary_y=False)
        fig.update_yaxes(range=[min_hist, max_hist], row=3, col=1, secondary_y=True)

        fig.update_yaxes(range=[min_hist, max_hist], row=4, col=1)

    fig.update_xaxes(range=[start_time, end_time])

    fig.update_layout(
        height=800,
        title="Interactive Shift & Correlation Viewer",
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="v", y=0.9),
    )
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Dash app on custom host/port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    args = parser.parse_args()
    app.run(debug=False, host=args.host, port=args.port)
