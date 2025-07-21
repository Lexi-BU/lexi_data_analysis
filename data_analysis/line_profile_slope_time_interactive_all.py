import datetime
import glob
import threading
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, dcc, html

# Data folder and file list
folder_name = "../data/line_profile_data/"
file_list = sorted(glob.glob(f"{folder_name}/*v0.*.csv"))

# Define sunset times
sunset_start_time = datetime.datetime(2025, 3, 16, 19, 38, 0)
sunset_end_time = datetime.datetime(2025, 3, 16, 20, 44, 0)

# Map integration time and version to file paths
integration_version_map = {}
for file in file_list:
    filename = Path(file).name
    integration_time = filename.split("_")[-3]
    version = filename.split("_")[-1].replace(".csv", "")
    key = (integration_time, version)
    integration_version_map[key] = file

# Extract unique integration times and versions
integration_times = sorted({k[0] for k in integration_version_map.keys()})
versions = sorted({k[1] for k in integration_version_map.keys()})

# Create the Dash app
app = Dash(__name__)
app.title = "Line Profile Slope Viewer"

# Layout with dark theme styling
app.layout = html.Div(
    [
        html.H2("Line Profile Slope over Time", style={"color": "white"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Data Version:", style={"color": "white"}),
                        dcc.Checklist(
                            id="version-checklist",
                            options=[{"label": v, "value": v} for v in versions],
                            value=["v0.1"],
                            labelStyle={
                                "display": "inline-block",
                                "margin-right": "15px",
                                "color": "white",
                            },
                            inputStyle={"margin-right": "5px"},
                        ),
                    ],
                    style={"margin-bottom": "15px"},
                ),
                html.Label("Select Integration Time(s):", style={"color": "white"}),
                dcc.Checklist(
                    id="integration-checklist",
                    options=[{"label": k, "value": k} for k in integration_times],
                    value=[],
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "15px",
                        "color": "white",
                    },
                    inputStyle={"margin-right": "5px"},
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Checklist(
                    id="show-average",
                    options=[{"label": " Show Running Average", "value": "show"}],
                    value=["show"],
                    labelStyle={"color": "white", "margin-right": "15px"},
                    inputStyle={"margin-right": "5px"},
                ),
                dcc.RadioItems(
                    id="average-type",
                    options=[
                        {"label": " Mean", "value": "mean"},
                        {"label": " Median", "value": "median"},
                    ],
                    value="mean",
                    labelStyle={"color": "white", "margin-right": "15px"},
                    inputStyle={"margin-right": "5px"},
                ),
                dcc.Input(
                    id="window-size",
                    type="number",
                    value=5,
                    min=1,
                    step=1,
                    style={"width": "80px", "margin-left": "10px"},
                    placeholder="Window",
                ),
            ],
            style={
                "display": "flex",
                "justify-content": "flex-end",
                "align-items": "center",
                "margin-bottom": "10px",
            },
        ),
        dcc.Graph(id="slope-plot"),
    ],
    style={"backgroundColor": "#1e1e1e", "padding": "20px"},
)


@app.callback(
    Output("slope-plot", "figure"),
    Input("integration-checklist", "value"),
    Input("version-checklist", "value"),
    Input("show-average", "value"),
    Input("average-type", "value"),
    Input("window-size", "value"),
)
def update_figure(selected_integrations, selected_versions, show_avg, avg_type, window_minutes):
    show_avg = "show" in show_avg
    fig = go.Figure()

    # Always include sunset markers
    fig.add_vline(x=sunset_start_time, line=dict(color="orange", dash="dash"), layer="below")
    fig.add_vline(x=sunset_end_time, line=dict(color="red", dash="dash"), layer="below")
    fig.add_vrect(
        x0=sunset_start_time,
        x1=sunset_end_time,
        fillcolor="yellow",
        opacity=0.05,
        layer="below",
        line_width=0,
        annotation_text="Sunset Period",
        annotation_position="top left",
    )

    import plotly.express as px

    color_cycle = px.colors.qualitative.Plotly

    trace_idx = 0
    for integration_time in selected_integrations:
        for version in selected_versions:
            key = (integration_time, version)
            if key not in integration_version_map:
                continue
            file = integration_version_map[key]
            data = pd.read_csv(file)

            start_time = pd.to_datetime(data["start_time"])
            end_time = pd.to_datetime(data["end_time"])
            middle_time = start_time + (end_time - start_time) / 2
            slope = data["slope"]

            color = color_cycle[trace_idx % len(color_cycle)]
            trace_idx += 1

            label = f"{integration_time} ({version})"

            fig.add_trace(
                go.Scatter(
                    x=middle_time,
                    y=slope,
                    mode="markers",
                    name=label,
                    marker=dict(size=6, color=color),
                )
            )

            if show_avg:
                df_temp = pd.DataFrame({"slope": slope})
                df_temp["time"] = middle_time
                df_temp.set_index("time", inplace=True)
                df_temp.index = pd.to_datetime(df_temp.index)

                window_str = f"{int(window_minutes)}min"
                if avg_type == "mean":
                    slope_avg = df_temp["slope"].rolling(window=window_str, min_periods=1).mean()
                else:
                    slope_avg = df_temp["slope"].rolling(window=window_str, min_periods=1).median()

                fig.add_trace(
                    go.Scatter(
                        x=slope_avg.index,
                        y=slope_avg.values,
                        mode="lines",
                        name=f"{label} ({window_minutes}min avg)",
                        line=dict(dash="dash", color=color),
                        showlegend=False,
                    )
                )

    fig.update_layout(
        template="plotly_dark",
        title="Slope vs Time (Select Integration Time and Version)",
        xaxis_title="Time",
        yaxis_title="Slope",
        height=600,
        margin=dict(t=50, r=20, l=20, b=50),
        legend_title_text="Integration Time and Version",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="#1e1e1e",
            bordercolor="white",
            borderwidth=1,
        ),
        xaxis=dict(
            showline=True,
            linecolor="white",
            range=[
                datetime.datetime(2025, 3, 16, 19, 0, 0),
                datetime.datetime(2025, 3, 16, 21, 15, 0),
            ],
        ),
        yaxis=dict(
            showline=True,
            linecolor="white",
            range=[-0.4, 0.09],
        ),
    )

    return fig


def save_figure(selected_integrations, selected_versions, show_avg, avg_type, window_minutes):
    fig = update_figure(
        selected_integrations, selected_versions, show_avg, avg_type, window_minutes
    )
    figure_folder = Path("../figures/line_profile_slope_interactive/")
    figure_folder.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=figure_folder / "slope_plot.html", include_plotlyjs="cdn")
    pio.write_image(fig, file=figure_folder / "slope_plot.png", width=1200, height=800)
    print("Saved slope_plot.html and slope_plot.png")


def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")


if __name__ == "__main__":
    selected_integrations = integration_times
    selected_versions = versions
    show_avg = ["show"]
    avg_type = "mean"
    window_minutes = 5

    save_figure(selected_integrations, selected_versions, show_avg, avg_type, window_minutes)

    # threading.Timer(1.5, open_browser).start()
    app.run(debug=False)
