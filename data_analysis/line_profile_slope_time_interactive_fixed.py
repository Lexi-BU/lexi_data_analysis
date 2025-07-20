import datetime
import glob
import threading
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

# Data folder and file list
folder_name = "../data/line_profile_data/fixed_el_az_data"
file_list = sorted(glob.glob(f"{folder_name}/*.csv"))

# Define sunset times
sunset_start_time = datetime.datetime(2025, 3, 16, 19, 38, 0)
sunset_end_time = datetime.datetime(2025, 3, 16, 20, 44, 0)

# Map integration time labels to file paths
integration_map = {}
for file in file_list:
    integration_time = file.split("/")[-1].split("_")[-2]
    integration_map[integration_time] = file

# Create the Dash app
app = Dash(__name__)
app.title = "Line Profile Slope Viewer (Fixed El/Az Data)"

# Layout with dark theme styling
app.layout = html.Div(
    [
        html.H2("Line Profile Slope over Time", style={"color": "white"}),
        html.Div(
            [
                dcc.Checklist(
                    id="integration-checklist",
                    options=[{"label": k, "value": k} for k in integration_map.keys()],
                    value=[],
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "15px",
                        "color": "white",
                    },
                    inputStyle={"margin-right": "5px"},
                )
            ],
            style={"display": "flex", "flex-wrap": "wrap", "margin-bottom": "20px"},
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
                    style={"display": "inline-block"},
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
    Input("show-average", "value"),
    Input("average-type", "value"),
    Input("window-size", "value"),
)
def update_figure(selected_integrations, show_avg, avg_type, window_minutes):
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

    for i, integration_time in enumerate(selected_integrations):
        file = integration_map[integration_time]
        data = pd.read_csv(file)
        start_time = pd.to_datetime(data["start_time"])
        end_time = pd.to_datetime(data["end_time"])
        middle_time = start_time + (end_time - start_time) / 2
        slope = data["slope"]

        color = color_cycle[i % len(color_cycle)]

        fig.add_trace(
            go.Scatter(
                x=middle_time,
                y=slope,
                mode="markers",
                name=integration_time,
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
                    name=f"{integration_time} ({window_minutes}min avg)",
                    line=dict(dash="dash", color=color),
                    showlegend=False,
                )
            )

    fig.update_layout(
        template="plotly_dark",
        title="Slope vs Time (Select Integration Times)",
        xaxis_title="Time",
        yaxis_title="Slope",
        height=800,
        margin=dict(t=50, r=20, l=20, b=50),
        legend_title_text="Integration Time",
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


def open_browser():
    webbrowser.open_new("http://127.0.0.0:8050")


if __name__ == "__main__":
    # threading.Timer(1.5, open_browser).start()
    # define the server and the port
    app.run(debug=False, host="0.0.0.0", port=8000)
