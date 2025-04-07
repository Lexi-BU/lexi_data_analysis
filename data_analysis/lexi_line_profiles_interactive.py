import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

save_folder = Path("../data/")
save_folder.mkdir(parents=True, exist_ok=True)
save_file = save_folder / "line_profile_data.pkl"
# Load the data
with open(save_file, "rb") as f:
    data = pickle.load(f)
    theta_list = data["theta_list"]
    sum_values = data["sum_values"]
    x_offset = data["x_offset"]
    y_offset = data["y_offset"]
    hist = data["hist"]
    xedges = data["xedges"]
    yedges = data["yedges"]
    ra_median = data["ra_median"]
    dec_median = data["dec_median"]

# Create figure with 2 subplots
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("2D Histogram with Line", "Sum of Histogram Values vs. Theta"),
    horizontal_spacing=0.1,
)

# Add 2D histogram
fig.add_trace(
    go.Heatmap(
        z=hist.T,
        x=np.linspace(xedges[0], xedges[-1], len(xedges) - 1),
        y=np.linspace(yedges[0], yedges[-1], len(yedges) - 1),
        colorscale="Inferno",
        colorbar=dict(title="Counts"),
        showscale=True,
    ),
    row=1,
    col=1,
)

# Add sum values plot
fig.add_trace(
    go.Scatter(
        x=theta_list,
        y=sum_values,
        mode="lines+markers",
        marker=dict(color="lime", size=4),
        line=dict(color="lime"),
        name="Sum of Values",
    ),
    row=1,
    col=2,
)

# Add initial line to histogram (at theta=0)
line_length = 0.2  # Adjust as needed
initial_theta = 0
x_line = np.array(
    [
        -line_length * np.cos(np.radians(initial_theta)),
        line_length * np.cos(np.radians(initial_theta)),
    ]
)
y_line = np.array(
    [
        -line_length * np.sin(np.radians(initial_theta)),
        line_length * np.sin(np.radians(initial_theta)),
    ]
)

line_trace = fig.add_trace(
    go.Scatter(
        x=x_line, y=y_line, mode="lines", line=dict(color="cyan", width=2), showlegend=False
    ),
    row=1,
    col=1,
)

# Update layout
fig.update_layout(
    template="plotly_dark",
    width=1200,
    height=500,
    margin=dict(l=50, r=50, b=50, t=50),
    hovermode="closest",
)

# Update axes properties
fig.update_xaxes(title_text="X", row=1, col=1, range=[-0.1, 0.1], nticks=5)
fig.update_yaxes(title_text="Y", row=1, col=1, range=[-0.1, 0.1], nticks=5)
fig.update_xaxes(title_text="Theta (degrees)", row=1, col=2, range=[0, 180], nticks=5)
fig.update_yaxes(title_text="Sum of Histogram Values", row=1, col=2, nticks=5)

# Add interactivity to update the line when hovering over the right plot
fig.update_traces(hoverinfo="none", hovertemplate=None, row=1, col=2)

# JavaScript for interactivity
fig.update_layout(
    hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    # Add custom JavaScript to update the line when hovering
    annotations=[dict(text="", showarrow=False, xref="paper", yref="paper", x=0, y=0)],
)

# Save the figure
save_theta = np.round(theta_list[0], 1)
save_theta = str(save_theta).zfill(6)
save_folder = Path("../figures/html/")
save_folder.mkdir(parents=True, exist_ok=True)
fig_name = f"20250407_sunset_double_linear_line_profile_sum_values.html"

# Add custom JavaScript for the interactivity
fig.write_html(
    save_folder / fig_name,
    config={"scrollZoom": True, "responsive": True},
    include_plotlyjs="cdn",
    full_html=True,
    auto_open=False,
)
