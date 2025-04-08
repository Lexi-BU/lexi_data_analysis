import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots

# Load the data
save_file = Path("../data/line_profile_data_0.0_180.0_181.pkl")
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

# Create a 5 point rolling average for sum_values
window_size = 5
sum_values_rolling = np.convolve(sum_values, np.ones(window_size) / window_size, mode="same")

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
        colorbar=dict(
            title="Counts/s",
            x=0.5,
            y=0.5,
            yanchor="middle",
            xanchor="right",
            len=0.95,
            thickness=10,
        ),
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
        mode="markers+lines",
        marker=dict(color="lime", size=4),
        line=dict(color="lime", width=0.1, dash="dashdot"),
        name="Value",
    ),
    row=1,
    col=2,
)

# Add the rolling average line
fig.add_trace(
    go.Scatter(
        x=theta_list,
        y=sum_values_rolling,
        mode="lines",
        line=dict(color="white", width=1, dash="dash"),
        name="Rolling Average",
        opacity=0.7,
    ),
    row=1,
    col=2,
)

# Fit a best fit line between theta_list and sum_values
coefficients = np.polyfit(theta_list, sum_values, 1)
fit_line = np.polyval(coefficients, theta_list)
fig.add_trace(
    go.Scatter(
        x=theta_list,
        y=fit_line,
        mode="lines",
        line=dict(color="orange", width=2),
        name="Best Fit Line",
    ),
    row=1,
    col=2,
)

# Display the equation of the best fit line
equation_text = f"best fit line: y = {coefficients[0]:.4f}x + {coefficients[1]:.4f}"
fig.add_annotation(
    text=equation_text,
    xref="x2 domain",
    yref="y2 domain",
    x=0.05,
    y=0.95,
    showarrow=False,
    font=dict(size=12, color="orange"),
    row=1,
    col=2,
)

# Initial cyan line in left plot (theta = 0)
line_length = 0.2
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
fig.add_trace(
    go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        line=dict(color="cyan", width=2),
        showlegend=False,
    ),
    row=1,
    col=1,
)

# Add vertical line placeholder in right plot (initially at theta=0)
fig.add_trace(
    go.Scatter(
        x=[initial_theta, initial_theta],
        y=[min(sum_values), max(sum_values)],
        mode="lines",
        line=dict(color="white", width=1, dash="dot"),
        showlegend=False,
    ),
    row=1,
    col=2,
)

# Update layout
fig.update_layout(
    template="plotly_dark",
    width=1200,
    height=500,
    margin=dict(l=50, r=50, b=50, t=50),
    hovermode="x unified",
)

# Axis configuration
fig.update_xaxes(title_text="X", row=1, col=1, range=[-0.1, 0.1], nticks=5)
fig.update_yaxes(title_text="Y", row=1, col=1, range=[-0.1, 0.1], nticks=5)
fig.update_xaxes(title_text="Theta (degrees)", row=1, col=2, range=[0, 181], nticks=5)
fig.update_yaxes(title_text="Sum of Histogram Values", row=1, col=2, nticks=5)

# Save path
save_folder = Path("../figures/html/")
save_folder.mkdir(parents=True, exist_ok=True)
fig_name = f"{save_file.stem}_interactive.html"
fig_path = save_folder / fig_name

# Generate HTML
html_str = pio.to_html(
    fig,
    config={"scrollZoom": True},
    include_plotlyjs="cdn",
    full_html=True,
)

# Inject JavaScript for interactivity
soup = BeautifulSoup(html_str, "html.parser")
script_tag = soup.new_tag("script")
script_tag.string = """
document.addEventListener("DOMContentLoaded", function() {
    var plotDiv = document.querySelector(".plotly-graph-div");

    function updateLines(theta) {
        var line_length = 0.2;

        // Compute line coordinates in left plot
        var x_line = [
            -line_length * Math.cos(theta * Math.PI / 180),
            line_length * Math.cos(theta * Math.PI / 180)
        ];
        var y_line = [
            -line_length * Math.sin(theta * Math.PI / 180),
            line_length * Math.sin(theta * Math.PI / 180)
        ];

        // Update cyan line in left plot (trace index 4)
        Plotly.restyle(plotDiv, {
            'x': [x_line],
            'y': [y_line]
        }, [4]);

        // Update vertical line in right plot (trace index 5)
        var y_min = Math.min.apply(null, plotDiv.data[1].y);
        var y_max = Math.max.apply(null, plotDiv.data[1].y);
        Plotly.restyle(plotDiv, {
            'x': [[theta, theta]],
            'y': [[y_min, y_max]]
        }, [5]);
    }

    plotDiv.on('plotly_hover', function(data) {
        if (data.points[0].xaxis._id === 'x2') {
            var theta = data.points[0].x;
            updateLines(theta);
        }
    });
});
"""
soup.body.append(script_tag)

# Save updated HTML
fig_path.write_text(str(soup), encoding="utf-8")

# Print that the file has been saved
print(f"Interactive figure saved to: {fig_path}")
