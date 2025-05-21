import importlib
import pickle
import warnings
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import map_coordinates

importlib.reload(lexi_functions)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
np.seterr(divide="ignore", invalid="ignore")  # Suppress division by zero warnings

rot_angle = 13.7
input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time": "2025-03-16T19:45:00Z",
    "end_time": "2025-03-16T21:15:00Z",
    "bins": 200,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
    "rotate_data": True,
    "rotation_angle": rot_angle,
}

read_data = False
normalize_against_ground = True

if "hist" not in locals() or "xedges" not in locals() or "yedges" not in locals() or read_data:
    org_hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
        **input_dict
    )
    save_folder = Path("../data/")
    save_folder.mkdir(parents=True, exist_ok=True)
    file_name = f"line_profile_histogram_data_moon_coordinate_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}.pkl"
    save_file = save_folder / file_name
    with open(save_file, "wb") as f:
        pickle.dump(
            {
                "hist": org_hist,
                "xedges": xedges,
                "yedges": yedges,
                "ra_median": ra_median,
                "dec_median": dec_median,
                "start_time": input_dict["start_time"],
                "end_time": input_dict["end_time"],
                "bins": input_dict["bins"],
                "bin_range": input_dict["bin_range"],
                "time_normalization": input_dict["time_normalization"],
                "x_key": input_dict["x_key"],
                "y_key": input_dict["y_key"],
            },
            f,
        )

if normalize_against_ground:
    ground_file_name = (
        "../data/ground_histogram_data_20240523_224500Z_20240530_024500Z_rot_angle_13.7.pkl"
    )
    with open(ground_file_name, "rb") as f:
        ground_data = pickle.load(f)

    ground_hist = ground_data["hist"]
    ground_hist = np.where(ground_hist == 0, np.nan, ground_hist)
    ground_xedges = ground_data["xedges"]
    ground_yedges = ground_data["yedges"]

    hist = org_hist / ground_hist
else:
    hist = org_hist

xcenters = 0.5 * (ground_xedges[:-1] + ground_xedges[1:])
ycenters = 0.5 * (ground_yedges[:-1] + ground_yedges[1:])
X, Y = np.meshgrid(xcenters, ycenters)
radius = np.sqrt(X**2 + Y**2)
mask = radius <= 0.04

hist_masked = hist.copy()
hist_masked[~mask] = np.nan

new_xedges = ground_xedges * 112.5 + 4.5
new_yedges = ground_yedges * 112.5 + 4.5

# Define scaled and unscaled coordinates
scaled_radius = 0.04 * 112.5
y_list = np.linspace(-0.04, 0.04, 10)
scaled_y_list = y_list * 112.5 + 4.5
scaled_x_centers = xcenters * 112.5 + 4.5

# ... [Keep all your previous imports and data processing code] ...

# Create the figure with subplots
fig = make_subplots(
    rows=1, cols=2, subplot_titles=("2D Histogram", "Line Profiles"), horizontal_spacing=0.15
)

# 1. Add the heatmap (left subplot)
fig.add_trace(
    go.Heatmap(
        x=new_xedges,
        y=new_yedges,
        z=hist_masked.T,
        colorscale="Viridis",
        zmin=np.nanmin(hist_masked[hist_masked > 0]),
        zmax=np.nanmax(hist_masked),
        zauto=False,
        colorbar=dict(title="Normalized Counts"),
        hoverongaps=False,
    ),
    row=1,
    col=1,
)

# 2. Add the circle outline (left subplot)
fig.add_shape(
    type="circle",
    xref="x1",
    yref="y1",
    x0=4.5 - scaled_radius,
    y0=4.5 - scaled_radius,
    x1=4.5 + scaled_radius,
    y1=4.5 + scaled_radius,
    line_color="red",
    line_dash="dash",
)

# 3. Store line profile data and add traces
line_profiles = []
for i, y_scaled in enumerate(scaled_y_list):
    # Get the data for this line profile
    y_bin_index = np.argmin(np.abs(ycenters - y_list[i]))
    hist_slice = hist_masked[y_bin_index, :]

    # Add the line profile (right subplot)
    fig.add_trace(
        go.Scatter(
            x=scaled_x_centers,
            y=hist_slice,
            name=f"{y_scaled:.1f}°",
            mode="lines",
            line=dict(width=2, color="blue"),
            opacity=0.1,
            showlegend=True,
            legendgroup="profiles",
        ),
        row=1,
        col=2,
    )

    # Add an INVISIBLE hover detector (left subplot)
    fig.add_trace(
        go.Scatter(
            x=[0, 9],  # Full width of the plot
            y=[y_scaled, y_scaled],
            mode="lines",
            line=dict(width=0),  # Invisible
            hoverinfo="text",
            text=f"Azimuth: {y_scaled:.1f}°",
            showlegend=False,
            customdata=[i],  # Index of the corresponding line profile
            opacity=0,  # Fully transparent
        ),
        row=1,
        col=1,
    )

# 4. Update layout
fig.update_layout(
    title_text=f"Histogram Analysis from {input_dict['start_time']} to {input_dict['end_time']}",
    hovermode="closest",
    width=1200,
    height=600,
    xaxis1=dict(title="Altitude (degrees)", range=[0, 9]),
    yaxis1=dict(title="Azimuth (degrees)", range=[0, 9], scaleanchor="x1", scaleratio=1),
    xaxis2=dict(title="Altitude (degrees)"),
    yaxis2=dict(title="Normalized Counts", type="log"),
    legend=dict(title="Azimuth Angle"),
    margin=dict(l=50, r=50, b=50, t=100),
    template="plotly_white",
)

# 5. JavaScript for hover interaction
custom_js = """
document.addEventListener('DOMContentLoaded', function() {
    const plotEl = document.querySelector('.plotly-graph-div');
    
    plotEl.on('plotly_hover', function(data) {
        // Check if we're hovering over an invisible trace (left subplot)
        if (data.points[0].data.opacity === 0) {
            const profileIndex = data.points[0].data.customdata[0];
            
            // Update all line profiles (right subplot)
            const numProfiles = %d;
            const updates = {
                'opacity': Array(numProfiles).fill(0.1),
                'line.width': Array(numProfiles).fill(2),
                'line.color': Array(numProfiles).fill('blue')
            };
            
            // Highlight the selected profile
            updates.opacity[profileIndex] = 1.0;
            updates.line.width[profileIndex] = 4;
            updates.line.color[profileIndex] = 'red';
            
            // Apply updates to the right subplot traces
            const lineProfileIndices = Array.from({length: numProfiles}, (_, i) => i * 2 + 1);
            Plotly.restyle(plotEl, updates, lineProfileIndices);
        }
    });
    
    plotEl.on('plotly_unhover', function() {
        // Reset all line profiles
        const numProfiles = %d;
        const updates = {
            'opacity': Array(numProfiles).fill(0.1),
            'line.width': Array(numProfiles).fill(2),
            'line.color': Array(numProfiles).fill('blue')
        };
        const lineProfileIndices = Array.from({length: numProfiles}, (_, i) => i * 2 + 1);
        Plotly.restyle(plotEl, updates, lineProfileIndices);
    });
});
""" % (
    len(scaled_y_list),
    len(scaled_y_list),
)

# 6. Save the interactive plot
save_fig_folder = Path("../figures/alt/az_hist/")
save_fig_folder.mkdir(parents=True, exist_ok=True)
fig_file_name = (
    f"interactive_plot_{input_dict['start_time'].replace(':', '').replace('-', '').replace('T', '_')}_"
    f"{input_dict['end_time'].replace(':', '').replace('-', '').replace('T', '_')}_rot_angle_{rot_angle}.html"
)

fig.write_html(
    save_fig_folder / fig_file_name,
    config={"responsive": True},
    include_plotlyjs="cdn",
    post_script=custom_js,
)
