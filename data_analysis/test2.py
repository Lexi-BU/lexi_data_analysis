import matplotlib.pyplot as plt
import numpy as np


def get_histogram_values_along_line_both_directions(
        hist, xedges, yedges, theta, x_offset, y_offset
):
    # Convert theta to a direction vector
    dx = np.cos(theta)
    dy = np.sin(theta)

    # Define line endpoints (adjust range if needed)
    line_length = max(xedges[-1] - xedges[0], yedges[-1] - yedges[0]) * 2
    x0 = x_offset - dx * line_length / 2
    x1 = x_offset + dx * line_length / 2
    y0 = y_offset - dy * line_length / 2
    y1 = y_offset + dy * line_length / 2

    # Generate points along the line
    num_points = 1000
    x_line = np.linspace(x0, x1, num_points)
    y_line = np.linspace(y0, y1, num_points)

    # Find bin indices for each point
    x_indices = np.searchsorted(xedges, x_line) - 1
    y_indices = np.searchsorted(yedges, y_line) - 1

    # Clip to valid range
    x_indices = np.clip(x_indices, 0, hist.shape[0] - 1)
    y_indices = np.clip(y_indices, 0, hist.shape[1] - 1)

    # Get unique bin indices the line passes through
    bin_indices = set(zip(x_indices, y_indices))

    # Print values and indices
    # print("Bins and values the line passes through:")
    value = []
    distance = []
    for xi, yi in bin_indices:
        # Compute bin center
        x_center = 0.5 * (xedges[xi] + xedges[xi + 1])
        y_center = 0.5 * (yedges[yi] + yedges[yi + 1])
        dx_bin = x_center - x_offset
        dy_bin = y_center - y_offset
        # Project onto direction vector (theta)
        directed_distance = dx_bin * dx + dy_bin * dy
        distance.append(directed_distance)
        value.append(hist[xi, yi])

    return x_line, y_line, distance, value    
# Make a 2 by 1 subplot
fig, axs = plt.subplots(2, 1, figsize=(8, 10))
# Plot histogram
axs[0].imshow(
    hist.T,
    origin="lower",
    aspect="equal",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap="viridis",
)
axs[0].set_xlim(xedges[0], xedges[-1])
axs[0].set_ylim(yedges[0], yedges[-1])
x_centers = (xedges[:-1] + xedges[1:]) / 2
y_centers = (yedges[:-1] + yedges[1:]) / 2

for i in range(len(x_centers)):
    for j in range(len(y_centers)):
        value_h = hist[i, j]
        if value_h > 0:  # Only display text for non-zero bins
            axs[0].text(
                x_centers[i],
                y_centers[j],
                f"{value_h:.1f}",
                color="k",
                ha="center",
                va="center",
                fontsize=5,
            )
axs[0].set_title("Histogram")
# Plot line on histogram
axs[0].plot(x_line, y_line, color="red", linewidth=2)
# Plot distances
axs[1].scatter(distance, value, color="lime", alpha=0.5)
# At each point, write down the hist value beside it
for dist, val in zip(distance, value):
    axs[1].text(dist, val, f"{val:.1f}", color="w", fontsize=8)
sum_values = sum(value)
axs[1].text(
    0.05,
    0.9,
    f"Sum of Values: {sum_values:.1f}",
    transform=axs[1].transAxes,
    verticalalignment="top",
    horizontalalignment="left",
    color="w",
    fontsize=12,
)
axs[1].set_title("Distance vs Value Along Line")
plt.tight_layout()
plt.savefig(f"../figures/line_profiles_v2/histogram_with_line_{theta:0.1f}.png")
