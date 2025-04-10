import importlib
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

importlib.reload(lexi_functions)


input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time": "2025-03-07T06:05:00Z",
    "end_time": "2025-03-07T06:35:00Z",
    "bins": 200,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
}

read_data = False
if read_data:
    hist, xedges, yedges, ra_median, dec_median = lexi_functions.get_single_histogram_array(
        **input_dict
    )

# Compute the 2d FFT of the histogram
fft_hist = np.fft.fft2(hist)
fft_hist_shifted = np.fft.fftshift(fft_hist)  # Shift zero frequency component to center
magnitude_spectrum = np.abs(fft_hist_shifted)

# Filter out low frequencies
# Define a threshold for filtering low frequencies
low_threshold = 0.1 * np.max(magnitude_spectrum)  # 10% of the maximum magnitude
filtered_fft_hist = np.where(magnitude_spectrum > low_threshold, fft_hist_shifted, 0)
# Inverse FFT to get the filtered histogram back
filtered_hist = np.fft.ifft2(np.fft.ifftshift(filtered_fft_hist)).real

# Filter out high frequencies
# Define a threshold for filtering high frequencies
high_threshold = 0.1 * np.max(magnitude_spectrum)  # 10% of the maximum magnitude
filtered_high_fft_hist = np.where(magnitude_spectrum < high_threshold, fft_hist_shifted, 0)
# Inverse FFT to get the filtered histogram back
filtered_high_hist = np.fft.ifft2(np.fft.ifftshift(filtered_high_fft_hist)).real

higher_threshold = 0.65 * np.max(magnitude_spectrum)
lower_threshold = 0.35 * np.max(magnitude_spectrum)
# Get the filtedred histogram between the lower and higher thresholds]
filtered_higher_fft_hist = np.where(
    (magnitude_spectrum > lower_threshold) & (magnitude_spectrum < higher_threshold),
    fft_hist_shifted,
    0,
)
# filtered_higher_fft_hist = np.where(magnitude_spectrum < higher_threshold, fft_hist_shifted, 0)
# Inverse FFT to get the filtered histogram back for higher threshold
filtered_higher_hist = np.fft.ifft2(np.fft.ifftshift(filtered_higher_fft_hist)).real

# Plotting the results
title_fontsize = 14
plt.rcParams.update({"font.size": 12})  # Set default font size for all plots
# Use dark background for better visibility
plt.style.use("dark_background")
# Plot all the histograms and FFT results in a single figure
fig, axs = plt.subplots(3, 2, figsize=(12, 18))
plt.subplots_adjust(hspace=0.1, wspace=0.1)  # Adjust spacing between subplots

# Original Histogram
axs[0, 0].imshow(
    hist.T,
    origin="lower",
    extent=input_dict["bin_range"],
    cmap="inferno",
    norm=mpl.colors.Normalize(),
)
axs[0, 0].set_title("Original Histogram")
axs[0, 0].set_xlabel(input_dict["x_key"])
axs[0, 0].set_ylabel(input_dict["y_key"])

# Add colorbar for the original histogram
cbar = fig.colorbar(
    axs[0, 0].images[0],
    ax=axs[0, 0],
    orientation="vertical",
    pad=0.02,
    aspect=70,
    fraction=0.02,
    shrink=0.7,
)
# Set the label for the colorbar at the top
cbar.set_label("Cts/s", rotation=270, labelpad=10)

# FFT Magnitude Spectrum
axs[0, 1].imshow(
    magnitude_spectrum.T,
    origin="lower",
    extent=input_dict["bin_range"],
    cmap="inferno",
    norm=mpl.colors.LogNorm(),
)
axs[0, 1].set_title("FFT Magnitude Spectrum", fontsize=title_fontsize)
axs[0, 1].set_xlabel("Frequency X (Hz)")
axs[0, 1].set_ylabel("Frequency Y (Hz)")
# Add colorbar for the FFT magnitude spectrum
cbar = fig.colorbar(
    axs[0, 1].images[0],
    ax=axs[0, 1],
    orientation="vertical",
    pad=0.02,
    aspect=70,
    fraction=0.02,
    shrink=0.7,
)
cbar.set_label("Magnitude", rotation=270, labelpad=15)
# Filtered FFT Magnitude Spectrum
axs[1, 1].imshow(
    np.abs(filtered_fft_hist).T,
    origin="lower",
    extent=input_dict["bin_range"],
    cmap="inferno",
    norm=mpl.colors.LogNorm(),
)
axs[1, 1].set_title("Filtered FFT Magnitude Spectrum", fontsize=title_fontsize)
axs[1, 1].set_xlabel("Frequency X (Hz)")
axs[1, 1].set_ylabel("Frequency Y (Hz)")
# Add colorbar for the filtered FFT magnitude spectrum
cbar = fig.colorbar(
    axs[1, 1].images[0],
    ax=axs[1, 1],
    orientation="vertical",
    pad=0.02,
    aspect=70,
    fraction=0.02,
    shrink=0.7,
)
cbar.set_label("Magnitude", rotation=270, labelpad=15)
# Filtered Histogram (Low Frequencies)
axs[1, 0].imshow(
    filtered_hist.T,
    origin="lower",
    extent=input_dict["bin_range"],
    cmap="inferno",
    norm=mpl.colors.Normalize(),
)
axs[1, 0].set_title("Filtered Histogram (Low Frequencies)", fontsize=title_fontsize)
axs[1, 0].set_xlabel(input_dict["x_key"])
axs[1, 0].set_ylabel(input_dict["y_key"])
# Add colorbar for the filtered low frequencies histogram
cbar = fig.colorbar(
    axs[1, 0].images[0],
    ax=axs[1, 0],
    orientation="vertical",
    pad=0.02,
    aspect=70,
    fraction=0.02,
    shrink=0.7,
)
cbar.set_label("Cts/s", rotation=270, labelpad=15)
# Filtered Histogram (High Frequencies)
axs[2, 0].imshow(
    filtered_high_hist.T,
    origin="lower",
    extent=input_dict["bin_range"],
    cmap="inferno",
    norm=mpl.colors.Normalize(),
)
axs[2, 0].set_title("Filtered Histogram (High Frequencies)", fontsize=title_fontsize)
axs[2, 0].set_xlabel(input_dict["x_key"])
axs[2, 0].set_ylabel(input_dict["y_key"])
# Add colorbar for the filtered high frequencies histogram
cbar = fig.colorbar(
    axs[2, 0].images[0],
    ax=axs[2, 0],
    orientation="vertical",
    pad=0.02,
    aspect=70,
    fraction=0.02,
    shrink=0.7,
)
cbar.set_label("Cts/s", rotation=270, labelpad=15)

# Filtered Histogram (Higher Frequencies)
axs[2, 1].imshow(
    filtered_higher_hist.T,
    origin="lower",
    extent=input_dict["bin_range"],
    cmap="inferno",
    norm=mpl.colors.Normalize(),
)
axs[2, 1].set_title(
    f"Filtered Histogram between {lower_threshold:.2f} and {higher_threshold:.2f} of FFT Magnitude",
    fontsize=title_fontsize,
)
axs[2, 1].set_xlabel(input_dict["x_key"])
axs[2, 1].set_ylabel(input_dict["y_key"])
# Add colorbar for the filtered higher frequencies histogram
cbar = fig.colorbar(
    axs[2, 1].images[0],
    ax=axs[2, 1],
    orientation="vertical",
    pad=0.02,
    aspect=70,
    fraction=0.02,
    shrink=0.7,
)
cbar.set_label("Cts/s", rotation=270, labelpad=15)

# In the axs[2, 1] position, print the ra and dec median values
# axs[2, 1].axis("off")  # Hide the empty subplot
# axs[2, 1].text(
#     0.5,
#     0.5,
#     f"LEXI look direction values:\n\nRA: {ra_median:.4f}°\nDec: {dec_median:.4f}°"
#     + f"\n\nPotentially looking at:\n\n"
#     # + "Magenetopause",
#     + "Blank sky (after sunset)",
#     horizontalalignment="center",
#     verticalalignment="center",
#     fontsize=title_fontsize,
#     color="w",
#     transform=axs[2, 1].transAxes,
#     bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.5"),
# )

# Set the overall title font size
# Add the overall title for the figure
fig.suptitle(
    f"FFT Histogram Analysis from {input_dict['start_time']} to {input_dict['end_time']}",
    fontsize=title_fontsize + 2,
    y=1.0,  # Adjust y position to avoid overlap with subplots
)
# Adjust layout to prevent overlap
for ax in axs.flat:
    ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25, color="c")
    # Add tickmarks inside the plots on all sides
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("w")
    ax.tick_params(
        which="both",
        direction="in",
        length=6,
        width=0.5,
        colors="w",
        grid_color="c",
        left=True,
        right=True,
        top=True,
        bottom=True,
    )
    # Set the maximum number of ticks for both axes to avoid clutter
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Maximum 5 ticks on x-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Maximum 5 ticks on y-axis
# Ensure the layout is tight to avoid overlapping
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to make space for the suptitle
# Final adjustment to the layout
plt.tight_layout()
# Save the figure
output_folder = "../figures/fft_analysis/"
Path(output_folder).mkdir(parents=True, exist_ok=True)
fig_name = f"fft_histogram_analysis_{input_dict['start_time'].replace(':', '-')}_to_{input_dict['end_time'].replace(':', '-')}.png"
fig.savefig(Path(output_folder) / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print("FFT Histogram Analysis complete. Figure saved.")
