from pathlib import Path

import imageio
import imageio.v2 as imageio  # imageio v2 API for compatibility
import numpy as np
from PIL import Image
from tqdm import tqdm

version_number = "v0.1"  # Change this to your version number
delta_time = 5  # Change this to your delta time in minutes
# Set your image folder and output path
input_folder = Path(
    f"/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_analysis/figures/line_profiles/el_az_{version_number}/{delta_time}min/"
)

image_extension = "*.png"  # Change if needed, e.g., "*.jpg"
crop_percent = 100  # Crop left side to 40% of width
fps = 5  # Frames per second


output_video = f"{version_number}_{delta_time}min_{fps}fps.mp4"

# Collect image paths
image_paths = sorted(input_folder.glob(image_extension))
if not image_paths:
    raise RuntimeError("No input images found.")

# Determine target size from first image
first_img = Image.open(image_paths[0]).convert("RGB")
w, h = first_img.size
cropped_w = int(w * (crop_percent / 100))
target_size = (cropped_w, h)

# Initialize writer
writer = imageio.get_writer(output_video, fps=fps, macro_block_size=None)

for img_path in tqdm(image_paths, desc="Processing images"):
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    cropped_width = int(width * (crop_percent / 100))
    cropped = img.crop((0, 0, cropped_width, height))

    # Resize to target size (only if different)
    if cropped.size != target_size:
        cropped = cropped.resize(target_size)

    writer.append_data(np.array(cropped))

writer.close()
