import glob
import os
from pathlib import Path

from moviepy import ImageSequenceClip
from PIL import Image

# Define the folder containing the PNG images

delta_time = 5

image_folder = f"/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_analysis/figures/line_profiles/el_az/{delta_time}min/"

# Define the frame rate (frames per second)
frame_rate = 1

# Filter the images based on the start and end times
images = []
image_list = sorted(glob.glob(image_folder + "*.png"))
for img_path in image_list:
    image_folder = Path(image_folder).expanduser().resolve()
    filename = os.path.basename(img_path)
    images.append(img_path)  # Add all images to the list first


standard_size = (4034, 1817)
# Resize images to a standard size
resized_images = []
for i, img_path in enumerate(images):
    with Image.open(img_path) as im:
        im_resized = im.resize(standard_size, Image.LANCZOS)
        temp_path = f"/tmp/resized_{os.path.basename(img_path)}"
        im_resized.save(temp_path)
        resized_images.append(temp_path)
        print(f"Resized image {i + 1}/{len(images)}", end="\r")

clip = ImageSequenceClip(resized_images, fps=frame_rate)

# Write the video file
output_folder = "../movies/"
output_folder = Path(output_folder).expanduser().resolve()
output_folder.mkdir(parents=True, exist_ok=True)
output_file = f"20250316_lunar_coordinate_line_profiles__{delta_time}min_{frame_rate}.mp4"

clip.write_videofile(output_folder / output_file, codec="libx264")


print(f"Video saved as {output_file}")
