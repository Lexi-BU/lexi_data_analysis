import glob
import os
from datetime import datetime
from pathlib import Path

from moviepy import ImageSequenceClip
from PIL import Image

# Define the folder containing the PNG images
image_folder = "/home/vetinari/Desktop/git/Lexi-Bu/lexi_data_pipeline/figures/exposure_maps/bg_corrected/from_l2/"

# Define the start and end times
start_time = "2025-03-16-19-45-00"
end_time = "2025-03-16-21-15-00"

# Define the frame rate (frames per second)
frame_rate = 5

# Convert the start and end times to datetime objects for comparison
# start_time_dt = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S")
# end_time_dt = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S")


# Function to extract the timestamp from the filename
def extract_timestamp(filename):
    # Assuming the filename format is x_cm_vs_y_cm_2D_histogram_2025-03-03_00-20-00_2025-03-03_00-29-59.png
    parts = filename.split("_")
    timestamp_str = f"{parts[7]}_{parts[8][:5]}"
    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M")


# Filter the images based on the start and end times
images = []
# image_list = sorted(glob.glob(image_folder + "*.png"))
# for img_path in image_list:
#     image_folder = Path(image_folder).expanduser().resolve()
#     filename = os.path.basename(img_path)
#     images.append(img_path)  # Add all images to the list first
#
# # Sort the images based on name
# images = sorted(images)
base_height = None
base_width = None
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(".png"):
        # timestamp = extract_timestamp(filename)
        # if start_time_dt <= timestamp <= end_time_dt:
        img_path = os.path.join(image_folder, filename)

        # Open and resize the image
        with Image.open(img_path) as img:
            # Choose one of these resizing methods:

            # METHOD 1: Resize to smallest dimensions found
            if base_height is None or base_width is None:
                base_width, base_height = img.size
            resized_img = img.resize((base_width, base_height))

            # METHOD 2: Resize to fixed dimensions (uncomment to use)
            # resized_img = img.resize((1920, 1080))  # 1080p

            # Save temp resized image
            temp_path = f"/tmp/resized_{filename}"
            resized_img.save(temp_path)
            images.append(temp_path)


# Create the video clip
clip = ImageSequenceClip(images, fps=frame_rate)

# Write the video file
output_folder = "../movies/"
output_folder = Path(output_folder).expanduser().resolve()
output_folder.mkdir(parents=True, exist_ok=True)
# output_file = f"output_video_{start_time}_{end_time}.mp4"
# output_file_gif = f"output_video_{start_time}_{end_time}_{frame_rate}.gif"
output_file = f"exposure_time_series_all.mp4"
# output_file_gif = (
#     f"line_profiles_{start_time.replace(':', '-')}_to_{end_time.replace(':', '-')}_{frame_rate}.gif"
# )

clip.write_videofile(output_folder / output_file, codec="libx264")  # , fps=frame_rate)
# clip.write_gif(output_folder / output_file_gif, fps=frame_rate)

# Cleanup temp files
# for temp_img in images:
#     os.remove(temp_img)

print(f"Video saved as {output_file}")
