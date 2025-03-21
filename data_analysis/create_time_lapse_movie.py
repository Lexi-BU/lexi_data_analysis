import os
from datetime import datetime
from pathlib import Path

from moviepy import ImageSequenceClip

# Define the folder containing the PNG images
image_folder = "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_pipeline/figures/1_min/"

# Define the start and end times
start_time = "2025-03-06_00-00-00"
end_time = "2025-03-10_00-00-00"

# Define the frame rate (frames per second)
frame_rate = 10

# Convert the start and end times to datetime objects for comparison
start_time_dt = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S")
end_time_dt = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S")


# Function to extract the timestamp from the filename
def extract_timestamp(filename):
    # Assuming the filename format is x_cm_vs_y_cm_2D_histogram_2025-03-03_00-20-00_2025-03-03_00-29-59.png
    parts = filename.split("_")
    timestamp_str = f"{parts[7]}_{parts[8]}"
    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")


# Filter the images based on the start and end times
images = []
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(".png"):
        timestamp = extract_timestamp(filename)
        if start_time_dt <= timestamp <= end_time_dt:
            images.append(os.path.join(image_folder, filename))

# Create the video clip
clip = ImageSequenceClip(images, fps=frame_rate)

# Write the video file
output_folder = "../movies/"
output_folder = Path(output_folder).expanduser().resolve()
output_folder.mkdir(parents=True, exist_ok=True)
output_file = f"output_video_{start_time}_{end_time}.mp4"
output_file_gif = f"output_video_{start_time}_{end_time}_{frame_rate}.gif"

clip.write_videofile(output_folder / output_file, codec="libx264")
# clip.write_gif(output_folder / output_file_gif, fps=frame_rate)

print(f"Video saved as {output_file}")
