import inspect
import os
from datetime import datetime
from pathlib import Path

from moviepy import ImageSequenceClip
from PIL import Image

# Inputs
image_folder = "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_analysis/figures/line_profiles/bg_corrected/from_l2/az_el/"
start_time = "2025-03-16-19-00-00"
end_time = "2025-03-16-21-15-00"

output_format = "mp4"  # "mp4" or "gif"
frame_rate = 2  # fps for both mp4 and gif


def _write_gif_compat(clip, out_path, fps):
    """Call write_gif across MoviePy versions (some removed 'program')."""
    params = inspect.signature(clip.write_gif).parameters
    if "program" in params:
        clip.write_gif(str(out_path), fps=fps, program="ffmpeg")
    else:
        clip.write_gif(str(out_path), fps=fps)


# Collect and normalize frames
images = []
temp_paths = []
base_w = base_h = None

for filename in sorted(os.listdir(image_folder)):
    # Print the progress
    print(f"Processing file number {len(images)+1}: of {len(os.listdir(image_folder))}", end="\r")
    if not filename.endswith(".png"):
        continue
    img_path = os.path.join(image_folder, filename)
    with Image.open(img_path) as img:
        if base_w is None or base_h is None:
            base_w, base_h = img.size
        resized = img.resize((base_w, base_h))
        temp_path = f"/tmp/resized_{filename}"
        resized.save(temp_path)
        images.append(temp_path)
        temp_paths.append(temp_path)

if not images:
    raise RuntimeError("No PNG frames found.")

clip = ImageSequenceClip(images, fps=frame_rate)

# Output
output_folder = Path("../movies/").expanduser().resolve()
output_folder.mkdir(parents=True, exist_ok=True)
stem = f"exposure_time_series_all_az_el_{start_time}_{end_time}_5min_res"

if output_format.lower() == "mp4":
    out_path = output_folder / f"{stem}.mp4"
    clip.write_videofile(str(out_path), fps=frame_rate, codec="libx264", audio=False)
elif output_format.lower() == "gif":
    out_path = output_folder / f"{stem}_{frame_rate}fps.gif"
    _write_gif_compat(clip, out_path, frame_rate)
else:
    raise ValueError("output_format must be 'mp4' or 'gif'")

# Cleanup
for p in temp_paths:
    try:
        os.remove(p)
    except OSError:
        pass

print(f"Saved: {out_path}")
