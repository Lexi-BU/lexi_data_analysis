import subprocess
from pathlib import Path
from typing import Optional


def mp4_to_gif_ffmpeg(
    mp4_path: str,
    gif_path: str,
    start_time: Optional[float] = None,  # seconds
    end_time: Optional[float] = None,  # seconds
    fps: int = 10,
    width: int = 480,
):
    """
    Convert MP4 to GIF using ffmpeg with palette generation for quality/size.
    - Places -ss BEFORE -i (fast seek) and uses -t AFTER -i.
    - Uses a split/palettegen/paletteuse chain to optimize colors.
    """

    mp4_path = str(Path(mp4_path))
    gif_path = str(Path(gif_path))

    # Compute duration if end_time is provided
    duration = None
    if end_time is not None:
        duration = end_time - (start_time or 0)
        if duration <= 0:
            raise ValueError(
                "end_time must be greater than start_time (or > 0 if start_time is None)."
            )

    # Build filter graph: fps -> scale -> split -> palettegen/paletteuse
    filter_complex = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen=stats_mode=diff[p];"
        f"[s1][p]paletteuse=new=1:dither=bayer:bayer_scale=5"
    )

    cmd = ["ffmpeg", "-y"]

    # Fast seek (before input)
    if start_time is not None:
        cmd += ["-ss", str(start_time)]

    # Input
    cmd += ["-i", mp4_path]

    # Duration (after input)
    if duration is not None:
        cmd += ["-t", str(duration)]

    # Filters and output
    cmd += ["-filter_complex", filter_complex, "-gifflags", "+transdiff", gif_path]

    # Run without shell to avoid quoting issues
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    mp4_to_gif_ffmpeg(
        "/home/vetinari/Desktop/git/Lexi-Bu/lexi_data_analysis/movies/exposure_time_series_all.mp4",
        "/home/vetinari/Desktop/git/Lexi-Bu/lexi_data_analysis/movies/exposure_time_series_all.gif",
        start_time=0,
        end_time=5,
        fps=5,
        width=1000,
    )
