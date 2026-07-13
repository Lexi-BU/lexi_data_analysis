import shutil
import subprocess
from pathlib import Path


def mp4_to_gif(
    input_path,
    output_path=None,
    max_width=None,
    max_height=None,
    colors=256,
    dither="sierra2_4a",
    loop=0,
    overwrite=True,
):
    """
    Convert an MP4 to GIF while preserving source FPS and aspect ratio.
    Requires ffmpeg and ffprobe on PATH.

    Parameters
    ----------
    input_path : str or Path
    output_path : str or Path, optional
        Defaults to input stem + ".gif" in the same directory.
    max_width : int, optional
        Optional downscale max width (aspect ratio preserved).
    max_height : int, optional
        Optional downscale max height (aspect ratio preserved).
    colors : int, default 256
        GIF palette size (2–256).
    dither : str, default "sierra2_4a"
        Dithering algorithm for paletteuse.
    loop : int, default 0
        0 = loop forever, >0 = loop count.
    overwrite : bool, default True
        Overwrite existing output.

    Returns
    -------
    Path
        Path to the generated GIF.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".gif")
    output_path = Path(output_path)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required on PATH.")

    # Probe source frame rate (as rational, e.g., 30000/1001)
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "0",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "csv=p=0",
        str(input_path),
    ]
    fps = subprocess.check_output(ffprobe_cmd, text=True).strip() or "0/0"
    if fps in ("0/0", "0", ""):
        fps = "25"  # fallback

    # Optional scaling while preserving aspect ratio
    scale_filter = ""
    if max_width and max_height:
        scale_filter = (
            f",scale=w={int(max_width)}:h={int(max_height)}:force_original_aspect_ratio=decrease"
        )
    elif max_width:
        scale_filter = f",scale=w={int(max_width)}:h=-1"
    elif max_height:
        scale_filter = f",scale=w=-1:h={int(max_height)}"

    # Two-pass palette generation for high-quality GIF
    # Split -> palettegen -> paletteuse, preserve fps
    filter_complex = (
        f"[0:v]fps={fps}{scale_filter},split[p0]time[p1];"
        f"[p0]palettegen=stats_mode=diff:nb_colors={int(colors)}[pal];"
        f"[p1][pal]paletteuse=dither={dither}"
    )

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_complex,
        "-an",
        "-loop",
        str(int(loop)),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


if __name__ == "__main__":
    mp4_to_gif(
        input_path="/home/cephandrius/Desktop/git/Lexi-BU/lexi_data_analysis/movies/exposure_time_series_all_2025-03-16-19-00-00_2025-03-16-21-15-00.mp4",
        output_path="/home/cephandrius/Desktop/git/Lexi-BU/lexi_data_analysis/movies/exposure_time_series_all_2025-03-16-19-00-00_2025-03-16-21-15-00.gif",
    )