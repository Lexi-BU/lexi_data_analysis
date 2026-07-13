import re
from datetime import datetime, timedelta
from ftplib import FTP_TLS
from pathlib import Path

from dateutil import parser


def floor_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def ceil_to_hour(dt):
    return (dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)) - timedelta(
        seconds=1
    )


def download_lexi_data(
    start_time, end_time, data_level="L1b", local_dir=Path("/home/cephandrius/Downloads/")
):
    """
    Downloads only the latest version of LEXI data files from FTPS server between given UTC start and end times,
    rounded to the full hour range. Stores each day's files in a corresponding subfolder.

    Args:
        start_time (str): ISO 8601 string, e.g., "2025-03-03T21:15:00Z"
        end_time (str): ISO 8601 string, e.g., "2025-03-03T22:45:00Z"
        data_level (str): One of "L1a", "L1b", "L1c"
        local_dir (Path or str): Root directory where files will be stored
    """
    ftp_host = "sptl-data.bu.edu"
    start_dt_raw = parser.isoparse(start_time)
    end_dt_raw = parser.isoparse(end_time)
    start_dt = floor_to_hour(start_dt_raw)
    end_dt = ceil_to_hour(end_dt_raw)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    ftps = FTP_TLS()
    ftps.connect(ftp_host, 21)
    ftps.login()
    ftps.prot_p()

    current_day = start_dt.date()
    while current_day <= end_dt.date():
        remote_dir = f"/sptl-data/lexi_data/{data_level}/sci/cdf/{current_day.isoformat()}"
        day_local_dir = local_dir / current_day.isoformat()
        day_local_dir.mkdir(parents=True, exist_ok=True)

        try:
            ftps.cwd(remote_dir)
            filenames = ftps.nlst()
        except Exception as e:
            print(f"Skipping {remote_dir} due to error: {e}")
            current_day += timedelta(days=1)
            continue

        # Group by start time and select latest version
        latest_files = {}
        for fname in filenames:
            match = re.search(
                r"payload_lexi_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_to_.*?_sci_output_.*?_v(\d+)\.(\d+)\.cdf",
                fname,
            )
            if not match:
                continue
            time_str, major, minor = match.groups()
            file_start = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S").replace(
                tzinfo=start_dt_raw.tzinfo
            )
            version = (int(major), int(minor))

            if not (start_dt <= file_start <= end_dt):
                continue

            if time_str not in latest_files or version > latest_files[time_str][1]:
                latest_files[time_str] = (fname, version)

        # Download only the latest version for each hour
        for time_str, (fname, _) in latest_files.items():
            local_path = day_local_dir / fname
            if local_path.exists():
                print(f"Skipping {fname} (already exists)")
                continue
            print(f"Downloading {fname} → {local_path}")
            try:
                with local_path.open("wb") as f:
                    ftps.retrbinary(f"RETR {fname}", f.write)
            except Exception as e:
                print(f"Error downloading {fname}: {e}")

        current_day += timedelta(days=1)

    ftps.quit()


# Example usage
# You do not need to modify any code above this line.
# In the example code below, you can change the start_time, end_time, data_level, and local_dir
# parameters to suit your needs.
# Start time valid values are from 2025-01-16T00:00:00Z to 2025-03-16T22:00:00Z.
# End time valid values are from 2024-10-16T00:00:00Z to 2025-03-16T22:00:00Z.
# Valid data_level values are "L1a", "L1b", "L1c".
if __name__ == "__main__":
    download_lexi_data(
        start_time="2025-03-16T18:00:00Z",
        end_time="2025-03-16T22:00:00Z",
        data_level="L1b",
        local_dir=Path("./Data/L1b_data"),
    )
