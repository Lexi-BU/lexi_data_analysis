import re
import shutil
from pathlib import Path

date_of_folder = "2024-06-01"
input_folder = Path(f"/mnt/cephadrius/bu_research/lexi_data/L1c/sci/cdf/{date_of_folder}")
output_folder = Path(f"/mnt/cephadrius/bu_research/lexi_data/L1c/sci/cdf_renamed/{date_of_folder}")

# Make sure output folder exists
output_folder.mkdir(parents=True, exist_ok=True)

# Regex pattern to match the old filenames
pattern = re.compile(
    r"payload_lexi_(\d{4}-\d{2}-\d{2})_(\d{2})-\d{2}-\d{2}_to_.*?_sci_output_L1c_v(\d+)\.(\d+)\.cdf$"
)

# Iterate through all CDF files
for file in input_folder.glob("*.cdf"):
    match = pattern.match(file.name)
    if match:
        date_str = match.group(1).replace("-", "")  # YYYYMMDD
        hour_str = match.group(2)  # HH
        version = f"V{match.group(3)}.{match.group(4)}"

        new_filename = f"lexi_l1c_{date_str}{hour_str}_{version}.cdf"
        new_path = output_folder / new_filename

        shutil.copy(file, new_path)
        print(f"Renamed: {file.name} → {new_filename}")
    else:
        print(f"Skipped (no match): {file.name}")
