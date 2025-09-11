import datetime
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from spacepy import pycdf
from spacepy.pycdf import CDF

# Input and output
input_root = Path("/mnt/cephadrius/bu_research/lexi_data/L1c/sci/cdf_renamed/")
output_file = Path(
    "/mnt/cephadrius/bu_research/lexi_data/L1c/sci/cdf/flat_field_data/lexi_l1c_flat_field_data_20240524_20240530.cdf"
)

# Step 1: Recursively find all .cdf files
cdf_files = sorted(input_root.rglob("*.cdf"))

if not cdf_files:
    raise FileNotFoundError("No CDF files found.")

# Step 2: Read and concatenate all variables into a dictionary
all_data = {}

for i, file in enumerate(cdf_files[:]):
    # Print the progress
    print(f"Reading file number {i + 1} of {len(cdf_files)}", end="\r")
    try:
        with CDF(str(file)) as f:  # Convert Path to string
            # print(f"{file.name}: {list(f.keys())}")  # Optional: see available variables
            for var in f:
                if var not in all_data:
                    all_data[var] = []
                all_data[var].append(f[var][...])
    except Exception as e:
        print(f"Error reading {file}: {e}")


# Step 3: Concatenate and build final DataFrame
for key in all_data:
    all_data[key] = np.concatenate(all_data[key])

df = pd.DataFrame(all_data)
# df["Epoch"] = pd.to_datetime(df["Epoch"], unit="s", utc=True)
df.sort_values("Epoch", inplace=True)
df = df.drop_duplicates(subset="Epoch")
# Set index to Epoch
df.set_index("Epoch", inplace=True)
df.index = pd.to_datetime(df.index, unit="s", utc=True)
# Convert the index to a timezone-aware datetime
df.index = df.index.tz_convert("UTC")
"""
# Step 4: Write to new CDF file
# Option 1: Copy a template file and overwrite contents
template_cdf = cdf_files[0]
shutil.copy(template_cdf, output_file)

# Convert datetime to CDF_EPOCH if needed
# if "Epoch" in df.columns and is_datetime64_any_dtype(df["Epoch"]):
#     df["Epoch"] = sptime.Ticktock(df["Epoch"].tolist(), "UTC").cdf_epoch()

# Delete output file if it exists
if output_file.exists():
    output_file.unlink()

# Create and write to CDF
with pycdf.CDF(str(output_file), create=True) as out:
    for key in df.columns:
        try:
            out.new(key, df[key].values)
        except Exception as e:
            print(f"❌ Error writing {key}: {e}")


print(f"✅ Combined CDF saved to: {output_file}")

"""

skeleton_path = Path(
    "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_pipeline/spdf_data_documents/l1c/lexi_l1c_0000000000_v0.1.cdf"
)

# Load the skeleton in read-only mode
skeleton_cdf = CDF(str(skeleton_path))

# Create new writable CDF file (overwrite if exists)
if output_file.exists():
    output_file.unlink()
cdf_data = CDF(str(output_file), "")

# Copy global attributes from skeleton
for key in skeleton_cdf.attrs:
    cdf_data.attrs[key] = skeleton_cdf.attrs[key][...]

# Update dynamic global attributes
cdf_data.attrs.update(
    {
        "Generation_date": str(datetime.datetime.now(datetime.timezone.utc)),
        "Logical_file_id": output_file.stem,
        "source": output_file.name,
    }
)
# ========== Variables ==========
cdf_data["Epoch"] = df.index

# Convert index to signed 32-bit integers (seconds since Unix epoch)
epoch_unix_vals = (df.index.astype(int) // 10**9).astype(np.int32)

# Explicitly create variable as CDF_INT4 (code 32)
cdf_data.new("Epoch_unix", data=epoch_unix_vals)
# Set internal fill value
# cdf_data["Epoch_unix"].pad = np.int32(-2147483648)
cdf_data["Epoch_unix"].attrs.update(
    {
        "FIELDNAM": "Time in Unix Epoch",
        "VALIDMIN": np.int32(epoch_unix_vals.min()),
        "VALIDMAX": np.int32(epoch_unix_vals.max()),
        "SCALEMIN": np.int32(epoch_unix_vals.min()),
        "SCALEMAX": np.int32(epoch_unix_vals.max()),
        "LABLAXIS": "Epoch Unix",
        "UNITS": "s",
        "MONOTON": "INCREASE",
        "VAR_TYPE": "support_data",
        "FORMAT": "I10",
        "FILLVAL": np.int32(-2147483648),  # standard ISTP fill value for INT4
        "DEPEND_0": "Epoch",
        "DICT_KEY": "time>Epoch_unix",
        "CATDESC": "Time, centered, in Unix Epoch seconds",
        "AVG_TYPE": " ",
        "DISPLAY_TYPE": " ",
        "VAR_NOTES": " ",
    }
)
photon_vars = [
    "photon_x_mcp",
    "photon_y_mcp",
    "photon_RA",
    "photon_Dec",
    "photon_az",
    "photon_el",
]

for var in photon_vars:
    if var in df.columns:
        # Let data type match the skeleton — no need to force type
        cdf_data[var] = df[var].values

for varname in skeleton_cdf:
    if varname in cdf_data:
        for attr in skeleton_cdf[varname].attrs:
            try:
                cdf_data[varname].attrs[attr] = skeleton_cdf[varname].attrs[attr][...]
            except Exception:
                cdf_data[varname].attrs[attr] = skeleton_cdf[varname].attrs[attr]

skeleton_cdf.close()
cdf_data.close()
