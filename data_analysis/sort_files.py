import datetime
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Get the user login name
user_name = os.getlogin()

spcacecraft_data_folder = f"/home/{user_name}/Desktop/git/Lexi-BU/lxi_gui/data/from_LEXI/surface/"
spcacecraft_data_folder = Path(spcacecraft_data_folder).expanduser().resolve()
level_zero_folder = f"/home/{user_name}/Desktop/git/Lexi-BU/lexi_data_pipeline/data/level_0/"
level_zero_folder = Path(level_zero_folder).expanduser().resolve()
level_zero_folder.mkdir(parents=True, exist_ok=True)

# Get all files in the folder and subfolders
file_val_list = glob.glob(str(spcacecraft_data_folder) + "/**/*.dat", recursive=True)

for file_val in file_val_list[0:]:
    # Get the time from the file name
    file_name = os.path.basename(file_val)
    unix_time = int(file_name.split("_")[-2])
    dt = datetime.datetime.fromtimestamp(unix_time, tz=datetime.timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")

    # Create the folder structure
    folder_path = level_zero_folder / date_str
    folder_path.mkdir(parents=True, exist_ok=True)

    # Copy the file to the new folder
    # If the file already exists, skip it
    file_name = os.path.basename(file_val)
    if (folder_path / file_name).exists():
        # print(f"File {file_name} already exists in {folder_path}. Skipping.")
        continue
    else:
        # Copy the file to the new folder
        shutil.copy(file_val, folder_path / file_name)
        print(f"Copied {file_name} to {folder_path}")
