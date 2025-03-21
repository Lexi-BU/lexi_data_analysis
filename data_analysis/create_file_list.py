# Create a list of all file names in the folder:
# /home/vetinari/Desktop/git/Lexi-Bu/lexi_data_pipeline/data/from_lexi/2024/processed_data/sci/level_1c/cdf/1.0.0/

import glob
import numpy as np

folder_name = "/home/vetinari/Desktop/git/Lexi-Bu/lexi_data_pipeline/data/from_lexi/2025/processed_data/sci/level_1c/cdf/1.0.0/"

file_name_list = np.sort(glob.glob(folder_name + "/*.cdf"))

# Remove the path from the file names
file_name_list = [file_name.split("/")[-1] for file_name in file_name_list]

# Extract the dates in the file names (The files are named as:
# lexi_payload_1740903003_32562_level_1c_1.0.0.cdf where 1740903003 is the date)
date_list = [
    int(file_name.split("/")[-1].split("_")[2]) for file_name in file_name_list
]

# Save the date list and associated file names in a csv file
np.savetxt(
    "file_list.csv", np.array([date_list, file_name_list]).T, delimiter=",", fmt="%s"
)
