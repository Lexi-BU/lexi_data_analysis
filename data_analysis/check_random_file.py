import glob
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from spacepy.pycdf import CDF as cdf

data_folder_location = "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/"

# Get the list of files in the directory
file_list = glob.glob(data_folder_location + "/**/*.cdf", recursive=True)

# pick a random number between 0 and the length of the file list
random_number = np.random.randint(0, len(file_list))
# pick a random file from the list
random_file = file_list[random_number]

dat = cdf(random_file)

# Print just the name of the file
print(random_file.split("/")[-1])
# print the keys in the file
print(dat["Epoch"][0])
print(dat["Epoch"][-1])
