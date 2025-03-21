from spacepy.pycdf import CDF
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

folder_name = "../data/from_lexi/2025/processed_data/sci/level_1c/cdf/1.0.0/"

folder_name = Path(folder_name).expanduser().resolve()

file_name_list = np.sort(glob.glob(str(folder_name) + "/*.cdf"))

# Read the first file
dat = CDF(file_name_list[9])

# Save the data to a dataframe
df = pd.DataFrame()
for key in dat.keys():
    df[key] = dat[key][:]

print(df.head())
# Plot the data
# fig, ax = plt.subplots(2, 1, figsize=(8, 8))
# ax[0].scatter(df.index, df["ra_J2000_deg"], label="RA", s=0.1, c="r", marker=".", alpha=0.5)
# ax[0].set_xlabel("Time")
# ax[0].set_ylabel("RA [deg]")
# ax[0].grid()
