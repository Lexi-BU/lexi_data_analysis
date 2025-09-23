import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter

mike_file = "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_pipeline/data/ephemeris_data/LEXIAngleData_ACTUAL_20250723.csv"
df_mike = pd.read_csv(mike_file)
# Rename "[Epoch (UTC)]" to "Epoch"
df_mike.rename(columns={"[Epoch (UTC)]": "Epoch"}, inplace=True)
# Convert Epoch from Mar 02 2025 09:34:00.00000000 to datetime format
df_mike["Epoch"] = pd.to_datetime(df_mike["Epoch"], format="%b %d %Y %H:%M:%S.%f")
# Set the index to be the Epoch column
df_mike.set_index("Epoch", inplace=True)
# Rename [SZA (deg)] to sza
df_mike.rename(columns={"[SZA (deg)]": "sza"}, inplace=True)
# Keep only the sza column
df_mike = df_mike[["sza"]]
# Resample to 10 minute intervals, starting from the first time in the file
# df_mike = df_mike.resample("1min").mean()
# # Interpolate to fill in any missing values
# df_mike.interpolate(method="time", inplace=True)

# Now read in the SZA file we created
our_file = "solar_zenith_angle.csv"
df_our = pd.read_csv(our_file)

# Rename Time to Epoch
df_our.rename(columns={"Time": "Epoch"}, inplace=True)
# Convert Epoch to datetime format
df_our["Epoch"] = pd.to_datetime(df_our["Epoch"], format="%Y-%m-%d %H:%M:%S")
# Set the index to be the Epoch column
df_our.set_index("Epoch", inplace=True)
# Rename SZA to sza
df_our.rename(columns={"SZA": "sza"}, inplace=True)

freq = pd.infer_freq(df_mike.index) or "H"
origin = df_mike.index.min()
df_our_hourly = df_our.resample(freq, origin=origin, label="left", closed="left").agg("mean")

# Align exactly to df1 timestamps and merge
df_our_hourly = df_our_hourly.reindex(df_mike.index)
df_combined = df_mike.join(df_our_hourly, how="left", rsuffix="_our")
# Rename the columns to sza_mike and sza_our
df_combined.rename(columns={"sza": "sza_mike", "sza_our": "sza_our"}, inplace=True)

# Calculate the difference between the two sza columns
df_combined["sza_diff"] = df_combined["sza_mike"] - df_combined["sza_our"]

# Plot the two sza columns and their difference
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(df_combined.index, df_combined["sza_mike"], label="Mike's SZA", color="blue")
ax1.plot(df_combined.index, df_combined["sza_our"], label="Our SZA", color="orange", linestyle="--")
ax1.set_ylabel("Solar Zenith Angle (degrees)")
ax1.set_title("Comparison of Solar Zenith Angle Calculations")
ax1.legend()
ax1.grid()
date_form = DateFormatter("%Y-%m-%d %H:%M")
ax1.xaxis.set_major_formatter(date_form)
ax2.plot(df_combined.index, df_combined["sza_diff"], label="Difference (Mike - Our)", color="green")
ax2.set_ylabel("Difference in SZA (degrees)")
ax2.set_xlabel("Time")
ax2.set_title("Difference Between Mike's and Our SZA Calculations")
ax2.legend()
ax2.grid()

# Set the x-axis limits from March 2, 2025 to March 16, 2025, 22:00
ax2.set_xlim(datetime.datetime(2025, 3, 2, 0, 0, 0), datetime.datetime(2025, 3, 16, 22, 0, 0))
# Set the y-axis limits to -1 to 1
ax2.set_ylim(df_combined["sza_diff"].min() * 1.1, df_combined["sza_diff"].max() * 1.1)
plt.tight_layout()
# plt.savefig("figures/solar_zenith_angle_comparison.png")
plt.show()
# plt.close()
