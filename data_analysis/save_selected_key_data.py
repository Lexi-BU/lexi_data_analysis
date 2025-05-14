import datetime
import glob
import importlib
from pathlib import Path

import matplotlib.dates as mdates

# import lexi_data_analysis_functions as lexi_functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spacepy.pycdf import CDF as cdf

"""
merged_file_name = (
    "../data/merged_lexi_hk_look_direction_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00.pkl"
)
# Save it to a csv file
# merged_file_name_csv = (
#     "../data/merged_lexi_hk_look_direction_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00.csv"
# )
# merged_df = pd.read_pickle(merged_file_name)
# merged_df.to_csv(merged_file_name_csv)
# Read the merged data from the pickle file
merged_df = pd.read_pickle(merged_file_name)


selected_keys = [
    "DeltaEvntCount",
    "DeltaDroppedCount",
    "DeltaLostEvntCount",
    "all_counts",
]
# Select the keys from the merged dataframe
selected_df = merged_df[selected_keys]

# Convert the index to a datetime object
selected_df.index = pd.to_datetime(selected_df.index)
selected_df["Epoch"] = selected_df.index

min_time = "2025-03-02T00:00:00Z"
max_time = "2025-03-16T23:59:59Z"

selected_df = selected_df[(selected_df["Epoch"] >= min_time) & (selected_df["Epoch"] <= max_time)]
# Save the selected keys to a new csv file
selected_file_name = (
    "../data/selected_lexi_hk_data_2025-03-02_00-00-00_to_2025-03-16_23-59-59.csv"
)
selected_df.to_csv(selected_file_name, index=False)
"""


look_direction_file_name = "/home/cephadrius/Desktop/git/Lexi-BU/look_direction/data/20241114_LEXIAngleData_20250302Landing.csv"

# Read the look direction data from the csv file
df_look_direction = pd.read_csv(look_direction_file_name)

selected_keys = [
    "Epoch",
    "az_earth",
    "el_earth",
    "ra_earth",
    "dec_earth",
    "az_sun",
    "el_sun",
    "ra_sun",
    "dec_sun",
    "ra_mag",
    "dec_mag",
]

# Convert "epoch_utc" from Mar 02 2025 08:01:00.000000000 to 2025-03-02T08:01:00Z
df_look_direction["epoch_utc"] = pd.to_datetime(
    df_look_direction["epoch_utc"], format="%b %d %Y %H:%M:%S.%f"
)
# Convert to UTC
df_look_direction["epoch_utc"] = df_look_direction["epoch_utc"].dt.tz_localize("UTC")
# Convert to datetime object
df_look_direction["epoch_utc"] = pd.to_datetime(df_look_direction["epoch_utc"])
# Convert the index to a datetime object
df_look_direction["Epoch"] = df_look_direction["epoch_utc"]

# save the selected keys to a new csv file
selected_df = df_look_direction[selected_keys]

# rename "ra_mag" to "expected_lexi_ra" and "dec_mag" to "expected_lexi_dec"
selected_df = selected_df.rename(
    columns={
        "ra_mag": "expected_lexi_ra",
        "dec_mag": "expected_lexi_dec",
    }
)
selected_look_direction_file_name = "/home/cephadrius/Desktop/git/Lexi-BU/look_direction/data/selected_look_direction_data_2025-03-02_00-00-00_to_2025-03-16_23-59-59.csv"
selected_df.to_csv(selected_look_direction_file_name, index=False)
