from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

lexi_ephemera_file = Path(
    "/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_pipeline/data/ephemeris_data/LEXIAngleData_ACTUAL_20250723_10min_linear.csv"
)
lexi_ephemera_df = pd.read_csv(lexi_ephemera_file)

# Set the Epoch column as datetime index
lexi_ephemera_df["Epoch"] = pd.to_datetime(lexi_ephemera_df["Epoch"], utc=True)
lexi_ephemera_df.set_index("Epoch", inplace=True)
lexi_ephemera_df.sort_index(inplace=True)

# Resample the data to 1-minute intervals using time interpolation
lexi_ephemera_df = lexi_ephemera_df.resample("1min").interpolate(method="time")

themis_spc = "c"  # THEMIS spacecraft to use
themis_ephemera_file = Path(
    f"/home/cephadrius/Desktop/git/Lexi-BU/lexi_data_analysis/data/themis_l2_electron_params/themis_{themis_spc}_l2_electron_params_2025-03-16T19:00:00_to_2025-03-16T22:00:00.csv"
)
themis_ephemera_df = pd.read_csv(themis_ephemera_file)

# Name the index column as "Epoch" and convert to datetime
themis_ephemera_df.rename(columns={"Unnamed: 0": "Epoch"}, inplace=True)
themis_ephemera_df["Epoch"] = pd.to_datetime(themis_ephemera_df["Epoch"], utc=True)
themis_ephemera_df.set_index("Epoch", inplace=True)
themis_ephemera_df.sort_index(inplace=True)

start_time = pd.to_datetime("2025-03-16 19:00:00", utc=True)
end_time = pd.to_datetime("2025-03-16 21:15:00", utc=True)

# Select the desired time range
lexi_ephemera_df = lexi_ephemera_df[
    (lexi_ephemera_df.index >= start_time) & (lexi_ephemera_df.index <= end_time)
]
themis_ephemera_df = themis_ephemera_df[
    (themis_ephemera_df.index >= start_time) & (themis_ephemera_df.index <= end_time)
]
# Merge the two dataframes on the datetime index
merged_df = pd.merge_asof(
    lexi_ephemera_df.sort_index(),
    themis_ephemera_df.sort_index(),
    left_index=True,
    right_index=True,
    direction="nearest",
    tolerance=pd.Timedelta("1min"),
)
# For any velocity components that are 0, set to NaN
velocity_components = [
    "th" + themis_spc + "_peef_velocity_gse_x",
    "th" + themis_spc + "_peef_velocity_gse_y",
    "th" + themis_spc + "_peef_velocity_gse_z",
    "th" + themis_spc + "_peeb_velocity_gse_x",
    "th" + themis_spc + "_peeb_velocity_gse_y",
    "th" + themis_spc + "_peeb_velocity_gse_z",
    "th" + themis_spc + "_peef_velocity_gse_mag",
    "th" + themis_spc + "_peeb_velocity_gse_mag",
]
for comp in velocity_components:
    merged_df.loc[merged_df[comp] == 0, comp] = np.nan
# Fill all the NaN values using time interpolation
merged_df = merged_df.interpolate(method="time")
# Earth radius in km
earth_radius_km = 6371.0
# Find the distance between Lexi and THEMIS
distance = np.sqrt(
    (merged_df["lexi_sc_pos_gse_x"] - merged_df["th" + themis_spc + "_pos_gse_x"] * earth_radius_km)
    ** 2
    + (
        merged_df["lexi_sc_pos_gse_y"]
        - merged_df["th" + themis_spc + "_pos_gse_y"] * earth_radius_km
    )
    ** 2
    + (
        merged_df["lexi_sc_pos_gse_z"]
        - merged_df["th" + themis_spc + "_pos_gse_z"] * earth_radius_km
    )
    ** 2
)
merged_df["distance_lexi_themis_km"] = distance


# Compute the time taken for solar wind to travel from THEMIS to Lexi
# Solar wind speed is: th" + themis_spc + "_peef_velocity_gse_mag
merged_df["time_delay_seconds"] = (
    merged_df["distance_lexi_themis_km"] / merged_df["th" + themis_spc + "_peef_velocity_gse_mag"]
)

save_folder = Path("../data/lexi_themis_analysis/")
save_folder.mkdir(parents=True, exist_ok=True)
output_file = save_folder / f"lexi_themis_{themis_spc}_analysis_lexi_spacecraft.csv"
merged_df.to_csv(output_file)
