from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyspedas
from pytplot import get_data

time_range = ["2025-03-16 19:00:00", "2025-03-16 22:00:00"]
sc = "c"  # Denotes which THEMIS probe to load data from

# Electrostatic Analyzer Reduced mode - 3sec resolution, low angular resolution
esa_redu = pyspedas.themis.esa(
    probe=sc,
    level="l2",
    trange=time_range,
    varnames=[
        "th" + sc + "_peer_density",
        "th" + sc + "_peef_density",
        "th" + sc + "_peeb_density",
        "th" + sc + "_peer_avgtemp",
        "th" + sc + "_peef_avgtemp",
        "th" + sc + "_peeb_avgtemp",
        "th" + sc + "_peer_velocity_gse",
        "th" + sc + "_peef_velocity_gse",
        "th" + sc + "_peeb_velocity_gse",
    ],
    no_update=False,
)

data_dict = {}
vars_to_extract = [
    "th" + sc + "_peer_density",
    "th" + sc + "_peef_density",
    "th" + sc + "_peeb_density",
    "th" + sc + "_peer_avgtemp",
    "th" + sc + "_peef_avgtemp",
    "th" + sc + "_peeb_avgtemp",
    "th" + sc + "_peer_velocity_gse",
    "th" + sc + "_peef_velocity_gse",
    "th" + sc + "_peeb_velocity_gse",
]

# Get the spacecraft position for context (GSE coordinates)
pos = pyspedas.themis.ssc(
    probe=sc,
    trange=time_range,
    level="l2",
    varnames=["XYZ_GSE"],
    no_update=False,
)
scalar_vars = vars_to_extract[:6]  # density and temperature
vector_vars = vars_to_extract[6:]  # velocity

# Temporary list to collect individual Series and then concatenate
series_list = []

# Extract scalar variables
for var in scalar_vars:
    data = get_data(var)
    if data is not None:
        times = pd.to_datetime(data.times, unit="s")
        series = pd.Series(data.y, index=times, name=var)
        series = series[~series.index.duplicated(keep="first")]
        series_list.append(series)

# Process vector variables
for var in vector_vars:
    data = get_data(var)
    if data is not None:
        times = pd.to_datetime(data.times, unit="s")
        times_unique, indices = pd.Series(times).drop_duplicates(keep="first").index, ~pd.Series(
            times
        ).duplicated(keep="first")
        for i, comp in enumerate(["x", "y", "z"]):
            col_name = f"{var}_{comp}"
            series = pd.Series(data.y[indices, i], index=times[indices], name=col_name)
            series_list.append(series)

# Combine all series into a single DataFrame
df = pd.concat(series_list, axis=1)
df.sort_index(inplace=True)


# Get the spacecraft position
spc_data = get_data("XYZ_GSE")
spc_series_list = []
if spc_data is not None:
    times = pd.to_datetime(spc_data.times, unit="s")
    times_unique, indices = pd.Series(times).drop_duplicates(keep="first").index, ~pd.Series(
        times
    ).duplicated(keep="first")
    for i, comp in enumerate(["x", "y", "z"]):
        col_name = f"th{sc}_pos_gse_{comp}"
        series = pd.Series(spc_data.y[indices, i], index=times[indices], name=col_name)
        spc_series_list.append(series)
    df_pos = pd.concat(spc_series_list, axis=1)
    df_pos.sort_index(inplace=True)
    # Merge position data into main dataframe using mergeasof
    df = pd.merge_asof(
        df,
        df_pos,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("1s"),
    )
# Get the magnitude of the velocities
df["th" + sc + "_peer_velocity_gse_mag"] = (
    df[
        [
            "th" + sc + "_peer_velocity_gse_x",
            "th" + sc + "_peer_velocity_gse_y",
            "th" + sc + "_peer_velocity_gse_z",
        ]
    ]
    .pow(2)
    .sum(axis=1)
    .pow(0.5)
)
df["th" + sc + "_peef_velocity_gse_mag"] = (
    df[
        [
            "th" + sc + "_peef_velocity_gse_x",
            "th" + sc + "_peef_velocity_gse_y",
            "th" + sc + "_peef_velocity_gse_z",
        ]
    ]
    .pow(2)
    .sum(axis=1)
    .pow(0.5)
)
df["th" + sc + "_peeb_velocity_gse_mag"] = (
    df[
        [
            "th" + sc + "_peeb_velocity_gse_x",
            "th" + sc + "_peeb_velocity_gse_y",
            "th" + sc + "_peeb_velocity_gse_z",
        ]
    ]
    .pow(2)
    .sum(axis=1)
    .pow(0.5)
)

# Get the flux (density * velocity magnitude)
df["th" + sc + "_peer_flux"] = (
    df["th" + sc + "_peer_density"] * df["th" + sc + "_peer_velocity_gse_mag"]
)
df["th" + sc + "_peef_flux"] = (
    df["th" + sc + "_peef_density"] * df["th" + sc + "_peef_velocity_gse_mag"]
)
df["th" + sc + "_peeb_flux"] = (
    df["th" + sc + "_peeb_density"] * df["th" + sc + "_peeb_velocity_gse_mag"]
)

df = df.sort_index()

# Select only the desired time range
df = df[(df.index >= pd.to_datetime(time_range[0])) & (df.index <= pd.to_datetime(time_range[1]))]

# Build 1-minute target index (preserve tz if present)
tz = df.index.tz
t0 = df.index.min().ceil("min")
t1 = df.index.max().floor("min")
target_idx = pd.date_range(t0, t1, freq="1min", tz=tz)

# Nearest fill within 300s, then time interpolation for gaps up to 10 minutes
df = df.reindex(target_idx, method="nearest", tolerance=pd.Timedelta("300s"))
df = df.interpolate(method="time", limit=10)  # up to 10 consecutive 1-min gaps


save_folder = "../data/themis_l2_electron_params/"
Path(save_folder).mkdir(parents=True, exist_ok=True)
df.to_csv(
    save_folder
    + f"themis_{sc}_l2_electron_params_{time_range[0].replace(' ', 'T')}_to_{time_range[1].replace(' ', 'T')}.csv"
)


# Plot all three xyz components of the positions with time on the x-axis
fig, axs = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
axs.plot(df.index, df[f"th{sc}_pos_gse_x"], label="X")
axs.plot(df.index, df[f"th{sc}_pos_gse_y"], label="Y")
axs.plot(df.index, df[f"th{sc}_pos_gse_z"], label="Z")
axs.set_ylabel("Position (GSE)")
axs.legend()
axs.set_title(f"THEMIS-{sc.upper()} Spacecraft Position (GSE) {time_range[0]} to {time_range[1]}")
axs.grid()
plt.tight_layout()
plt.show()