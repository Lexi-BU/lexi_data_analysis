# Get THEMIS plasma parameters
import matplotlib.pyplot as plt
import pandas as pd
import pyspedas
from pytplot import get_data

time_range = ["2025-03-16", "2025-03-17"]
sc = "c"  # Denotes which THEMIS probe to load data from


# Electrostatic Analyzer Reduced mode - 3sec resolution, low angular resolution
esa_redu = pyspedas.themis.esa(
    probe=sc,
    level="l2",
    trange=time_range,
    varnames=[
        "th" + sc + "_peir_density",
        "th" + sc + "_peif_density",
        "th" + sc + "_peib_density",
        "th" + sc + "_peir_avgtemp",
        "th" + sc + "_peif_avgtemp",
        "th" + sc + "_peib_avgtemp",
        "th" + sc + "_peir_velocity_gse",
        "th" + sc + "_peif_velocity_gse",
        "th" + sc + "_peib_velocity_gse",
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
    "th" + sc + "_peir_density",
    "th" + sc + "_peif_density",
    "th" + sc + "_peib_density",
    "th" + sc + "_peer_density",
    "th" + sc + "_peef_density",
    "th" + sc + "_peeb_density",
    "th" + sc + "_peir_avgtemp",
    "th" + sc + "_peif_avgtemp",
    "th" + sc + "_peib_avgtemp",
    "th" + sc + "_peer_avgtemp",
    "th" + sc + "_peef_avgtemp",
    "th" + sc + "_peeb_avgtemp",
    "th" + sc + "_peir_velocity_gse",
    "th" + sc + "_peif_velocity_gse",
    "th" + sc + "_peib_velocity_gse",
    "th" + sc + "_peer_velocity_gse",
    "th" + sc + "_peef_velocity_gse",
    "th" + sc + "_peeb_velocity_gse",
]

# Separate scalar and vector variables
scalar_vars = vars_to_extract[:12]  # densities and avg temperatures
vector_vars = vars_to_extract[12:]  # velocities

# Temporary list to collect individual Series and then concatenate
series_list = []

# Process scalar variables
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

# Combine into single DataFrame
df = pd.concat(series_list, axis=1)
df.sort_index(inplace=True)

# Get the magnitude of the velocity vectors (peir, peif, peib)
df["th" + sc + "_peir_velocity_magnitude"] = (
    df["th" + sc + "_peir_velocity_gse_x"] ** 2
    + df["th" + sc + "_peir_velocity_gse_y"] ** 2
    + df["th" + sc + "_peir_velocity_gse_z"] ** 2
) ** 0.5
df["th" + sc + "_peif_velocity_magnitude"] = (
    df["th" + sc + "_peif_velocity_gse_x"] ** 2
    + df["th" + sc + "_peif_velocity_gse_y"] ** 2
    + df["th" + sc + "_peif_velocity_gse_z"] ** 2
) ** 0.5
df["th" + sc + "_peib_velocity_magnitude"] = (
    df["th" + sc + "_peib_velocity_gse_x"] ** 2
    + df["th" + sc + "_peib_velocity_gse_y"] ** 2
    + df["th" + sc + "_peib_velocity_gse_z"] ** 2
) ** 0.5

# Get the solar wind flux for the three species
df["th" + sc + "_peir_flux"] = (
    df["th" + sc + "_peir_density"] * df["th" + sc + "_peir_velocity_magnitude"]
)
df["th" + sc + "_peif_flux"] = (
    df["th" + sc + "_peif_density"] * df["th" + sc + "_peif_velocity_magnitude"]
)
df["th" + sc + "_peib_flux"] = (
    df["th" + sc + "_peib_density"] * df["th" + sc + "_peib_velocity_magnitude"]
)
# Save to CSV
csv_filename = f"themis_{sc}_esa_parameters_{time_range[0]}_to_{time_range[1]}_flux.csv"
df.to_csv(csv_filename)
