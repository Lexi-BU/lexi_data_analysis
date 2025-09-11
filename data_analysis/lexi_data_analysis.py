import importlib

import lexi_data_analysis_functions_istp as ldaf

importlib.reload(ldaf)

# Read all data files in a given time range
# NOTE: Make sure to set the correct time range and data folder location.
# df = ldaf.read_all_data_files(
#     file_list=None,
#     start_time="2025-03-16 18:00",
#     end_time="2025-03-16 19:05",
#     return_data_type="dataframe",
#     kwargs={
#         "data_folder_location": "data",
#         "version": "latest",
#         "start_time": "2025-03-16 18:00",
#         "end_time": "2025-03-16 19:05",
#     },
# )

# Plot time series for a specific time range
# ldaf.plot_time_series(
#     # df=df,
#     start_time="2025-03-16 21:01",
#     end_time="2025-03-16 21:05",
#     # keys=["key1", "key2"],
#     x_axis="Epoch",
#     output_path="figures",
#     data_folder_location="data",
# )

# Plot histogram for a specific time range
results = ldaf.plot_histogram(
    # df=df,
    start_time="2025-03-16 21:01",
    end_time="2025-03-16 21:05",
    x_key="photon_az",
    y_key="photon_el",
    time_normalization=True,
    norm_scale="log",
    flat_field_correction=True,
    cmap="plasma",
    data_folder_location="data",
    plot_flat_field=True,
)

H_orig = results["H_orig"]
H_flat = results["H_flat"]
H_result = results["H_result"]
xedges = results["xedges"]
yedges = results["yedges"]
