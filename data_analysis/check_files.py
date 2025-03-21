import glob

folder = "/mnt/cephadrius/bu_research/lexi_data/L1b/sci/cdf/"
file_version_0 = "v0.0"
file_version_1 = "v0.1"
file_list_v_0 = sorted(glob.glob(str(folder) + f"/**/*{file_version_0}.cdf", recursive=True))
file_list_v_1 = sorted(glob.glob(str(folder) + f"/**/*{file_version_1}.cdf", recursive=True))


# Get the list of file names that are in v0 but not in v1
file_list_v_0_not_v_1 = []
for file in file_list_v_0:
    if file.replace(file_version_0, file_version_1) not in file_list_v_1:
        file_list_v_0_not_v_1.append(file)

# Get the list of file names that are in v1 but not in v0
file_list_v_1_not_v_0 = []
for file in file_list_v_1:
    if file.replace(file_version_1, file_version_0) not in file_list_v_0:
        file_list_v_1_not_v_0.append(file)

# Print the length of the lists
print(
    f"Number of files in {file_version_0} but not in {file_version_1}: {len(file_list_v_0_not_v_1)}"
)
print(
    f"Number of files in {file_version_1} but not in {file_version_0}: {len(file_list_v_1_not_v_0)}"
)
