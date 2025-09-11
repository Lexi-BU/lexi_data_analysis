from spacepy.pycdf import CDF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import pandas as pd

look_direction_file = (
    "../data/from_spacecraft/2025/20241114_LEXIAngleData_20250302Landing_rad.csv"
)

look_direction_df = pd.read_csv(look_direction_file)

# Convert mp_ra and mp_dec to degrees
look_direction_df["mp_ra"] = look_direction_df["mp_ra"] * 180 / np.pi
look_direction_df["mp_dec"] = look_direction_df["mp_dec"] * 180 / np.pi

# Set the epoch_utc column as index
look_direction_df["epoch_unix"] = pd.to_datetime(
    look_direction_df["epoch_utc"], utc=True, unit="s"
)
look_direction_df = look_direction_df.set_index("epoch_unix")

ra_min_look_direction = np.nanmax([0, np.nanmin(look_direction_df["mp_ra"])])
ra_max_look_direction = np.nanmin([360, np.nanmax(look_direction_df["mp_ra"])])
dec_min_look_direction = np.nanmax([-90, np.nanmin(look_direction_df["mp_dec"])])
dec_max_look_direction = np.nanmin([90, np.nanmax(look_direction_df["mp_dec"])])

t_min_datetime = pd.to_datetime("2025-03-05 08:04:38", utc=True)
t_max_datetime = pd.to_datetime("2025-03-08 23:47:38", utc=True)
t_min = np.nanmax([look_direction_df.index[0], t_min_datetime])
t_max = np.nanmin([look_direction_df.index[-1], t_max_datetime])

# Make a histogram plot of the look direction mp_ra and mp_dec
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=0.32, wspace=0.0)

ax[0].hist(look_direction_df["mp_ra"], bins=100, color="r", alpha=0.5)  # RA
ax[0].set_xlabel("RA [deg]")
ax[0].set_ylabel("Counts")
ax[0].grid()
ax[0].set_title(f"LEXI look direction RA for about {len(look_direction_df)} points")
ax[0].set_xlim(ra_min_look_direction, ra_max_look_direction)
ax[0].set_yscale("log")

ax[1].hist(look_direction_df["mp_dec"], bins=100, color="b", alpha=0.5)  # Dec
ax[1].set_xlabel("Dec [deg]")
ax[1].set_ylabel("Counts")
ax[1].grid()
ax[1].set_title(f"LEXI look direction Dec for about {len(look_direction_df)} points")
ax[1].set_xlim(dec_min_look_direction, dec_max_look_direction)
ax[1].set_yscale("log")
plt.savefig("../figures/ra_dec_hist_look_direction.png")

plt.close()

# Plot 2d histogram of ra and dec with number of counts as the color using plt.hist2d
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
hb = ax.hist2d(
    look_direction_df["mp_ra"],
    look_direction_df["mp_dec"],
    bins=[360, 100],
    cmap="inferno",
    norm=mpl.colors.LogNorm(),
)
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Dec [deg]")
ax.grid()

cb = fig.colorbar(hb[3], ax=ax)
cb.set_label("Counts")
ax.set_xlim(ra_min_look_direction, ra_max_look_direction)
ax.set_ylim(dec_min_look_direction, dec_max_look_direction)
plt.savefig("../figures/ra_dec_2d_hist_look_direction.png")


plt.close()

# Make a time series plot of RA, dec
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].scatter(
    look_direction_df.index,
    look_direction_df["mp_ra"],
    label="RA",
    s=0.1,
    c="r",
    marker=".",
    alpha=0.5,
)

ax[0].set_xlabel("Time")
ax[0].set_ylabel("RA [deg]")
ax[0].set_ylim(ra_min_look_direction, ra_max_look_direction)
ax[0].grid()
ax[0].set_xlim(t_min, t_max)

# Rotate the x-axis labels by 45 degrees
ax[0].tick_params(axis="x", rotation=45)

ax[1].scatter(
    look_direction_df.index,
    look_direction_df["mp_dec"],
    label="dec",
    s=0.1,
    c="b",
    marker=".",
    alpha=0.5,
)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Dec [deg]")
ax[1].set_ylim(dec_min_look_direction, dec_max_look_direction)
ax[1].grid()
ax[1].set_xlim(t_min, t_max)

# Rotate the x-axis labels by 45 degrees
ax[1].tick_params(axis="x", rotation=45)

plt.savefig("../figures/ra_dec_time_series_look_direction.png")


folder_name = "/home/cephadrius/Desktop/git/Lexi-Bu/lexi_data_pipeline/data/from_lexi/2025/processed_data/sci/level_1c/cdf/1.0.0"

file_name_list = np.sort(glob.glob(folder_name + "/*.cdf"))
print(file_name_list)

ra_list = []
dec_list = []
epoch_unix_list = []

# for file_name in file_name_list[:]:
#     dat = CDF(file_name)
#     ra_list.append(dat["ra_J2000_deg"][:])
#     dec_list.append(dat["dec_J2000_deg"][:])
#     epoch_unix_list.append(dat["Epoch"][:])
#     dat.close()


# Turn on the latex rendering
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

ra = np.concatenate(ra_list)
dec = np.concatenate(dec_list)
epoch_unix = np.concatenate(epoch_unix_list)

# Get the indices where ra is not in the range 0 to 360 and dec is not in the range -90 to 90
bad_indices = np.where(
    (ra < 0) | (ra > 360) | np.isnan(ra) | (dec < -90) | (dec > 90) | np.isnan(dec)
)

# Remove the bad indices
ra = np.delete(ra, bad_indices)
dec = np.delete(dec, bad_indices)
epoch_unix = np.delete(epoch_unix, bad_indices)

ra_min = np.nanmax([0, np.nanmin(ra)])
ra_max = np.nanmin([360, np.nanmax(ra)])
dec_min = np.nanmax([-90, np.nanmin(dec)])
dec_max = np.nanmin([90, np.nanmax(dec)])

ra_min = ra_min_look_direction - 10
ra_max = ra_max_look_direction + 10
dec_min = dec_min_look_direction - 5
dec_max = dec_max_look_direction + 5

# Turn on the dark mode
plt.style.use("dark_background")

# Make the ra and dec histograms
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=0.32, wspace=0.0)

ax[0].hist(ra, bins=100, range=(ra_min, ra_max), color="r", alpha=0.5)  # RA
ax[0].set_xlabel("RA [deg]")
ax[0].set_ylabel("Counts")
ax[0].grid()
ax[0].set_title(f"LEXI look direction RA for about {len(ra)} points")
ax[0].set_xlim(ra_min, ra_max)
ax[0].set_yscale("log")

ax[1].hist(dec, bins=100, range=(dec_min, dec_max), color="b", alpha=0.5)  # Dec
ax[1].set_xlabel("Dec [deg]")
ax[1].set_ylabel("Counts")
ax[1].grid()
ax[1].set_title(f"LEXI look direction Dec for about {len(dec)} points")
ax[1].set_xlim(dec_min, dec_max)
ax[1].set_yscale("log")
plt.savefig("../figures/ra_dec_hist.png")

plt.close()

# Plot 2d histogram of ra and dec with number of counts as the color
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
hb = ax.hexbin(ra, dec, gridsize=[360, 100], cmap="inferno", bins="log", mincnt=100)
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Dec [deg]")
ax.grid()

cb = fig.colorbar(hb, ax=ax)
cb.set_label("Counts")
ax.set_xlim(ra_min, ra_max)
ax.set_ylim(dec_min, dec_max)
plt.savefig("../figures/ra_dec_2d_hist.png")

plt.close()

# Make a time series plot of RA, dec
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].scatter(epoch_unix, ra, label="RA", s=0.1, c="r", marker=".", alpha=0.5)
ax[0].set_xlabel("Time")
ax[0].set_ylabel("RA [deg]")
ax[0].set_ylim(ra_min, ra_max)
ax[0].grid()
ax[0].set_xlim(t_min, t_max)

# Rotate the x-axis labels by 45 degrees
ax[0].tick_params(axis="x", rotation=45)

ax[1].scatter(epoch_unix, dec, label="dec", s=0.1, c="b", marker=".", alpha=0.5)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Dec [deg]")
ax[1].set_ylim(dec_min, dec_max)
ax[1].grid()
ax[1].set_xlim(t_min, t_max)

# Rotate the x-axis labels by 45 degrees
ax[1].tick_params(axis="x", rotation=45)

plt.savefig("../figures/ra_dec_time_series.png")
