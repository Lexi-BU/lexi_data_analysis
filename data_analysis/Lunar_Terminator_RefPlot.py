from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pylunar
from irfpy.moon import moon_map
from mpl_toolkits.basemap import Basemap
from pylunar import MoonInfo

# This is needed just to "create" the moon in the code
observer_lat = 0  # Earth obs latitude
observer_lon = 0  # Earth obs longitude
observer_alt = 0.0
date = datetime(2025, 3, 3, 19, 28, 0)  # This needs to be UTC
moon_info = MoonInfo([observer_lat, 0, 0], [observer_lon, 0, 0])  # Create the moon for PyLunar
moon_info.update(date)  # Update the time so the moon is correctly "lit" by the sun

# This class, Lunar Feature, creates a instance of the BGM1 landing site
BGM1 = pylunar.LunarFeature(
    name="BGM1",
    diameter=0.001,
    latitude=18.562,
    longitude=61.81,
    delta_latitude=0,
    delta_longitude=0,
    feature_type="Satellite Feature",
    quad_name="LAC-44",
    quad_code="LAC44",
    code_name="BGM1",
    lunar_club_type="BGM1",
)

# Set up data
map = moon_map.MoonMapSmall()
map_on_sphere = map.gridsphere()
blon, blat = map_on_sphere.get_bgrid(delta=0.0001)
level = np.array(map_on_sphere.get_average())

# Trim arrays to the smallest common size
min_rows = min(blon.shape[0], blat.shape[0], level.shape[0])
min_cols = min(blon.shape[1], blat.shape[1], level.shape[1])

blon_trimmed = blon[:min_rows, :min_cols]
blat_trimmed = blat[:min_rows, :min_cols]
level_trimmed = level[:min_rows, :min_cols]

# Replace masked or invalid values with np.nan
blon_filled = np.ma.filled(blon_trimmed, fill_value=np.nan)
blat_filled = np.ma.filled(blat_trimmed, fill_value=np.nan)
level_filled = np.ma.filled(level_trimmed, fill_value=np.nan)

level_filled[level_filled == 0] = np.nan

m = Basemap(projection="ortho", lon_0=61.81, lat_0=18.562)

# Plot with Basemap using contourf
fig = plt.figure(figsize=(10, 10))
x, y = m(blon_filled, blat_filled)

cs = m.contourf(
    blon_filled, blat_filled, level_filled, cmap="gray", levels=50, alpha=1, latlon=True
)

# Display the axes grid
# m.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=10)
# m.drawmeridians(np.arange(-180, 180, 20), labels=[0, 0, 0, 1], fontsize=10)

# Add BGM1 landing site marker
x_site, y_site = m(61.81, 18.562)
# Convert coordinates to map projection
# x_site_lon, y_site_lat = m(61.81, 18.562)
m.plot(x_site, y_site, "ro", markersize=8)
#
# # Add arrow and text annotation
plt.annotate(
    "BGM1 Landing Site",
    xy=(x_site, y_site),
    xytext=(x_site + 200000, y_site + 200000),
    arrowprops=dict(arrowstyle="->", color="cyan", linewidth=2),
    color="cyan",
    fontsize=12,
    ha="left",
    va="bottom",
)
# Add title and colorbar
plt.title("Moon Surface Map with BGM1 Landing Site", fontsize=16)
plt.colorbar(cs, shrink=0.7)

figure_folder = Path("../figures/tutorial/")
figure_folder.mkdir(parents=True, exist_ok=True)
figure_folder = figure_folder.expanduser().resolve()
figure_name = figure_folder / "moon_surface_map_contourf.png"
plt.tight_layout()
plt.savefig(figure_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"Figure saved to folder: {figure_folder}")
plt.close()
