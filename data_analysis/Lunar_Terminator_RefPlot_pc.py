from datetime import datetime

import irfpy
import matplotlib.pyplot as plt
import numpy as np
import pylunar
from irfpy.moon import moon_map
from mpl_toolkits.basemap import Basemap
from pylunar import MoonInfo

# Set my working folder
Folder = "/Users/emilatz/Dropbox/Research/LEXI/MoonMap/"
# Create a MoonInfo object for a specific date and observer location
# This is needed just to "create" the moon in the code
observer_lat = 0  # Earth obs latitude
observer_lon = 90  # Earth obs longitude
observer_alt = 0.0
date = datetime(2025, 3, 10, 19, 28, 0)  # This needs to be UTC
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
# Solar_altituide gives the angle of the sun above the horizon for the location
solaralt = moon_info.solar_altitude(BGM1)
print(f"Solar Alt: {solaralt:.2f} degrees")

map = moon_map.MoonMapSmall()
map_on_sphere = map.gridsphere()
blon, blat = map_on_sphere.get_bgrid(delta=0.001)  # Grid specifying the longitude and latitude.
level = np.array(map_on_sphere.get_average())

m = Basemap(
    projection="ortho", lon_0=61.81, lat_0=18.562
)  # ,satellite_height=1000.,resolution='l')

# Use contourf to plot the moon surface map
m.contourf(blon[1:, 1:], blat[1:, 1:], level, cmap="gray", levels=50, alpha=0.5, latlon=True)
# m.pcolormesh(blon, blat, level, latlon=True, cmap="gray")
plt.savefig(f"../figures/tutorial/moonmap_03.png")
plt.close()
