import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(
    "/home/cephadrius/Desktop/git/Lexi-BU/look_direction/data/20241114_LEXIAngleData_20250302Landing.csv"
)

# Select the first 1000 data points
df = df.loc[:]

delta_x = df.el_sun - df.el_earth
delta_y = df.az_sun - df.az_earth

slope = (delta_y / delta_x).values

theta_radians = np.atan2(delta_y, delta_x)
theta_degrees = 90 - theta_radians * 180 / np.pi
plt.figure(figsize=(10, 6))
plt.plot(df.epoch_utc, theta_degrees, label="Slope (theta)", color="blue")
plt.xlabel("Date")
plt.ylabel("Slope (theta)")
plt.title("Slope of Sun and Earth Angles")
plt.legend()
plt.grid()
# Set numebr of ticks to 5
plt.xticks(ticks=range(0, len(df.epoch_utc), max(1, len(df.epoch_utc) // 5)), rotation=20)

plt.tight_layout()
plt.savefig("slope_plot.png")
# plt.show()
