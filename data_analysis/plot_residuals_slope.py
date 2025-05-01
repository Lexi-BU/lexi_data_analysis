import importlib
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_name = "../data/line_profile_best_fit_theta_offset_2042_2102.csv"

df = pd.read_csv(file_name)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    df["theta"],
    df["slope"],
    marker="o",
    linestyle="None",
    color="black",
    label="Slope",
)

ax.set_ylim(100, 9e3)
# ax.set_yscale("log")
ax2 = ax.twinx()
ax2.plot(
    df["theta"],
    df["residuals"],
    marker="o",
    linestyle="None",
    color="red",
    label="Residuals",
)
ax2.set_ylim(3e5, 7e6)
# ax2.set_yscale("log")
ax.set_xlabel("Theta (degrees)")
ax.set_ylabel("Slope [Normalized Units]")
ax2.set_ylabel("Residuals [Normalized Units]")
ax.set_title("Slope and Residuals of best fit line vs Theta")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax.grid()
plt.tight_layout()
plt.savefig(
    "../figures/slope_residuals_vs_theta_v2.png", dpi=300, bbox_inches="tight", pad_inches=0.1
)
plt.close()
