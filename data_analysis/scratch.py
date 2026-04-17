import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot([0, 1], [0, 1])
sc = ax2.scatter([0, 1], [0, 1], c=[0, 1])

# Use inset_axes to make the colorbar identical in width to ax2
cax = ax2.inset_axes([0, 1.05, 1, 0.05])
cbar = plt.colorbar(sc, cax=cax, orientation="horizontal", label="Delay")
cax.xaxis.set_ticks_position("top")
cax.xaxis.set_label_position("top")

fig.savefig("scratch.png")
