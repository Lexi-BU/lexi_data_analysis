import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DX_to_BX = np.cos((180 - 55.36) * np.pi / 180)
DX_to_BY = np.cos((38.17) * np.pi / 180)
DX_to_BZ = np.cos((75.96) * np.pi / 180)

DY_to_BX = np.cos((180 - 76.31) * np.pi / 180)
DY_to_BY = np.cos((116.02) * np.pi / 180)
DY_to_BZ = np.cos((29.89) * np.pi / 180)

DZ_to_BX = np.cos((180 - 142.0) * np.pi / 180)
DZ_to_BY = np.cos((64.19) * np.pi / 180)
DZ_to_BZ = np.cos((64.19) * np.pi / 180)

# Rdb rotates vectors from Fb to Fd
Rdb = np.zeros((3, 3))
Rdb[0, 0] = DX_to_BX
Rdb[0, 1] = DX_to_BY
Rdb[0, 2] = DX_to_BZ
Rdb[1, 0] = DY_to_BX
Rdb[1, 1] = DY_to_BY
Rdb[1, 2] = DY_to_BZ
Rdb[2, 0] = DZ_to_BX
Rdb[2, 1] = DZ_to_BY
Rdb[2, 2] = DZ_to_BZ

Rbd = Rdb.transpose()


print("\nRdb:")
print(Rdb)

# Sum the rows (sum of elements in each row)
# axis=1 means we sum along the columns, which gives us the row sums
row_sums = np.linalg.norm(Rdb, axis=1)

print("\nSum of each row:")
print(row_sums)

col_sums = np.linalg.norm(Rdb, axis=0)

print("\nSum of each col:")
print(col_sums)

print("\nInv - Transpose:")
print(np.linalg.inv(Rdb) - Rdb.transpose())

# Solve for the "theta2", the "roll" angle
th3_deg = np.atan2(-Rdb[1, 0], Rdb[0, 0]) * 180 / np.pi
print("\n Roll angle (theta3) in degrees:")
print(th3_deg)

# even though we don't really need it, also solve for th1 and th2:
th2_deg = np.asin(Rdb[2, 0]) * 180 / np.pi
th1_deg = np.atan2(-Rdb[2, 1], Rdb[2, 2]) * 180 / np.pi
print("\n Theta1 in degrees:")
print(th1_deg)
print("\n Theta2 in degrees:")
print(th2_deg)


def plot_vector_as_arrow(
    vector,
    ax,
    color="red",
    alpha=0.8,
    arrow_length_ratio=0.3,
    thickness=0.05,
    head_width=0.2,
    head_length=0.2,
    linestyle="-",
):
    """
    Plot a 3D vector as a thick arrow centered at the origin

    Parameters:
    - vector: [x, y, z] coordinates
    - ax: matplotlib 3D axis
    - color: arrow color
    - alpha: transparency
    - arrow_length_ratio: ratio of head length to arrow length
    - thickness: thickness of the arrow shaft
    - head_width: width of the arrow head
    - head_length: length of the arrow head
    """
    # Normalize and get vector length
    vector = np.array(vector)
    length = np.linalg.norm(vector)
    unit_vector = vector / length

    # Start and end points (centered on origin)
    start = np.array([0, 0, 0])
    end = vector

    # Arrow shaft (cylinder)
    # Move back by half the head_length
    shaft_end = end - unit_vector * head_length

    # Plot the shaft as a line for simplicity
    ax.plot(
        [start[0], shaft_end[0]],
        [start[1], shaft_end[1]],
        [start[2], shaft_end[2]],
        color=color,
        linewidth=thickness * 20,
        alpha=alpha,
        linestyle=linestyle,
    )

    # Plot the arrowhead using a cone
    # For simplicity, we'll use quiver for the arrowhead
    arrow_length = head_length
    ax.quiver(
        shaft_end[0],
        shaft_end[1],
        shaft_end[2],
        unit_vector[0],
        unit_vector[1],
        unit_vector[2],
        length=arrow_length,
        color=color,
        arrow_length_ratio=1.0,
        alpha=alpha,
        linewidth=2,
        linestyle=linestyle,
    )


# Create figure and 3D ax1es
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection="3d")

# Vector to plot [x, y, z]
b1_b = [1, 0, 0]
b2_b = [0, 1, 0]
b3_b = [0, 0, 1]

d1_d = [1, 0, 0]
d2_d = [0, 1, 0]
d3_d = [0, 0, 1]

d1_b = Rbd @ d1_d
d2_b = Rbd @ d2_d
d3_b = Rbd @ d3_d


# Plot the vector
plot_vector_as_arrow(b1_b, ax1, thickness=0.1, head_width=0.3, color="red", linestyle="-")
plot_vector_as_arrow(b2_b, ax1, thickness=0.1, head_width=0.3, color="green", linestyle="-")
plot_vector_as_arrow(b3_b, ax1, thickness=0.1, head_width=0.3, color="blue", linestyle="-")
plot_vector_as_arrow(d1_b, ax1, thickness=0.1, head_width=0.3, color="red", linestyle="--")
plot_vector_as_arrow(d2_b, ax1, thickness=0.1, head_width=0.3, color="green", linestyle="--")
plot_vector_as_arrow(d3_b, ax1, thickness=0.1, head_width=0.3, color="blue", linestyle="--")

# Set equal aspect ratio
ax1.set_box_aspect([1, 1, 1])

# Set ax1is limits
limit = 1.0
ax1.set_xlim(-limit, limit)
ax1.set_ylim(-limit, limit)
ax1.set_zlim(-limit, limit)

# Add ax1is labels
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Add a title
ax1.set_title("3D Vector [1,0,0] as Arrow")

# Add gridlines
ax1.grid(True)

# Show ax1es at the origin
ax1.plot([0, 0], [0, 0], [-limit, limit], "k--", alpha=0.3)
ax1.plot([0, 0], [-limit, limit], [0, 0], "k--", alpha=0.3)
ax1.plot([-limit, limit], [0, 0], [0, 0], "k--", alpha=0.3)

plt.tight_layout()

# Different viewing angles to demonstrate
views = [
    (90, 270, "View 1 (X-Y Plane)"),  # Looking down at X-Y plane
    (0, 0, "View 2 (Y-Z Plane)"),  # Looking at Y-Z plane
    (0, 90, "View 3 (X-Z Plane)"),  # Looking at X-Z plane
    (30, 45, "Default 3D View"),  # Common 3D perspective
]

# Create a 2x2 grid of subplots to show different views
fig2 = plt.figure(figsize=(8, 8))

for i, (elev, azim, title) in enumerate(views, 1):
    ax = fig2.add_subplot(2, 2, i, projection="3d")

    # Plot the vector
    plot_vector_as_arrow(b1_b, ax, thickness=0.1, head_width=0.3, color="red", linestyle="-")
    plot_vector_as_arrow(b2_b, ax, thickness=0.1, head_width=0.3, color="green", linestyle="-")
    plot_vector_as_arrow(b3_b, ax, thickness=0.1, head_width=0.3, color="blue", linestyle="-")
    plot_vector_as_arrow(d1_b, ax, thickness=0.1, head_width=0.3, color="red", linestyle="--")
    plot_vector_as_arrow(d2_b, ax, thickness=0.1, head_width=0.3, color="green", linestyle="--")
    plot_vector_as_arrow(d3_b, ax, thickness=0.1, head_width=0.3, color="blue", linestyle="--")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Set axis limits
    limit = 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    # Add axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Add a title
    ax.set_title("3D Vector [1,0,0] as Arrow")

    # Add gridlines
    ax.grid(True)

    # Show axes at the origin
    ax.plot([0, 0], [0, 0], [-limit, limit], "k--", alpha=0.3)
    ax.plot([0, 0], [-limit, limit], [0, 0], "k--", alpha=0.3)
    ax.plot([-limit, limit], [0, 0], [0, 0], "k--", alpha=0.3)

    plt.tight_layout()

    # Set the view angle
    ax.view_init(elev=elev, azim=azim)

    # Add grid lines for reference
    ax.grid(True)

    #    ax.set_title(f'{title}\nelev={elev}, azim={azim}')
    ax.set_title(f"{title}")

plt.tight_layout()
plt.show()
