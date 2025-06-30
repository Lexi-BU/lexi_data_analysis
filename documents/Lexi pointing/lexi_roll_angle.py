import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d


#
# function to convert quaternion to a rotation matrix
def quat2dcm(q):
    # "q" is a 4-element numpy array holding the quaternion,
    q0 = q[0]  # the scalar element
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    dcm[0, 1] = 2 * (q1 * q2 + q0 * q3)
    dcm[0, 2] = 2 * (q1 * q3 - q0 * q2)
    dcm[1, 0] = 2 * (q1 * q2 - q0 * q3)
    dcm[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    dcm[1, 2] = 2 * (q2 * q3 + q0 * q1)
    dcm[2, 0] = 2 * (q1 * q3 + q0 * q2)
    dcm[2, 1] = 2 * (q2 * q3 - q0 * q1)
    dcm[2, 2] = q0**2 - q1**2 - q2**2 + q3**2

    return dcm


# these times used to filter the data to only that around sunset
start_date = "2025-03-16 19:45:00"
end_date = "2025-03-16 21:15:00"
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)


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

#
# section to read in the LEXI boresight pointing during sunset
#

# Read the CSV file into a pandas DataFrame
radec_file_path = "~/projects/lexi/surface_real/final_pointing/lexi_look_direction_data_sunset.csv"
df_radec = pd.read_csv(radec_file_path)

df_radec["Epoch"] = pd.to_datetime(df_radec["Epoch"])

# Remove timezone information from dataframe column
df_radec["Epoch"] = df_radec["Epoch"].dt.tz_localize(None)


cutoff_date = pd.to_datetime("2025-03-16 21:15:00")

# If your datetime is in a column named 'timestamp'
df_radec = df_radec[df_radec["Epoch"] <= cutoff_date]

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_xlabel("Epoch")
ax2.set_ylabel("DEC, deg")
ax2.plot(df_radec["Epoch"], df_radec["dec_lexi"], marker="s", label="DEC")
ax2.tick_params(axis="y")
plt.xticks(rotation=45)  # Rotate labels for better fit
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.set_xlabel("Epoch")
ax3.set_ylabel("RA, deg")
ax3.plot(df_radec["Epoch"], df_radec["ra_lexi"], marker="s", label="RA")
ax3.tick_params(axis="y")
plt.xticks(rotation=45)  # Rotate labels for better fit
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))


# # Get only every Nth tick (e.g., every 5th tick)
# N = 1
# ticks = ax2.get_xticks()
# labels = ax2.get_xticklabels()
# ax2.set_xticks(ticks[::N])
# ax2.set_xticklabels(labels[::N])


# plt.show()


#
# function to convert a unit vector expressed in the lander body frame
# into the theta 1 and theta 1 angles
def thetaFromUnitVec(v):
    # returns a numpy array holding theta1 and theta2
    thetas = np.zeros(2)
    thetas[0] = np.atan2(v[1], v[2])
    thetas[1] = np.asin(v[0])
    return thetas


#
# section to read in the attitude quaternions output by the FreeFlyer simulation
#
def interpolate_scipy(df, time_column, data_col, target_time, method="linear"):

    # Convert to datetime
    if isinstance(target_time, str):
        target_time = pd.to_datetime(target_time)

    # Convert time to numeric for scipy
    if time_column in df.columns:
        time_series = pd.to_datetime(df[time_column])
    else:
        time_series = df.index

    time_numeric = time_series.astype(np.int64)  # Convert to nanoseconds
    target_numeric = pd.to_datetime(target_time).value

    # Check bounds
    if target_numeric < time_numeric.min() or target_numeric > time_numeric.max():
        print(f"Warning: Target time outside data range")
        return None

    # Create interpolation function
    f = interp1d(
        time_numeric, df[data_col], kind=method, bounds_error=False, fill_value="extrapolate"
    )

    # Interpolate at target time
    return f(target_numeric)


#
# NOTE: ITRF is virtually identical to J2000 and can be treated as the same
#
# NOTE: "Lander Nominal Body Frame" is the ideal situation, where lander X points to zenith,
# lander Y points to north, and lander Z points ot west. The "Lander Actual Body Frame" is
# based on how the lander actually landed based on Firefly post-landing estimated quaternions delivered during mission
#


# this is the file that contains the quaternions that map from ITRF to Lander-Nominal
nominal_landing_file_path = "/home/vetinari/Desktop/git/Lexi-Bu/lexi_data_analysis/documents/Lexi pointing/LEXI_q_LandNom_ITRF.csv"
df_q_LandNom_ITRF = pd.read_csv(nominal_landing_file_path)
# print(df_q_LandNom_ITRF.head())
# print("Actual column names:")
# print(df_q_LandNom_ITRF.columns.tolist())
df_q_LandNom_ITRF["Epoch (UTC)"] = pd.to_datetime(df_q_LandNom_ITRF["Epoch (UTC)"])
# let's consider only the data between a certain time around sunset
df_q_LandNom_ITRF = df_q_LandNom_ITRF[
    (df_q_LandNom_ITRF["Epoch (UTC)"] > start_date) & (df_q_LandNom_ITRF["Epoch (UTC)"] < end_date)
]
# print(df_q_LandNom_ITRF.head())
df_q_LandNom_ITRF = df_q_LandNom_ITRF.reset_index(drop=True)

actual_landing_file_path = "/home/vetinari/Desktop/git/Lexi-Bu/lexi_data_analysis/documents/Lexi pointing/LEXI_q_LandAct_ITRF.csv"
df_q_LandAct_ITRF = pd.read_csv(actual_landing_file_path)
df_q_LandAct_ITRF["Epoch (UTC)"] = pd.to_datetime(df_q_LandAct_ITRF["Epoch (UTC)"])
df_q_LandAct_ITRF = df_q_LandAct_ITRF[
    (df_q_LandAct_ITRF["Epoch (UTC)"] > start_date) & (df_q_LandAct_ITRF["Epoch (UTC)"] < end_date)
]
df_q_LandAct_ITRF = df_q_LandAct_ITRF.reset_index(drop=True)


q_LandNom_ITRF = np.zeros(4)  # quaternion that maps from ITRF to Nominal Lander Body frame
dcm_LandNom_ITRF = np.zeros((3, 3))  # DCM that maps from ITRF to Nominal Lander Body frame
q_LandAct_ITRF = np.zeros(4)  # quaternion that maps from ITRF to Actual Lander Body frame
dcm_LandAct_ITRF = np.zeros((3, 3))  # DCM that maps from ITRF to Actual Lander Body frame

vboresight_ITRF = np.zeros(3)  # boresight vector expressed in ITRF frame
vboresight_LandNom = np.zeros(3)  # boresight vector expressed in Nominal Lander Body frame
vboresight_LandAct = np.zeros(3)  # boresight vector expressed in Act Lander Body frame

thetaAngles = np.zeros(2)  # the calculated theta1 and theta2 in radians

num_rows = len(df_q_LandAct_ITRF)

history_theta1_deg = np.zeros(num_rows)
history_theta2_deg = np.zeros(num_rows)

# NOTE: I'm only iterating over one of the pandas dataframes, but this works because I know that both
# dataframes are output at the same time step
# for index, row in df_q_LandNom_ITRF.iterrows():
for i, index in enumerate(df_q_LandNom_ITRF.index):
    row = df_q_LandNom_ITRF.loc[index]

    row_act = df_q_LandAct_ITRF.loc[index]

    q_LandNom_ITRF[0] = row["q0"]
    q_LandNom_ITRF[1] = row["q1"]
    q_LandNom_ITRF[2] = row["q2"]
    q_LandNom_ITRF[3] = row["q3"]

    q_LandAct_ITRF[0] = row_act["q0"]
    q_LandAct_ITRF[1] = row_act["q1"]
    q_LandAct_ITRF[2] = row_act["q2"]
    q_LandAct_ITRF[3] = row_act["q3"]

    dcm_LandNom_ITRF = quat2dcm(q_LandNom_ITRF)
    dcm_LandAct_ITRF = quat2dcm(q_LandAct_ITRF)

    boresight_RA = interpolate_scipy(
        df_radec, "Epoch", "ra_lexi", row["Epoch (UTC)"], method="linear"
    )
    boresight_DEC = interpolate_scipy(
        df_radec, "Epoch", "dec_lexi", row["Epoch (UTC)"], method="linear"
    )

    # print(boresight_RA,',',boresight_DEC)

    boresight_DEC_rad = boresight_DEC * np.pi / 180
    boresight_RA_rad = boresight_RA * np.pi / 180

    # compute the boresight vector in the ITRF frame
    x_ITRF = np.cos(boresight_DEC_rad) * np.cos(boresight_RA_rad)
    y_ITRF = np.cos(boresight_DEC_rad) * np.sin(boresight_RA_rad)
    z_ITRF = np.sin(boresight_DEC_rad)

    vboresight_ITRF = (x_ITRF, y_ITRF, z_ITRF)

    # use the rotation matrix to express the boresight vector in the actual lander body frame
    v_boresight_LandAct = dcm_LandAct_ITRF @ vboresight_ITRF

    # calculate theta1 and theta2 from the boresight
    thetaAngles = thetaFromUnitVec(v_boresight_LandAct)

    theta1_deg = thetaAngles[0] * 180 / np.pi
    theta2_deg = thetaAngles[1] * 180 / np.pi

    #  print(theta1_deg,',',theta2_deg)

    history_theta1_deg[index] = theta1_deg
    history_theta2_deg[index] = theta2_deg


# as a sanity check, theta1 and theta2 should be nearly constant, since they are in the
# body frame and LEXI was parked during the sunset time, so compute the mean and std:
mean_theta1_deg = history_theta1_deg.mean()
std_theta1_deg = history_theta1_deg.std()
mean_theta2_deg = history_theta2_deg.mean()
std_theta2_deg = history_theta2_deg.std()

print("mean and std of theta1 (deg):\n")
print(mean_theta1_deg, ",", std_theta1_deg)
print("mean and std of theta2 (deg):\n")
print(mean_theta2_deg, ",", std_theta2_deg)

plt.show()
