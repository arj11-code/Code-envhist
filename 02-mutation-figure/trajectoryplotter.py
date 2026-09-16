import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl

# use Arial for all plot text, keep it editable in PDF export
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"

def unique_steps(arr):
    # keep only rows where something actually changes
    diff = np.any(np.diff(arr, axis=0) != 0, axis=1)
    keep = np.concatenate([[True], diff])
    return arr[keep]

files = sorted(glob.glob("trait_trajectory_*.csv"))

fig, ax = plt.subplots(figsize=(1.5,1.5))

for fname in files:
    df = pd.read_csv(fname, header=None)
    traj = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").values
    traj = traj[~np.isnan(traj).any(axis=1)]
    traj = unique_steps(traj)
    if len(traj) < 2:
        continue

    # build line segments
    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    n = len(segments)

    # grey to red
    colors = np.zeros((n, 4))
    colors[:, 0] = np.linspace(0.5, 1.0, n)   # R
    colors[:, 1] = np.linspace(0.5, 0.0, n)   # G
    colors[:, 2] = np.linspace(0.5, 0.0, n)   # B
    colors[:, 3] = np.linspace(0.2, 0.6, n)  # alpha

    lc = LineCollection(segments, colors=colors, linewidths=0.75)
    lc.set_rasterized(True) # Makes the trajectories not be true trajectories, instead an image
    ax.add_collection(lc)

df_ad = pd.read_csv("adaptive_dynamics_trajectory.csv", header=None)
traj_ad = df_ad.iloc[:, :2].apply(pd.to_numeric, errors="coerce").values
traj_ad = traj_ad[~np.isnan(traj_ad).any(axis=1)]
traj_ad = unique_steps(traj_ad)

ax.plot(traj_ad[:, 0], traj_ad[:, 1], linewidth=2)

ax.set_xlabel(r" value")
ax.set_ylabel(r" value")
ax.autoscale()

ESS_f_r = 0.548533
ESS_phi_R0max = 0.036088
ax.plot(ESS_f_r, ESS_phi_R0max, marker="x", markersize=10, markeredgewidth=2, linestyle="none", color="blue")

# adaptive dynamics line (legend works automatically)
ax.plot(traj_ad[:, 0], traj_ad[:, 1], linewidth=2, color="blue")

# proxy for evolutionary trajectories
evo_handle = Line2D([], [], color="red", linewidth=1)

ax.set_xticks([0.5, 0.75, 1])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.savefig("Phase_space_trajectories.svg", format="svg", bbox_inches="tight", dpi=2000)
plt.show()
