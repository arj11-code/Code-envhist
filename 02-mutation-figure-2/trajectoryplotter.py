import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib as mpl
import os

# matplotlib formatting
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"

files = sorted(glob.glob("full_trajectory_*.csv"))

START = np.array([0.9999, 0.0001])
TOL = 1e-3

def clean_steps(df):
    cleaned = []
    for _, row in df.iterrows():
        step = row["mutation_step"]
        species = []
        for i in range(1, 5):
            sid = row[f"id_{i}"]
            fr = row[f"f_r_{i}"]
            phi = row[f"phi_{i}"]
            if pd.isna(sid):
                continue
            species.append((int(sid), fr, phi))

        if len(species) == 0:
            cleaned.append((step, []))
            continue

        groups = []
        for s in species:
            placed = False
            for g in groups:
                rep = g[0]
                if np.linalg.norm([s[1]-rep[1], s[2]-rep[2]]) < TOL:
                    g.append(s)
                    placed = True
                    break
            if not placed:
                groups.append([s])

        reduced = []
        for g in groups:
            chosen = min(g, key=lambda x: x[0])
            reduced.append(chosen)
        if len(reduced) > 2:
            reduced = sorted(reduced, key=lambda x: x[0])[:2]

        cleaned.append((step, reduced))
    return cleaned

def deduplicate_steps(cleaned):
    out = []
    seen = set()
    for step, species in cleaned:
        key = tuple(sorted([s[0] for s in species]))
        if key not in seen:
            seen.add(key)
            out.append((step, species))
    return out

def remove_one_step_mutants(cleaned, df):
    out = cleaned.copy()
    parent_map = {}
    for _, row in df.iterrows():
        for i in range(1, 5):
            sid = row.get(f"id_{i}", np.nan)
            pid = row.get(f"parent_{i}", -1)
            if not pd.isna(sid):
                parent_map[int(sid)] = int(pid)

    for i in range(1, len(cleaned)-1):
        prev = out[i-1][1]
        curr = out[i][1]
        nxt = out[i+1][1]
        if not (len(prev) == 1 and len(curr) == 2 and len(nxt) == 1):
            continue

        prev_id = prev[0][0]
        keep = None
        for s in curr:
            sid = s[0]
            p = sid
            while p != -1:
                if p == prev_id:
                    keep = s
                    break
                p = parent_map.get(p, -1)
            if keep is not None:
                break

        if keep is not None:
            out[i] = (out[i][0], [keep])
    return out

fig, ax = plt.subplots(figsize=(2, 2))

counter = 0
for fname in files:
    if counter <= 130:
        print(counter)
        counter += 1
        df = pd.read_csv(fname)

        cleaned = clean_steps(df)
        cleaned = deduplicate_steps(cleaned)
        cleaned = remove_one_step_mutants(cleaned, df)

        # rebuild lineage structure
        lineages = {}
        parents = {}
        for step, species in cleaned:
            for sid, fr, phi in species:
                if sid not in lineages:
                    lineages[sid] = []
                    parents[sid] = None
                lineages[sid].append((step, fr, phi))

        # parent map
        for _, row in df.iterrows():
            for i in range(1, 5):
                sid = row[f"id_{i}"]
                pid = row[f"parent_{i}"]
                if pd.isna(sid):
                    continue
                sid = int(sid)
                pid = int(pid)
                if parents.get(sid) is None:
                    parents[sid] = pid

        segments = []
        seg_time = []

        # lineage continuation
        for sid, traj in lineages.items():
            traj = sorted(traj, key=lambda x: x[0])
            for k in range(len(traj) - 1):
                p1, p2 = traj[k], traj[k+1]
                segments.append([[p1[1], p1[2]], [p2[1], p2[2]]])
                seg_time.append(p2[0])

        # branching (with ancestor fallback)
        for sid, pid in parents.items():
            if sid not in lineages or pid == -1:
                continue

            child_traj = sorted(lineages[sid], key=lambda x: x[0])
            child_start = child_traj[0]

            if pid in lineages:
                parent_traj = sorted(lineages[pid], key=lambda x: x[0])
                parent_point = min(parent_traj, key=lambda x: abs(x[0] - child_start[0]))
            else:
                p = pid
                found = None
                while p != -1:
                    if p in lineages:
                        found = p
                        break
                    p = parents.get(p, -1)
                if found is None:
                    continue
                parent_traj = sorted(lineages[found], key=lambda x: x[0])
                parent_point = min(parent_traj, key=lambda x: abs(x[0] - child_start[0]))

            segments.append([[parent_point[1], parent_point[2]], [child_start[1], child_start[2]]])
            seg_time.append(child_start[0])

        # connect start
        all_points = [p for traj in lineages.values() for p in traj]
        if len(all_points) > 0:
            earliest = min(all_points, key=lambda x: x[0])
            segments.append([START, [earliest[1], earliest[2]]])
            seg_time.append(earliest[0])

        if len(segments) == 0:
            continue

        segments = np.array(segments)
        seg_time = np.array(seg_time)
        order = np.argsort(seg_time)
        segments = segments[order]

        # gradient (per trajectory)
        n = len(segments)
        colors = np.zeros((n, 4))
        colors[:, 0] = np.linspace(0.5, 1.0, n)
        colors[:, 1] = np.linspace(0.5, 0.0, n)
        colors[:, 2] = np.linspace(0.5, 0.0, n)
        colors[:, 3] = np.linspace(0.2, 0.5, n)

        lc = LineCollection(segments, colors=colors, linewidths=0.6)
        lc.set_rasterized(True)
        ax.add_collection(lc)

# adaptive dynamics overlay
df_ad = pd.read_csv("adaptive_dynamics_trajectory.csv", header=None)
traj_ad = df_ad.iloc[:, :2].apply(pd.to_numeric, errors="coerce").values
traj_ad = traj_ad[~np.isnan(traj_ad).any(axis=1)]

def unique_steps(arr):
    diff = np.any(np.diff(arr, axis=0) != 0, axis=1)
    keep = np.concatenate([[True], diff])
    return arr[keep]

traj_ad = unique_steps(traj_ad)
ax.plot(traj_ad[:, 0], traj_ad[:, 1], color="blue", linewidth=2, label="Adaptive dynamics")

ax.set_xlabel(r"$f_r$")
ax.set_ylabel(r"$\phi_R^\mathrm{0,max}$")
ax.legend(handles=[Line2D([], [], color="red", linewidth=1, label="Simulated trajectories"), Line2D([], [], color="blue", linewidth=2, label="Adaptive dynamics")], frameon=False, fontsize=8, handlelength=1.0, handletextpad=0.4, loc="lower left")
plt.savefig("Heatmap_branching.svg", format="svg", bbox_inches="tight", dpi=2000)
plt.show()
