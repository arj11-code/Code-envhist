import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
import os

files = sorted(glob.glob("full_trajectory_*.csv"))

START = np.array([0.9999, 0.0001])
TOL = 1e-3

# clean steps (<=2 species, merge similar)
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

        # group similar species
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

        # keep parent (smallest ID)
        reduced = []
        for g in groups:
            chosen = min(g, key=lambda x: x[0])
            reduced.append(chosen)

        # keep at most 2
        if len(reduced) > 2:
            reduced = sorted(reduced, key=lambda x: x[0])[:2]

        cleaned.append((step, reduced))
    return cleaned

# remove duplicate rows
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

    # build parent lookup once
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

        # only trigger on strict 1 -> 2 -> 1 pattern
        if not (len(prev) == 1 and len(curr) == 2 and len(nxt) == 1):
            continue

        prev_id = prev[0][0]
        next_id = nxt[0][0]

        # find which species in curr is descendant of prev
        keep = None
        for s in curr:
            sid = s[0]
            # check if sid descended from prev_id
            p = sid
            while p != -1:
                if p == prev_id:
                    keep = s
                    break
                p = parent_map.get(p, -1)
            if keep is not None:
                break

        # only modify if we found a valid continuation
        if keep is not None:
            out[i] = (out[i][0], [keep])
    return out

# main loop
for fname in files:
    df = pd.read_csv(fname)

    # clean data
    cleaned = clean_steps(df)
    cleaned = deduplicate_steps(cleaned)
    cleaned = remove_one_step_mutants(cleaned, df)

    # debug: save cleaned trajectory
    rows = []
    for step, species in cleaned:
        row = [step]
        for i in range(2):  # at most 2 species after cleaning
            if i < len(species):
                sid, fr, phi = species[i]
                # recover parent from original df (safe lookup)
                pid = -1
                for _, r in df.iterrows():
                    for k in range(1, 5):
                        if not pd.isna(r[f"id_{k}"]) and int(r[f"id_{k}"]) == sid:
                            pid = int(r[f"parent_{k}"])
                            break
                row.extend([sid, pid, fr, phi])
            else:
                row.extend([np.nan, np.nan, np.nan, np.nan])
        rows.append(row)

    clean_df = pd.DataFrame(rows, columns=["mutation_step", "id_1","parent_1","f_r_1","phi_1", "id_2","parent_2","f_r_2","phi_2"])
    out_debug = os.path.splitext(os.path.basename(fname))[0] + "_cleaned.csv"
    #clean_df.to_csv(out_debug, index=False)

    # rebuild lineage structure
    lineages = {}
    parents = {}
    for step, species in cleaned:
        for sid, fr, phi in species:
            if sid not in lineages:
                lineages[sid] = []
                parents[sid] = None
            lineages[sid].append((step, fr, phi))

    # recover parent map
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

    # build segments with time
    segments = []
    seg_time = []

    # lineage continuation
    for sid, traj in lineages.items():
        traj = sorted(traj, key=lambda x: x[0])
        for k in range(len(traj) - 1):
            p1 = traj[k]
            p2 = traj[k+1]
            segments.append([[p1[1], p1[2]], [p2[1], p2[2]]])
            seg_time.append(p2[0])

    # branching
    for sid, pid in parents.items():
        # skip invalid child or root
        if sid not in lineages or pid == -1:
            continue

        child_traj = sorted(lineages[sid], key=lambda x: x[0])
        child_start = child_traj[0]

        if pid in lineages:
            # normal case: parent present
            parent_traj = sorted(lineages[pid], key=lambda x: x[0])
            parent_point = min(parent_traj, key=lambda x: abs(x[0] - child_start[0]))
        else:
            # fallback: climb ancestry
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

    # connect START to earliest point
    all_points = []
    for sid, traj in lineages.items():
        for p in traj:
            all_points.append(p)

    if len(all_points) > 0:
        earliest = min(all_points, key=lambda x: x[0])
        segments.append([START, [earliest[1], earliest[2]]])
        seg_time.append(earliest[0])

    if len(segments) == 0:
        continue

    segments = np.array(segments)
    seg_time = np.array(seg_time)

    # sort by time for correct gradient
    order = np.argsort(seg_time)
    segments = segments[order]

    # grey to red gradient
    n = len(segments)
    colors = np.zeros((n, 4))
    colors[:, 0] = np.linspace(0.5, 1.0, n)
    colors[:, 1] = np.linspace(0.5, 0.0, n)
    colors[:, 2] = np.linspace(0.5, 0.0, n)
    colors[:, 3] = np.linspace(0.2, 0.6, n)

    # plot
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    lc = LineCollection(segments, colors=colors, capstyle='round', joinstyle='round', linewidths=2)
    lc.set_rasterized(True)
    ax.add_collection(lc)
    ax.set_xlim((0.6, 1.03))
    ax.set_ylim(-0.005, 0.06)
    #ax.plot(START[0], START[1], marker='o', markersize=3, color='grey', zorder=5)
    xticks = (0.6, 0.8, 1)
    yticks = (0, 0.03, 0.06)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # force .2f formatting
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel("Invader")
    ax.set_ylabel("Resident")

    #ax.autoscale()
    plt.savefig("fig3_f.svg", format="svg", dpi=2000, bbox_inches="tight")
    plt.show()
