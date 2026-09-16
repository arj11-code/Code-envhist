import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
import os

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10

def draw_coex_boundary(ax):
    for i in range(ND):
        for j in range(NC):
            if not multi_mask[i, j]:
                continue
            x0 = Xedges[j]
            x1 = Xedges[j+1]
            y0 = Yedges[i]
            y1 = Yedges[i+1]
            if j == 0 or not multi_mask[i, j-1]:
                ax.plot([x0, x0], [y0, y1], color="black", linewidth=1.5)
            if j == NC-1 or not multi_mask[i, j+1]:
                ax.plot([x1, x1], [y0, y1], color="black", linewidth=1.5)
            if i == 0 or not multi_mask[i-1, j]:
                ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.5)
            if i == ND-1 or not multi_mask[i+1, j]:
                ax.plot([x0, x1], [y1, y1], color="black", linewidth=1.5)

def draw_split_cell(ax, x0, x1, y0, y1, v_low, v_high, cmap, norm):
    tri_low = Polygon([(x0, y0), (x1, y0), (x0, y1)], facecolor=cmap(norm(v_low)), edgecolor="none")
    tri_high = Polygon([(x1, y1), (x1, y0), (x0, y1)], facecolor=cmap(norm(v_high)), edgecolor="none")
    ax.add_patch(tri_low)
    ax.add_patch(tri_high)

abundance_cutoff = 1e-4
nfiles = 100

data = {}
cleaned_data = {}

# read invasion files
for i in range(nfiles):
    fname = f"invasion_result_{i}.csv"
    if not os.path.exists(fname):
        continue
    df = pd.read_csv(fname)
    D = df["Dilution"].iloc[0]
    cr = df["c_r value"].iloc[0]
    df = df[df["final_abundance"] >= abundance_cutoff]
    data[(D, cr)] = df

# read cleaned files
for i in range(nfiles):
    fname = f"cleaned_result_{i}.csv"
    if not os.path.exists(fname):
        continue
    df = pd.read_csv(fname)
    D = df["Dilution"].iloc[0]
    cr = df["c_r value"].iloc[0]
    cleaned_data[(D, cr)] = df

# construct parameter grid
D_vals = sorted({k[0] for k in data.keys()} | {k[0] for k in cleaned_data.keys()})
cr_vals = sorted({k[1] for k in data.keys()} | {k[1] for k in cleaned_data.keys()})
D_vals = np.array(D_vals)
cr_vals = np.array(cr_vals)
ND = len(D_vals)
NC = len(cr_vals)
D_index = {v: i for i, v in enumerate(D_vals)}
cr_index = {v: j for j, v in enumerate(cr_vals)}

f_r_mat = np.full((ND, NC), np.nan)
phi_mat = np.full((ND, NC), np.nan)

single_mask = np.zeros((ND, NC), dtype=bool)
multi_mask = np.zeros((ND, NC), dtype=bool)
missing_mask = np.ones((ND, NC), dtype=bool)

for (D, cr), df in data.items():
    i = D_index[D]
    j = cr_index[cr]
    missing_mask[i, j] = False
    if len(df) == 1 and df["f_r"].iloc[0] > 0.01:
        single_mask[i, j] = True
        f_r_mat[i, j] = df["f_r"].iloc[0]
        phi_mat[i, j] = df["phi_R0max"].iloc[0]
    elif len(df) > 1:
        multi_mask[i, j] = True

# fill missing values from cleaned data
for (D, cr), df in cleaned_data.items():
    i = D_index[D]
    j = cr_index[cr]
    if np.isnan(f_r_mat[i, j]):
        f_r_mat[i, j] = df["f_r"].iloc[0]
        phi_mat[i, j] = df["phi_R0max"].iloc[0]

def edges(vals):
    ratios = vals[1:] / vals[:-1]
    r0 = ratios[0]
    r1 = ratios[-1]
    left = vals[0] / np.sqrt(r0)
    right = vals[-1] * np.sqrt(r1)
    mids = np.sqrt(vals[:-1] * vals[1:])
    return np.concatenate([[left], mids, [right]])

Xedges = edges(cr_vals)
Yedges = edges(D_vals)

cmap = mpl.cm.Blues.copy()
cmap.set_bad("dimgray")

# coexistence values ordered by f_r, used to split cells with more than one surviving strain
fr_low = np.full((ND, NC), np.nan)
fr_high = np.full((ND, NC), np.nan)
phi_low = np.full((ND, NC), np.nan)
phi_high = np.full((ND, NC), np.nan)

for (D, cr), df in data.items():
    if len(df) > 1:
        i = D_index[D]
        j = cr_index[cr]
        fr_vals = df["f_r"].values
        phi_vals = df["phi_R0max"].values
        order = np.argsort(fr_vals)
        fr_low[i, j] = fr_vals[order[0]]
        fr_high[i, j] = fr_vals[order[1]]
        phi_low[i, j] = phi_vals[order[0]]
        phi_high[i, j] = phi_vals[order[1]]

# f_r combined heatmap
fr_single = f_r_mat.copy()
fr_single[multi_mask] = np.nan
norm = mpl.colors.Normalize(vmin=np.nanmin([fr_single, fr_low, fr_high]), vmax=np.nanmax([fr_single, fr_low, fr_high]))

plt.figure(figsize=(2.6, 2.6))
ax = plt.gca()
pcm = ax.pcolormesh(Xedges, Yedges, fr_single, cmap=cmap, norm=norm, shading="auto", rasterized=True)

for i in range(ND):
    for j in range(NC):
        if not multi_mask[i, j]:
            continue
        x0 = Xedges[j]
        x1 = Xedges[j+1]
        y0 = Yedges[i]
        y1 = Yedges[i+1]
        draw_split_cell(ax, x0, x1, y0, y1, fr_low[i, j], fr_high[i, j], cmap, norm)

draw_coex_boundary(ax)
cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label=r'$f_r$ at ESS')
plt.xscale("log")
plt.yscale("log")
ax.set_xlim(Xedges[0], Xedges[-1])
ax.set_ylim(Yedges[0], Yedges[-1])
ax.margins(0)
ax.set_box_aspect(1)
plt.xlabel(r'Nutrient concentration ratio, $c_r/c_p$')
plt.ylabel(r'Dilution factor, D')
plt.savefig("fr_combined_heatmap.svg", bbox_inches="tight")
plt.show()

# phi_R0max combined heatmap
phi_single = phi_mat.copy()
phi_single[multi_mask] = np.nan
phi_norm = mpl.colors.Normalize(vmin=np.nanmin([phi_single, phi_low, phi_high]), vmax=np.nanmax([phi_single, phi_low, phi_high]))

plt.figure(figsize=(2.6, 2.6))
ax = plt.gca()
pcm = ax.pcolormesh(Xedges, Yedges, phi_single, cmap=cmap, norm=phi_norm, shading="auto", rasterized=True)

for i in range(ND):
    for j in range(NC):
        if not multi_mask[i, j]:
            continue
        x0 = Xedges[j]
        x1 = Xedges[j+1]
        y0 = Yedges[i]
        y1 = Yedges[i+1]
        draw_split_cell(ax, x0, x1, y0, y1, phi_low[i, j], phi_high[i, j], cmap, phi_norm)

draw_coex_boundary(ax)
cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label=r'$\phi_R^\mathrm{0,max}$ at ESS')
plt.xscale("log")
plt.yscale("log")
ax.set_xlim(Xedges[0], Xedges[-1])
ax.set_ylim(Yedges[0], Yedges[-1])
ax.margins(0)
ax.set_box_aspect(1)
plt.xlabel(r'Nutrient concentration ratio, $c_r/c_p$')
plt.ylabel(r'Dilution factor, D')
plt.savefig("phi_R0max_combined_heatmap.svg", bbox_inches="tight")
plt.show()

phi_max = 0.5
def lambd(medium, phi_R0max, e_max, k_n, f, species):
    return (phi_max - phi_R0max[species])* ((1/e_max[species] + 1/(k_n[medium] * f[species][medium]))**(-1))
def lambda_max(phi_R0max, e_max, species):
    return (phi_max - phi_R0max[species]) * e_max[species]

# physiology scatter: max growth rate vs. rich-medium growth rate, single vs. multi-strain communities
k_n = {'r': 8, 'p': 0.2}
lambda_max_limit = 2.75
lambda_r_limit = 0.6

lambda_max_single = []
lambda_r_ratio_single = []
lambda_max_multi = []
lambda_r_ratio_multi = []

for (D, cr), df in data.items():
    for _, row in df.iterrows():
        f_r = row["f_r"]
        phi = row["phi_R0max"]
        f = [{}, {}]
        e_max = [5.5, 5.5]
        f[0]['p'] = 1
        f[1]['p'] = 1
        f[0]['r'] = f_r
        f[1]['r'] = f_r
        phi_R0max = [phi, phi]
        lam_max = lambda_max(phi_R0max, e_max, 0)
        lam_r = lambd('r', phi_R0max, e_max, k_n, f, 0)
        ratio = lam_r / lam_max
        if len(df) == 1:
            lambda_max_single.append(lam_max)
            lambda_r_ratio_single.append(ratio)
        else:
            lambda_max_multi.append(lam_max)
            lambda_r_ratio_multi.append(ratio)

plt.figure(figsize=(2, 2))
plt.scatter(lambda_max_single, lambda_r_ratio_single, color="blue", alpha=0.5)
plt.scatter(lambda_max_multi, lambda_r_ratio_multi, color="red", alpha=0.5)
plt.xlabel(r"$\lambda_\mathrm{max}$")
plt.ylabel(r"$\frac{\lambda_\mathrm{r}}{\lambda_\mathrm{max}}$", rotation=0)
plt.xlim(2.4, lambda_max_limit * 1.02)
plt.ylim(0, lambda_r_limit * 1.1)
plt.axvspan(lambda_max_limit, lambda_max_limit*1.1, color="grey", alpha=0.3)
plt.axhspan(lambda_r_limit, lambda_r_limit*1.1, color="grey", alpha=0.3)
plt.savefig("physiology.svg", format="svg", bbox_inches="tight")
plt.show()
