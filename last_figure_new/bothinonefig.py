import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import glob

# use Arial for all plot text, keep it editable in PDF export
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"

df = pd.read_csv('fig1a_data.csv')

media1 = ['M9 Glucose', 'M9 Xylose']
labels1 = ['Glucose', 'Xylose']

exclude = ['MG1655', 'S5Vir']
df1 = df[~df['Strain'].isin(exclude)]

sub1 = df1[df1['Medium'].isin(media1)]
strain_counts1 = sub1.groupby('Strain')['Medium'].nunique()
common_strains1 = strain_counts1[strain_counts1 == 2].index

plot_df1 = df1[(df1['Medium'].isin(media1)) & (df1['Strain'].isin(common_strains1))].copy()

# highlight logic
glu1 = plot_df1[plot_df1['Medium'] == media1[0]][['Strain', 'Growth rate (h-1)']]
sorted1 = glu1.sort_values('Growth rate (h-1)')
lowest_strain_1 = sorted1.iloc[0]['Strain']
second_highest_strain_1 = sorted1.iloc[-4]['Strain'] if len(sorted1) > 1 else sorted1.iloc[-1]['Strain']
highlight_sp1 = {lowest_strain_1, second_highest_strain_1}

phi_max = 0.5

def lambd(medium, phi_R0max, e_max, k_n, f):
    return (phi_max - phi_R0max) * ((1/e_max + 1/(k_n[medium] * f[medium]))**(-1))

files = glob.glob("trait_trajectory_*.csv")
species_list = []

for file in files:
    df_tmp = pd.read_csv(file)
    if df_tmp.empty:
        continue

    last = df_tmp.iloc[-1]
    phi = last.get("phi_1", np.nan)
    r1  = last.get("r1_1", np.nan)
    r2  = last.get("r2_1", np.nan)
    p1  = last.get("p1_1", np.nan)
    p2  = last.get("p2_1", np.nan)

    if pd.notna(phi):
        file_index = file.split("trait_trajectory_")[1].split(".csv")[0]
        species_list.append([phi, r1, r2, p1, p2, file_index])

e_max = 5.5
k_n = {'r1': 5.3, 'r2': 5.3}

growth_pairs = []

for element in species_list:
    phi_R0max = element[0]

    f1 = {'r1': element[1], 'p1': element[3]}
    g1 = lambd('r1', phi_R0max, e_max, k_n, f1)

    f2 = {'r2': element[2], 'p2': element[4]}
    g2 = lambd('r2', phi_R0max, e_max, k_n, f2)

    growth_pairs.append((g1, g2, element[5]))

plot_df2 = pd.DataFrame(growth_pairs, columns=['g1', 'g2', 'Strain'])
plot_df2['Strain'] = plot_df2.index.astype(str)

media2 = ['Resource 1', 'Resource 2']
labels2 = ['Nutrient 1', 'Nutrient 2']

# highlight species
sorted2 = plot_df2.sort_values('g1')
lowest_strain_2 = sorted2.iloc[1]['Strain']
highest_strain_2 = sorted2.iloc[-4]['Strain']
highlight_sp2 = {lowest_strain_2, highest_strain_2}

x1 = np.arange(len(media1))
gap = 0
x2 = np.arange(len(media2)) + x1[-1] + gap + 1

fig, ax1 = plt.subplots(figsize=(3,2))

for strain, group in plot_df1.groupby('Strain'):
    if strain in highlight_sp1:
        continue
    group = group.set_index('Medium').loc[media1].reset_index()
    ax1.plot(x1, group['Growth rate (h-1)'], marker='o', color='lightgrey', alpha=1)

for strain, group in plot_df1.groupby('Strain'):
    if strain not in highlight_sp1:
        continue
    group = group.set_index('Medium').loc[media1].reset_index()
    ax1.plot(x1, group['Growth rate (h-1)'], marker='o', color='#4C72B0', alpha=1, linewidth=2)

ax1.axvline(x=x1[-1] + 0.5, linestyle=':', linewidth=1)
ax1.set_ylabel(r'Growth rate (h$^{-1}$)')

ax2 = ax1.twinx()

# first pass: grey
for i, row in plot_df2.iterrows():
    if row['Strain'] in highlight_sp2:
        continue
    ax2.plot(x2, [row['g1'], row['g2']], marker='o', color='lightgrey', alpha=1)

for i, row in plot_df2.iterrows():
    if row['Strain'] not in highlight_sp2:
        continue
    ax2.plot(x2, [row['g1'], row['g2']], marker='o', color='#4C72B0', alpha=1, linewidth=2)

xticks = list(x1) + list(x2)
xticklabels = labels1 + labels2

ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels)

#plt.savefig("fig_5b.svg", format="svg", bbox_inches="tight")
plt.show()

highlight1 = sorted2.iloc[1]
highlight2 = sorted2.iloc[-4]
print(f"Highlighted strain 1: {highlight1['Strain']}, " f"growth rate on nutrient 1 = {highlight1['g1']:.3f}, " f"growth rate on nutrient 2 = {highlight1['g2']:.3f}")
print(f"Highlighted strain 2: {highlight2['Strain']}, " f"growth rate on nutrient 1 = {highlight2['g1']:.3f}, " f"growth rate on nutrient 2 = {highlight2['g2']:.3f}")
