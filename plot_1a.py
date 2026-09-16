import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10

# Ensure editable text in PDF
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"

df = pd.read_csv('fig1a_data.csv')

# First species
media1 = ['M9 Glucose', 'M9 Xylose']
labels1 = ['Glucose', 'Xylose']

# MG1655 and S5Vir aren't part of this comparison, drop them
exclude = ['MG1655', 'S5Vir']
df1 = df[~df['Strain'].isin(exclude)]

# Only keep strains that have data in both media, otherwise the lines look broken
sub1 = df1[df1['Medium'].isin(media1)]
strain_counts1 = sub1.groupby('Strain')['Medium'].nunique()
common_strains1 = strain_counts1[strain_counts1 == 2].index
plot_df1 = df1[(df1['Medium'].isin(media1)) & (df1['Strain'].isin(common_strains1))].copy()

# Highlight the slowest strain and the 4th-fastest strain on glucose
glu1 = plot_df1[plot_df1['Medium'] == media1[0]][['Strain', 'Growth rate (h-1)']]
sorted1 = glu1.sort_values('Growth rate (h-1)')
lowest_strain_1 = sorted1.iloc[0]['Strain']
second_highest_strain_1 = sorted1.iloc[-4]['Strain'] if len(sorted1) > 1 else sorted1.iloc[-1]['Strain']
highlight_sp1 = {lowest_strain_1, second_highest_strain_1}

# Second species
species2 = df['Species'].unique()[-1]
df2 = df[df['Species'] == species2]

media2 = df2['Medium'].unique()[:2]
labels2 = ['Glucose', 'Pyruvate']

sub2 = df2[df2['Medium'].isin(media2)]
strain_counts2 = sub2.groupby('Strain')['Medium'].nunique()
common_strains2 = strain_counts2[strain_counts2 == 2].index
plot_df2 = df2[(df2['Medium'].isin(media2)) & (df2['Strain'].isin(common_strains2))].copy()

# Highlight the slowest and fastest strain on glucose
glu2 = plot_df2[plot_df2['Medium'] == media2[0]][['Strain', 'Growth rate (h-1)']]
sorted2 = glu2.sort_values('Growth rate (h-1)')
lowest_strain_2 = sorted2.iloc[0]['Strain']
highest_strain_2 = sorted2.iloc[-1]['Strain']
highlight_sp2 = {lowest_strain_2, highest_strain_2}

# X positions, species 2 sits right after species 1 with a gap of 1
x1 = np.arange(len(media1))
gap = 0
x2 = np.arange(len(media2)) + x1[-1] + gap + 1

# Plot
fig, ax1 = plt.subplots(figsize=(3, 2))

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

for strain, group in plot_df2.groupby('Strain'):
    if strain in highlight_sp2:
        continue
    group = group.set_index('Medium').loc[media2].reset_index()
    ax2.plot(x2, group['Growth rate (h-1)'], marker='o', color='lightgrey', alpha=1)

for strain, group in plot_df2.groupby('Strain'):
    if strain not in highlight_sp2:
        continue
    group = group.set_index('Medium').loc[media2].reset_index()
    ax2.plot(x2, group['Growth rate (h-1)'], marker='o', color='#C44E52', alpha=1, linewidth=2)

# Axes
xticks = list(x1) + list(x2)
xticklabels = labels1 + labels2
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels)

plt.savefig("fig_1_a.svg", format="svg", bbox_inches="tight")
plt.show()

# Report peak growth rates for each species
max1 = plot_df1['Growth rate (h-1)'].max()
max2 = plot_df2['Growth rate (h-1)'].max()

print(f"Species 1 max growth rate: {max1:.3f} h^-1")
print(f"Species 2 max growth rate: {max2:.3f} h^-1")
