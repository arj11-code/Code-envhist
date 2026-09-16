import numpy as np
from scipy.integrate import solve_ivp
import sys
import random
from scipy.optimize import root_scalar
import linecache
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.cm as cm
import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10

phi_max = 0.5

def lambd(medium, phi_R0max, e_max, k_n, f, species):
    return (phi_max - phi_R0max[species])* ((1/e_max[species] + 1/(k_n[medium] * f[species][medium]))**(-1))
def phi_P(medium, phi_R0max, e_max, k_n, f, species):
    return (phi_max - phi_R0max[species]) * (e_max[species]  / (e_max[species] + k_n[medium] * f[species][medium]))
def phi_R(medium, phi_R0max, e_max, k_n, f, species):
    return (e_max[species] * phi_R0max[species] + k_n[medium] * f[species][medium] * phi_max)/(e_max[species] + k_n[medium] * f[species][medium])

def lambd_0(medium, phi_R0max, e_max, k_n, f, species):
    if medium == 'r':
        return e_max[species] * (phi_R('r', phi_R0max, e_max, k_n, f, species)/(phi_R('r', phi_R0max, e_max, k_n, f, species) + phi_R0max[species])) * phi_R('p', phi_R0max, e_max, k_n, f, species) * (1 - ((phi_R0max[species]**2) /(phi_R('r', phi_R0max, e_max, k_n, f, species)**2)))
    if medium == 'p':
        return (1 - f[species]['r']) * phi_P('r', phi_R0max, e_max, k_n, f, species) * k_n['p']
def lag_time(medium, phi_R0max, e_max, k_n, f, species):
    T_lag = (1/lambd(medium, phi_R0max, e_max, k_n, f, species)) * np.log(lambd(medium, phi_R0max, e_max, k_n, f, species)/lambd_0(medium, phi_R0max, e_max, k_n, f, species))
    return max(0, T_lag)

def steady_state_mass(medium, c):
    if medium == 'r':
        return (c['r'] + D * c['p'])/(D**2 - 1)
    if medium == 'p':
        return (c['p'] + D * c['r'])/(D**2 - 1)
def depletion_time(medium, phi_R0max, e_max, k_n, f, species):
    if medium == 'r':
        time_taken = lag_time('r', phi_R0max, e_max, k_n, f, species) + (1/lambd('r', phi_R0max, e_max, k_n, f, species)) * np.log((c['r'] + steady_state_mass('r',c))/steady_state_mass('r', c))
    if medium == 'p':
        time_taken = lag_time('p', phi_R0max, e_max, k_n, f, species) + (1/lambd('p', phi_R0max, e_max, k_n, f, species)) * np.log((c['p'] + steady_state_mass('p',c))/steady_state_mass('p', c))
    return time_taken #min(time_taken,500)

def consumption_function(t, phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass): #This is the function we need to find roots of
    lambdas = np.array([lambd(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    lags = np.array([lag_time(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    growth_amount = initial_biomass * (np.exp(lambdas * np.maximum(t - lags, 0)) - 1)
    return np.sum(growth_amount) - c[medium]

def consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass):
    # lower bound: no growth before min lag
    t_lo = np.min([lag_time(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    # value at lower bound
    f_lo = consumption_function(t_lo, phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
    if f_lo > 0:
        # consumption already exceeded at lag
        return t_lo
    # adaptive upper bound
    t_hi = t_lo + 10.0
    for _ in range(600):
        f_hi = consumption_function(t_hi, phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
        if f_hi > 0:
            break
        t_hi *= 2
    else:
        # never crossed zero → no solution
        return None
    sol = root_scalar(consumption_function, args=(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass), bracket=[t_lo, t_hi], method="brentq")
    return sol.root if sol.converged else None

def growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val):
    lambdas = np.array([lambd(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    lags = np.array([lag_time(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    final_biomass = initial_biomass * np.exp(lambdas * (np.maximum(consumption_val - lags,0)))
    return final_biomass #Note that the output of this function is NOT diluted

# Mean consumption time of the random community
k_n = {'r':8, 'p':0.2}
c = {'r': None, 'p':1}
D_values = np.logspace(1,3,10)
c_r_values = np.logspace(-3,3,10)
num = 1 #int(sys.argv[1])
i = num // 10
j = num % 10
D = 10 # D_values[i]
c['r'] = 1 #c_r_values[j]

nspecies = 900
nvals = 30
e_max = [5.5] * nspecies
phi_R0max = [0] * nspecies
f = [{} for _ in range(nspecies)]
f_rlist = np.linspace(1e-2,(1-1e-2),nvals)
phi_R0max_list = np.linspace(1e-3, 0.1-1e-3, nvals)
fgrid, phigrid = np.meshgrid(f_rlist, phi_R0max_list, indexing="ij")
grid_points = np.column_stack((fgrid.ravel(), phigrid.ravel()))
for k in range(nspecies):
    f_r, phi_val = grid_points[k]
    f[k]['p'] = 1
    f[k]['r'] = f_r
    phi_R0max[k] = phi_val

initial_biomass = np.full(nspecies, 0.01/nspecies)
community_list = []
community_list.append(initial_biomass)
counter=0
while counter <100000:
    if counter % 100 == 0:
        print(counter)
    if counter > 100 and np.allclose(community_list[-2], community_list[-1], atol=1e-10):
        break
    medium = 'p'
    consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
    final_biomass_poor = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val)
    final_biomass_poor[final_biomass_poor < 1e-8] = 0
    # community_list.append(final_biomass_poor)
    medium = 'r'
    initial_biomass = final_biomass_poor/D
    consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
    final_biomass_rich = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val)
    final_biomass_rich[final_biomass_rich < 1e-8] = 0
    community_list.append(final_biomass_rich)
    initial_biomass = final_biomass_rich/D
    counter +=1

final_community = community_list[-1]
mask = final_community != 0

# write to CSV
with open("trajectory_1.csv", "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["time", "Dilution", "c_r value", "species", "f_r", "phi_R0max", "abundance"])
    
    for t, community in enumerate(community_list):
        for i in range(nspecies):
            if community[i] != 0:  # optional: keep only surviving species
                writer.writerow([t, D, c['r'], i, f[i]['r'], phi_R0max[i], community[i]])

'''
#Plotting
cmap = plt.get_cmap("Blues")
fr_values = np.array([f[i]['r'] for i in range(nspecies)])
norm = plt.Normalize(fr_values.min(), fr_values.max())
colors = cmap(norm(fr_values))
fig, ax = plt.subplots(figsize=(5,1.75))
x = np.arange(1, len(community_list) + 1)
eps = 1e-6
for k in range(nspecies):
    y = np.array([comm[k] for comm in community_list], dtype=float)
    y = np.clip(y, eps, None)
    ax.plot(x, y, color=colors[k])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="value")
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.2))
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(bottom=eps)
ax.set_xlabel("Number of cycles")
ax.set_ylabel("Abundance")
plt.tight_layout()
plt.savefig("community_root_fr_1.svg", format="svg", bbox_inches="tight")
plt.show()

cmap = plt.get_cmap("Blues")
phi_values = np.array(phi_R0max)
norm = plt.Normalize(phi_values.min(), phi_values.max())
colors = cmap(norm(phi_values))
fig, ax = plt.subplots(figsize=(5,1.75))
x = np.arange(1, len(community_list) + 1)
eps = 1e-6
for k in range(nspecies):
    y = np.array([comm[k] for comm in community_list], dtype=float)
    y = np.clip(y, eps, None)
    ax.plot(x, y, color=colors[k])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="value")
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.03))
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(bottom=eps)
ax.set_xlabel("Number of cycles")
ax.set_ylabel("Abundance")
plt.tight_layout()
plt.savefig("community_root_phi_1.svg", format="svg", bbox_inches="tight")
plt.show()
'''
















