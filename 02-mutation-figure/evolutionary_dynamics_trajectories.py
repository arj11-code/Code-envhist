import csv
import numpy as np
from scipy.integrate import solve_ivp
import sys
import random
from scipy.optimize import root_scalar
import linecache
from matplotlib.ticker import FormatStrFormatter
import os
import matplotlib.pyplot as plt
import pandas as pd

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
    # Lower bound to search is minimum lag time
    t_lo = np.min([lag_time(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    f_lo = consumption_function(t_lo, phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)

    if f_lo > 0:
        # consumption already exceeded at lag (just in case)
        return t_lo

    # Adaptive upper bound
    t_hi = t_lo + 10.0
    for _ in range(600):
        f_hi = consumption_function(t_hi, phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
        if f_hi > 0:
            break
        t_hi *= 2

    else:
        return None

    sol = root_scalar(consumption_function, args=(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass), bracket=[t_lo, t_hi], method="brentq")

    return sol.root if sol.converged else None

def growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val):
    lambdas = np.array([lambd(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    lags = np.array([lag_time(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    final_biomass = initial_biomass * np.exp(lambdas * (np.maximum(consumption_val - lags,0)))
    return final_biomass #Note that the output of this function is NOT diluted

def invasion_tester(phi_R0max, e_max, k_n, f, poor_cons, rich_cons): #Change the number at end for species index. RN it is zero because I am getting consumption time from random community
    quantity = lambd('p', phi_R0max, e_max, k_n, f, 2) * (poor_cons - lag_time('p', phi_R0max, e_max, k_n, f, 2)) + lambd('r', phi_R0max, e_max, k_n, f, 2) * (rich_cons - lag_time('r', phi_R0max, e_max, k_n, f, 2))
    return quantity - (2 * np.log(D))

# Mean consumption time of the random community
k_n = {'r':8, 'p':0.2}
c = {'r': 1, 'p':1}
D = 10
index = int(sys.argv[1])

# Creating lists that will store all the data
abundance_history = []
f_r_history = []
phi_history = []
time_history = []
lineage_history = []
trait_trajectory = []

# Initializing the parameters which record what species is what
t_global = 0
next_lineage_id = 0
current_lineage_ids = [0]

# Start off with a randomly generated species who reaches steady state
nspecies = 1

f_r = 0.9999
phi = 0.0001

f = [{'p': 1, 'r': f_r}]
phi_R0max = [phi]
e_max = [5.5]

initial_biomass = np.array([0.1])
trait_trajectory.append([f_r, phi])
mutation_counter = 0

#Start evolving
while mutation_counter < 5000:
    if mutation_counter % 10 == 0:
        print(mutation_counter)
    counter = 0
    skip_record = False
    # Ecological dynamics
    biomasses_store_compare = []
    while counter < 100000: #each iteration of this is a rich-poor cycle
        if counter > 1 and abs(biomasses_store_compare[-2][0] - initial_biomass[0]) < 1e-9: # checking for steady state
            break
        
        # poor
        medium = 'p'
        consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
        final_biomass_poor = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val)

        # rich
        medium = 'r'
        initial_biomass = final_biomass_poor / D
        consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
        final_biomass_rich = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val)

        # We do skip_record for the first step because of some numerical issue
        if not skip_record: # This code records so that we plot trajectories
            for i in range(nspecies):
                abundance_history.append(final_biomass_rich[i])
                f_r_history.append(f[i]['r'])
                phi_history.append(phi_R0max[i])
                lineage_history.append(current_lineage_ids[i])
                time_history.append(t_global)
            t_global += 1
        else:
            skip_record = False

        initial_biomass = final_biomass_rich/D
        biomasses_store_compare.append(initial_biomass)
        counter += 1
    
    # Mutate the two parameters
    # We want to allow both to mutate, or only one to mutate. So we sample until atleast one has mutated
    while True:
        df = np.random.normal(0, 0.005)
        dphi = np.random.normal(0, 0.0025)
        if np.random.rand() < 0.5:
            df = 0.0
        if np.random.rand() < 0.5:
            dphi = 0.0
    
        if df != 0 or dphi != 0:
            break

    f_r_mut = np.clip(f[0]['r'] + df, 0, 1)
    phi_mut = np.clip(phi_R0max[0] + dphi, 0, 0.2)

    current_lineage_ids = [current_lineage_ids[0], next_lineage_id + 1]
    next_lineage_id += 1

    f = [{'p': 1, 'r': f[0]['r']}, {'p': 1, 'r': f_r_mut}]
    phi_R0max = [phi_R0max[0], phi_mut]
    e_max = [5.5]*2
    nspecies = 2

    initial_biomass = np.array([final_biomass_rich[0], 1e-6])

    skip_record = True   # Skips first mutant cycle (avoiding numerical error)

    #This stuff is competition between mutant and parent
    counter = 0
    biomasses_store_compare = []
    ratio_history = []
    while counter < 100000:
        if counter > 1 and abs(biomasses_store_compare[-2][0] - initial_biomass[0]) < 1e-9: # checking for steady state
            break

        # poor
        medium = 'p'
        consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
        final_biomass_poor = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val)

        # rich
        medium = 'r'
        initial_biomass = final_biomass_poor/D
        consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
        final_biomass_rich = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val)

        # Record the dynamics
        if not skip_record:
            for i in range(nspecies):
                abundance_history.append(final_biomass_rich[i])
                f_r_history.append(f[i]['r'])
                phi_history.append(phi_R0max[i])
                lineage_history.append(current_lineage_ids[i])
                time_history.append(t_global)

            t_global += 1
        else:
            skip_record = False

        initial_biomass = final_biomass_rich/D
        biomasses_store_compare.append(initial_biomass)
        R_final = final_biomass_rich[0]
        M_final = final_biomass_rich[1]
        if R_final > 0:
            ratio_history.append(M_final/R_final)
        counter += 1

    # Decide winner
    ratio_start = ratio_history[0]
    ratio_end = ratio_history[-1]
    log_ratio_change = np.log(ratio_end) - np.log(ratio_start)
    winner = 1 if log_ratio_change > 0 else 0

    f = [f[winner]]
    phi_R0max = [phi_R0max[winner]]
    current_lineage_ids = [current_lineage_ids[winner]]

    nspecies = 1
    e_max = [5.5]
    initial_biomass = np.array([0.1])
    
    trait_trajectory.append([f[0]['r'], phi_R0max[0]])
    mutation_counter += 1

# Save f_r and phi_R0max into a csv file for plotting the phase space trajectories
df_traits = pd.DataFrame(trait_trajectory, columns=["f_r", "phi_R0max"])
df_traits.to_csv(f"trait_trajectory_{index}.csv", index=False)



#Plotting
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10
df_dyn = pd.DataFrame({"time": time_history, "abundance": abundance_history, "f_r": f_r_history, "phi_R0max": phi_history, "lineage": lineage_history})
# We want to stop plotting after the last invasion
last_invasion_idx = np.where(df_dyn["lineage"].values[1:] != df_dyn["lineage"].values[:-1])[0].max() + 1
buffer = int(0.05 * len(df_dyn))  # Another 5% buffer to show it is uninvasible
stop_idx = min(last_invasion_idx + buffer, len(df_dyn))

df_plot = df_dyn.iloc[:stop_idx]

fig, ax = plt.subplots(figsize=(6,2))
norm = plt.Normalize(df_plot["f_r"].min(), df_plot["f_r"].max())
cmap = plt.cm.viridis
for lin in df_plot["lineage"].unique():
    sub = df_plot[df_plot["lineage"] == lin]
    ax.plot(sub["time"], sub["abundance"], color=cmap(norm(sub["f_r"].iloc[-1])), linewidth=2.5)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r"$f_r$ of strain", format=FormatStrFormatter('%.2f'))
ax.set_xlabel("Number of cycles")
ax.set_ylabel("Abundance")
fig.savefig(f"f_r_dynamics_{index}.svg", format="svg", bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6,2))
norm = plt.Normalize(df_plot["phi_R0max"].min(), df_plot["phi_R0max"].max())
cmap = plt.cm.viridis
for lin in df_plot["lineage"].unique():
    sub = df_plot[df_plot["lineage"] == lin]
    ax.plot(sub["time"], sub["abundance"], color=cmap(norm(sub["phi_R0max"].iloc[-1])), linewidth=2.5)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r"$\phi_R^{\mathrm{0,max}}$ of strain", format=FormatStrFormatter('%.2f'))
ax.set_xlabel("Number of cycles")
ax.set_ylabel("Abundance")
fig.savefig(f"phi_R0max_dynamics_{index}.svg", format="svg", bbox_inches="tight")
plt.close()










