# The previous code returned many communities that were not really at steady state: persistent coexistence which needs to be removed (sp. that are almost the same are coexisting).
# This code will clean that up, remove sp. with low abundance and rewrite the files
# Then we can easily do further analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import os
import csv
from scipy.integrate import solve_ivp
import sys
import random
import linecache
from scipy.optimize import root_scalar 
from scipy.optimize import least_squares
import itertools

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
    return time_taken
    
def consumption_function(t, phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass): #This is the function we need to find roots of
    lambdas = np.array([lambd(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    lags = np.array([lag_time(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    growth_amount = initial_biomass * (np.exp(lambdas * np.maximum(t - lags, 0)) - 1)
    return np.sum(growth_amount) - c[medium]

def consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass):
    t_lo = np.min([lag_time(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    f_lo = consumption_function(t_lo, phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
    if f_lo > 0:
        return t_lo
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


abundance_cutoff = 1e-4
i = int(sys.argv[1])

k_n = {'r': 8, 'p': 0.2}
fname = f"final_{i}.csv"
df = pd.read_csv(fname)
D  = df["Dilution"].iloc[0]
cr = df["c_r value"].iloc[0]
survivors = df[df["final_abundance"] >= abundance_cutoff]
f_r = survivors["f_r"].to_numpy()
phi = survivors["phi_R0max"].to_numpy()
N   = survivors["final_abundance"].to_numpy()
f = []
for fr in f_r:
    f.append({'r': fr, 'p': 1})
phi_R0max = phi.tolist()
e_max = [5.5] * len(f)
c = {'r': cr, 'p': 1}
nspecies = len(N)
initial_biomass = N
community_list = []
community_list.append(initial_biomass)
counter=0
while counter <1000000:
    if counter > 0 and counter % 1000 == 0:
        print(counter)
        print(community_list[-1])
    if counter > 100 and np.all(community_list[-2] == community_list[-1]):
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

final_state = community_list[-1]
rows = []
for k in range(len(final_state)):
    if final_state[k] > abundance_cutoff:
        rows.append([D, cr, f_r[k], phi_R0max[k], final_state[k]])
outname = f"cleaned_result_{i}.csv"
with open(outname, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["Dilution", "c_r value", "f_r", "phi_R0max", "final_abundance"])
    writer.writerows(rows)



































