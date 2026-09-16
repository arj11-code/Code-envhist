import numpy as np
from scipy.integrate import solve_ivp
import sys
import random
from scipy.optimize import root_scalar
import linecache
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

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

def mutate(value, sigma=0.002):
    while True:
        new = value + np.random.normal(0, sigma)
        if 0 < new < 1:
            break
    return new
# Mean consumption time of the random community
k_n = {'r':8, 'p':0.2}
c = {'r': None, 'p':1}
D_values = np.logspace(1,3,10)
c_r_values = np.logspace(-3,3,10)
num = int(sys.argv[1])
rep = num % 10          # replicate index (0–9)
tmp = num // 10
j = tmp % 10            # column
i = tmp // 10           # row
D = D_values[0]
c['r'] = 100 #c_r_values[]

species_id = [0]
parent_map = {0: -1}   # root has no parent
next_id = 1
nspecies = 1
e_max = [5.5] * nspecies
phi_R0max = [0] * nspecies
f = [{} for _ in range(nspecies)]
for k in range(nspecies):
    f[k]['p'] = 1
    f[k]['r'] = 0.9999
    phi_R0max[k] = 0.0001
initial_biomass = np.full(nspecies, 0.01/nspecies)
parent=0
nspecies += 1
phi_R0max.append(mutate(phi_R0max[parent]))
e_max.append(e_max[parent])
f.append({})
f[-1]['p'] = f[parent]['p']
f[-1]['r'] = mutate(f[parent]['r'])
# lineage tracking
species_id.append(next_id)
parent_map[next_id] = species_id[parent]
next_id += 1

mutant_size = 0.01 * initial_biomass[parent]
initial_biomass = np.append(initial_biomass[0], mutant_size)
initial_biomass[parent] *= 0.99

trajectory_data = []
counter=0
for mutation in range(1000):
    if mutation % 10 == 0:
        print(mutation)
    community_list = []
    while counter <1000000:
        if counter > 100 and np.allclose(community_list[-2], community_list[-1], atol=1e-9):
            break
        medium = 'p'
        consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
        final_biomass_poor = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val)
        final_biomass_poor[final_biomass_poor < 1e-7] = 0
        # community_list.append(final_biomass_poor)
        medium = 'r'
        initial_biomass = final_biomass_poor/D
        consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass)
        final_biomass_rich = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, initial_biomass, consumption_val)
        final_biomass_rich[final_biomass_rich < 1e-7] = 0
        community_list.append(final_biomass_rich)
        initial_biomass = final_biomass_rich/D
        counter +=1
    counter = 0
    final_state = community_list[-1]
    alive = np.where(final_state > 0)[0]
    n_alive = len(alive)
    phi_R0max = [phi_R0max[i] for i in alive]
    e_max = [e_max[i] for i in alive]
    f = [f[i] for i in alive]
    species_id = [species_id[i] for i in alive]
    final_state = final_state[alive]
    nspecies = len(alive)
    # store traits of surviving strains
    row = [mutation]
    
    # store up to 2 strains (pad with NaN if only 1 exists)
    for i in range(4):
        if i < nspecies:
            sid = species_id[i]
            pid = parent_map.get(sid, -1)
            row.extend([sid, pid, f[i]['r'], phi_R0max[i]])
        else:
            row.extend([np.nan, np.nan, np.nan, np.nan])
    trajectory_data.append(row)
    parent = np.random.choice(nspecies)
    parent_id = species_id[parent]
    nspecies += 1
    phi_R0max.append(mutate(phi_R0max[parent]))
    e_max.append(e_max[parent])
    f.append({})
    f[-1]['p'] = f[parent]['p']
    f[-1]['r'] = mutate(f[parent]['r'])
    # lineage tracking
    species_id.append(next_id)
    parent_map[next_id] = parent_id
    next_id += 1
    mutant_size = 0.01 * final_state[parent]
    initial_biomass = np.append(final_state, mutant_size)
    initial_biomass[parent] *= 0.99    


df = pd.DataFrame(trajectory_data, columns = ["mutation_step", "id_1","parent_1","f_r_1","phi_1", "id_2","parent_2","f_r_2","phi_2", "id_3","parent_3","f_r_3","phi_3", "id_4","parent_4","f_r_4","phi_4"])
df.to_csv(f"full_trajectory_{num}.csv", index=False)

















