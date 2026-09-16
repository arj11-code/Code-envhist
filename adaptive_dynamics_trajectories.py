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

def invasion_tester(phi_R0max, e_max, k_n, f):
    quantity = lambd('p', phi_R0max, e_max, k_n, f, 1) * max(0,(depletion_time('p', phi_R0max, e_max, k_n, f, 0) - lag_time('p', phi_R0max, e_max, k_n, f, 1))) + lambd('r', phi_R0max, e_max, k_n, f, 1) * max(0, (depletion_time('r', phi_R0max, e_max, k_n, f, 0) - lag_time('r', phi_R0max, e_max, k_n, f, 1)))
    return quantity - (2 * np.log(D))

# Mean consumption time of the random community
k_n = {'r':8, 'p':0.2}
c = {'r': 100, 'p':1}
D = 10
index = 0 #int(sys.argv[1])

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

# Pure adaptive dynamics loop
step_size = 1e-3
n_directions = 64
fitness_tol = 1e-10

mutation_counter = 0
trait_trajectory = []

# initial resident
f_r = f[0]['r']
phi = phi_R0max[0]
trait_trajectory.append([f_r, phi])

while mutation_counter < 5000:
    best_fitness = -np.inf
    best_trait = None
    angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    for th in angles:
        f_try = f_r + step_size * np.cos(th)
        phi_try = phi + step_size * np.sin(th)
        # trait bounds
        if not (0 <= f_try <= 1 and 0 <= phi_try <= 0.2):
            continue
        # resident + mutant
        f_tmp = [{'p': 1, 'r': f_r}, {'p': 1, 'r': f_try}]
        phi_tmp = [phi, phi_try]
        e_tmp = [5.5, 5.5]
        # invasion fitness of mutant (index 1)
        fitness = invasion_tester(phi_tmp, e_tmp, k_n, f_tmp)
        if fitness > best_fitness:
            best_fitness = fitness
            best_trait = (f_try, phi_try)
    # stop at evolutionary fixed point
    if best_fitness < fitness_tol:
        break
    f_r, phi = best_trait
    trait_trajectory.append([f_r, phi])
    mutation_counter += 1

# Save f_r and phi_R0max into a csv file for plotting the phase space trajectories
df_traits = pd.DataFrame(trait_trajectory, columns=["f_r", "phi_R0max"])
df_traits.to_csv("100_case.csv", index=False)
#df_traits.to_csv(f"adaptive_dynamics_trajectory_{D}_{c['r']}.csv", index=False)
