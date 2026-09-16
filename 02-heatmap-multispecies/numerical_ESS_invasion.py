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

def invasion_tester(phi_R0max, e_max, k_n, f):
    quantity = lambd('p', phi_R0max, e_max, k_n, f, 1) * max(0,(depletion_time('p', phi_R0max, e_max, k_n, f, 0) - lag_time('p', phi_R0max, e_max, k_n, f, 1))) + lambd('r', phi_R0max, e_max, k_n, f, 1) * max(0, (depletion_time('r', phi_R0max, e_max, k_n, f, 0) - lag_time('r', phi_R0max, e_max, k_n, f, 1)))
    return quantity - (2 * np.log(D))

#Root finding to get the ESS directly
def selection_gradient(resident, e_max, k_n, f): #This is a numerical approximation of the selection gradient. When this is zero, we check for ESS there. 
    #Here we are taking the partial derivative with respect to just f_r first. Then we will take partial derivative with respect to phi_R0max. 
    phi_R0max_local = [resident[1]]*2
    h = 1e-5
    f[0]['r'] = resident[0]
    f[1]['r'] = resident[0] + h
    s_plus = invasion_tester(phi_R0max_local, e_max, k_n, f)
    f[1]['r'] = resident[0] - h
    s_minus = invasion_tester(phi_R0max_local, e_max, k_n, f)
    f_rgrad = (s_plus - s_minus)/(2*h)
    f[1]['r'] = resident[0] #Reset this so we can do phi_R0max gradient now
    h = 1e-5
    phi_R0max_local[1] = resident[1] + h
    s_plus = invasion_tester(phi_R0max_local, e_max, k_n, f)
    phi_R0max_local[1] = resident[1] - h
    s_minus = invasion_tester(phi_R0max_local, e_max, k_n, f)
    phi_R0maxgrad = (s_plus - s_minus)/(2*h)
    return np.array([f_rgrad, phi_R0maxgrad])

def find_ESS(phi_R0max, e_max, k_n, f, f_bounds=(0,1), phi_R0max_bounds=(0,0.4)):
    x0 = [np.mean(f_bounds), np.mean(phi_R0max_bounds)]
    sol = least_squares(selection_gradient, x0=x0, bounds=([f_bounds[0], phi_R0max_bounds[0]],[f_bounds[1], phi_R0max_bounds[1]]), args=(e_max, k_n, f),ftol=1e-12, xtol=1e-12, max_nfev=5000)
    r, phi = sol.x
    grad_norm = np.linalg.norm(sol.fun)
    if sol.success and grad_norm < 1e-6:
        return np.round([r, phi], 6)
    
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

def invasion_testore(phi_R0max, e_max, k_n, f, poor_cons, rich_cons): #Change the number at end for species index. RN it is zero because I am getting consumption time from random community
    quantity = lambd('p', phi_R0max, e_max, k_n, f, 2) * (poor_cons - lag_time('p', phi_R0max, e_max, k_n, f, 2)) + lambd('r', phi_R0max, e_max, k_n, f, 2) * (rich_cons - lag_time('r', phi_R0max, e_max, k_n, f, 2))
    return quantity - (2 * np.log(D))

abundance_cutoff = 1e-4

for i in range(1):
    i = 62
    data = {}
    fname = f"cleaned_result_{i}.csv"
    df = pd.read_csv(fname)
    D  = df["Dilution"].iloc[0]
    cr = df["c_r value"].iloc[0]
    survivors = df[df["final_abundance"] >= abundance_cutoff]
    data.setdefault((D, cr), []).append(survivors)
    
    ESS_results = {}
    k_n = {'r': 8, 'p': 0.2}
    for (D, cr), dfs in data.items():
        c = {'r': cr, 'p': 1}
        f = [{},{}]
        phi_R0max = [None]*2
        e_max = [5.5]*2
        f[0]['p'] = 1
        f[1]['p'] = 1
        ESS = find_ESS(phi_R0max,e_max,k_n,f,f_bounds=(0.001, 0.999),phi_R0max_bounds=(0, 0.4))
        if ESS is None:
            ESS = [-1,-1]
        f_r_ESS = ESS[0]
        phi_R0max_ESS = ESS[1]
        ESS_results[(D, cr)] = {"f_r_ESS": f_r_ESS,"phi_R0max_ESS": phi_R0max_ESS}
    
    invader_fraction = 0.1
    results_invasion = {}
    
    df = pd.read_csv(fname)
    D  = df["Dilution"].iloc[0]
    cr = df["c_r value"].iloc[0]
    survivors = df[df["final_abundance"] >= abundance_cutoff]
    f_r_res = survivors["f_r"].to_numpy()
    phi_res = survivors["phi_R0max"].to_numpy()
    N_res   = survivors["final_abundance"].to_numpy()
    ESS = ESS_results[(D, cr)]
    if ESS["f_r_ESS"] > 0:
        f_r_inv  = ESS["f_r_ESS"]
        phi_inv  = ESS["phi_R0max_ESS"]
        N_inv = invader_fraction * np.min(N_res)
        # append invader
        f_r_all  = np.append(f_r_res, f_r_inv)
        phi_all  = np.append(phi_res, phi_inv)
        N0_all   = np.append(N_res, N_inv)
        species_type = ["resident"] * len(f_r_res) + ["ESS_invader"]
        f = []
        for fr in f_r_all:
            f.append({'r': fr, 'p': 1})
        
        phi_R0max = phi_all.tolist()
        e_max = [5.5] * len(f)
        c = {'r': cr, 'p': 1}
        nspecies = len(N0_all)
        initial_biomass = N0_all
        community_list = []
        community_list.append(initial_biomass)
        counter=0
        while counter <500000:
            if counter > 0 and counter % 1000 == 0:
                print(i, counter)
                print(community_list[-1])
            if counter > 100 and np.allclose(community_list[-2], community_list[-1], atol=1e-13):
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
            rows.append([D, cr, f_r_all[k], phi_all[k], final_state[k], species_type[k]])
        outname = f"invasion_result_{i}.csv" 
        with open(outname, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["Dilution", "c_r value", "f_r", "phi_R0max", "final_abundance", "species_type"])
            writer.writerows(rows)



































