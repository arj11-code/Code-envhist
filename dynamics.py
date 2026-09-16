import csv
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
import traceback
import sys
import matplotlib.patches as patches
import linecache
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10
# Global variables
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['lines.solid_capstyle'] = 'round'
phi_max = 0.5

#####
# Writing it out as dynamical stuff because we can easily generalize to multiple species
def lambd(medium, phi_R0max, e_max, k_n, f):
    return (phi_max - phi_R0max)* ((1/e_max + 1/(k_n[medium] * f[medium]))**(-1))
def phi_P(medium, phi_R0max, e_max, k_n, f):
    return (phi_max - phi_R0max) * (e_max  / (e_max + k_n[medium] * f[medium]))
def phi_R(medium, phi_R0max, e_max, k_n, f):
    return (e_max * phi_R0max + k_n[medium] * f[medium] * phi_max)/(e_max + k_n[medium] * f[medium])

def lambd_0(medium, phi_R0max, e_max, k_n, f):
    if medium == 'r':
        return e_max * (phi_R('r', phi_R0max, e_max, k_n, f)/(phi_R('r', phi_R0max, e_max, k_n, f) + phi_R0max)) * phi_R('p', phi_R0max, e_max, k_n, f) * (1 - ((phi_R0max**2) /(phi_R('r', phi_R0max, e_max, k_n, f)**2)))
    if medium == 'p':
        return (1 - f['r']) * phi_P('r', phi_R0max, e_max, k_n, f) * k_n['p']
def lag_time(medium, phi_R0max, e_max, k_n, f):
    T_lag = (1/lambd(medium, phi_R0max, e_max, k_n, f)) * np.log(lambd(medium, phi_R0max, e_max, k_n, f)/lambd_0(medium, phi_R0max, e_max, k_n, f))
    return max(0, T_lag)
def steady_state_mass(medium, c):
    if medium == 'r':
        return (c['r'] + D * c['p'])/(D**2 - 1)
    if medium == 'p':
        return (c['p'] + D * c['r'])/(D**2 - 1)
def depletion_time(medium, phi_R0max, e_max, k_n, f):
    if medium == 'r':
        return lag_time('r', phi_R0max, e_max, k_n, f) + (1/lambd('r', phi_R0max, e_max, k_n, f)) * np.log((c['r'] + steady_state_mass('r',c))/steady_state_mass('r', c))
    if medium == 'p':
        return lag_time('p', phi_R0max, e_max, k_n, f) + (1/lambd('p', phi_R0max, e_max, k_n, f)) * np.log((c['p'] + steady_state_mass('p',c))/steady_state_mass('p', c))
def frac_inact(medium, phi_R0max, e_max, k_n, f):
    return (1 - ((phi_R0max**2)/(phi_R(medium, phi_R0max, e_max, k_n, f)**2)))

'''
e_max = 5.5
phi_R0max = 0.007
f = {'r': 0.534, 'p': 1}
c = {'r': 1, 'p': 1}
k_n = {'r': 8, 'p': 0.2}
print("Strain 1: Unequal resources")
print("Growth rate on resource 1")
print(lambd('r', phi_R0max, e_max, k_n, f))
f = {'r': 0.666, 'p': 1}
print("Growth rate on resource 2")
print(lambd('r', phi_R0max, e_max, k_n, f))
print("Strain 2: Equal resources")
f = {'r': 0.621, 'p': 1}
print("Growth rate on resource 1")
print(lambd('r', phi_R0max, e_max, k_n, f))
f = {'r': 0.622, 'p': 1}
print("Growth rate on resource 2")
print(lambd('r', phi_R0max, e_max, k_n, f))
'''

# Inactive fraction of ribosomes
'''
e_max = 5.5
k_nlist = np.linspace(0,10, 1000)
f = {'r': 1, 'p': 1}
c = {'r': 1, 'p': 1}
k_n = {'r': 8, 'p': 0.2}
growthrate1 = []
lamdlist1 = []
lamdlist2 = []
growthrate2 = []
phi_R0max_values = [1e-3, 0.05, 0.1, 0.2]
colors = plt.cm.Reds(np.linspace(0.4, 1, len(phi_R0max_values)))

plt.figure(figsize=(3, 3))

for phi_R0max, color in zip(phi_R0max_values, colors):
    lamdlist = []
    growthrate = []
    for k_nval in k_nlist:
        k_n['r'] = k_nval
        lamda = lambd('r', phi_R0max, e_max, k_n, f)
        gr = frac_inact('r', phi_R0max, e_max, k_n, f)
        lamdlist.append(lamda)
        growthrate.append(1 - gr)
    plt.plot(lamdlist, growthrate, color=color, label=rf"$\phi_R^{{0,\mathrm{{max}}}}$ = {phi_R0max}")

plt.xlabel(r"Growth rate (h$^{-1}$)")
plt.ylabel("Inactive ribosome fraction, $\phi_\mathrm{R}^0$ ")
plt.legend()
plt.tight_layout()
plt.savefig("inact_fraction.pdf", format='pdf')
plt.show()
'''
# Growth lag tradeoff with inactive ribosomes
'''
e_max = 5.5
phi_R0max = 0.01
phi_R0maxlist = np.linspace(0,0.1, 1000)
f = {'r': 0.75, 'p': 1}
k_n = {'r': 8, 'p': 0.2}
growthrate1 = []
lamdlist1 = []
lamdlist2 = []
growthrate2 = []
for phi_R0max in phi_R0maxlist:
    lamda = lambd('p', phi_R0max, e_max, k_n, f)
    gr2 = lag_time('r', phi_R0max, e_max, k_n, f)
    lamdlist1.append(lamda)
    growthrate2.append(gr2)
    
plt.figure(figsize=(1.75,1.75))
plt.plot(lamdlist1, growthrate2)
plt.xlabel(r"Growth rate (poor) (h$^{-1}$)")
plt.ylabel(r"Upshift lag time (h)")
#plt.savefig("Growth-lag_ribosomes.svg", format="svg", bbox_inches="tight")
plt.show()
'''
# Growth lag tradeoff with inactive enzymes
'''
e_max = 5.5
phi_R0max = 0.01
phi_R0maxlist = np.linspace(0.05,0.95, 1000)
f = {'r': 0.75, 'p': 1}
k_n = {'r': 8, 'p': 0.2}
growthrate1 = []
lamdlist1 = []
lamdlist2 = []
growthrate2 = []
for f_r in phi_R0maxlist:
    f['r'] = f_r
    lamda = lambd('r', phi_R0max, e_max, k_n, f)
    gr2 = lag_time('p', phi_R0max, e_max, k_n, f)
    lamdlist1.append(lamda)
    growthrate2.append(gr2)
    
plt.figure(figsize=(1.75,1.75))
plt.plot(lamdlist1, growthrate2)
plt.xlabel(r"Growth rate (rich) (h$^{-1}$)")
plt.ylabel(r"Downshift lag time (h)")
#plt.savefig("Growth-lag_enzymes.svg", format="svg", bbox_inches="tight")
plt.show()
'''
# Inactive ribosome fraction vs growth rate
'''
e_max = 5.5
phi_R0maxlist = np.linspace(0,1, 1000)
f = {'r': 0.75, 'p': 1}
c = {'r': 1, 'p': 1}
k_n = {'r': 8, 'p': 2}
growthrate1 = []
lamdlist1 = []
lamdlist2 = []
growthrate2 = []
phi_R0max_values = [0, 0.01, 0.1, 0.2]
colors = plt.cm.Reds(np.linspace(0.4, 1, len(phi_R0max_values)))
plt.figure(figsize=(5, 5))
for phi_R0max, color in zip(phi_R0max_values, colors):
    lamdlist = []
    growthrate = []
    for f_r in phi_R0maxlist:
        f['r'] = f_r
        lamda = lambd('r', phi_R0max, e_max, k_n, f)
        gr = frac_inact('r', phi_R0max, e_max, k_n, f)
        lamdlist.append(lamda)
        growthrate.append(gr)
    plt.plot(lamdlist, growthrate, color=color, label=rf"$\phi_R^{{0,\mathrm{{max}}}}$ = {phi_R0max}")
plt.xlabel(r"$\lambda$")
plt.ylabel("Fraction of inactive ribosomes")
plt.legend()
plt.tight_layout()
plt.savefig("inact_fraction.pdf", format='pdf')
plt.show()
'''


# This code simulates actual dynamics
# parameters
e_max = 5.5
f = {'r': 0.3, 'p': 1}
k_n = {'r': 8, 'p': 0.2}
c = {'r':100, 'p':1}
D = 10
phi_R0max = 0.001

# simulations
t_sat = 5   # short saturation duration
media_sequence = ['r','p','r','p']   # steady-state cycles

t_dep_all = {m: depletion_time(m, phi_R0max, e_max, k_n, f) for m in ['r', 'p']}
cycle_duration = max(t_dep_all.values()) + 5

time = []
pop = []
t = 0
res = []
for medium in media_sequence:

    lam = lambd(medium, phi_R0max, e_max, k_n, f)
    lag = lag_time(medium, phi_R0max, e_max, k_n, f)
    t_dep = depletion_time(medium, phi_R0max, e_max, k_n, f)
    t_sat = cycle_duration - t_dep

    N0 = steady_state_mass(medium, c)
    # lag phase
    time += [t, t+lag]
    pop  += [N0, N0]
    res+= [c[medium], c[medium]]

    # exponential growth until depletion
    tg = np.linspace(0, t_dep-lag, 120)
    Ng = N0 * np.exp(lam * tg)

    time += list(t+lag+tg)
    pop  += list(Ng)
    res += list(c[medium] - np.array(list(Ng - N0)))

    Nmax = Ng[-1]
    t += t_dep

    # short saturation plateau
    time += [t, t+t_sat]
    pop  += [Nmax, Nmax]
    res += [0,0]
    t += t_sat

    # dilution before next medium
    N_after = Nmax / D
    time.append(t)
    pop.append(N_after)
    res.append(0)

import matplotlib.ticker as ticker
plt.figure(figsize=(5,0.75))
plt.plot(time, res, linewidth=3)
plt.yscale('log')
ax = plt.gca()
#ax.set_yticks([1e-2, 1e0, 1e2])
#ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
ax.yaxis.set_minor_locator(ticker.NullLocator())
plt.xlabel("Time (h)")
plt.ylabel("Quantity")
plt.xlim((0,53))
plt.ylim((1e-3, 200))
plt.savefig("Result_fig3_res.svg", format="svg", bbox_inches="tight")
plt.show()


'''
D = 27.825594022071200
e_max = 5.5
f = {'r': 0.7357422341584520, 'p': 1}
phi_R0max = 0.0392850791258908
k_n = {'r': 8, 'p': 0.01}
c = {'r':215.44346900318800, 'p':1}
poordep = depletion_time('p', phi_R0max, e_max, k_n, f)
poorlag = lag_time('p', phi_R0max, e_max, k_n, f)
richdep = depletion_time('r', phi_R0max, e_max, k_n, f)
richlag = lag_time('r', phi_R0max, e_max, k_n, f)
print("1: ", poordep, poorlag, richdep, richlag)
print(poordep + richdep)

f = {'r': 0.9999971414371460, 'p': 1}
phi_R0max = 0.04959962799637330
poordep = depletion_time('p', phi_R0max, e_max, k_n, f)
poorlag = lag_time('p', phi_R0max, e_max, k_n, f)
richdep = depletion_time('r', phi_R0max, e_max, k_n, f)
richlag = lag_time('r', phi_R0max, e_max, k_n, f)
print("2: ", poordep, poorlag, richdep, richlag)
print(poordep + richdep)
'''















