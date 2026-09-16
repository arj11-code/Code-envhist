import numpy as np
from scipy.optimize import root_scalar
import pandas as pd

phi_max = 0.5

def lambd(medium, phi_R0max, e_max, k_n, f, species):
    return (phi_max - phi_R0max[species])* ((1/e_max[species] + 1/(k_n[medium] * f[species][medium]))**(-1))
def phi_P(medium, phi_R0max, e_max, k_n, f, species):
    return (phi_max - phi_R0max[species]) * (e_max[species]  / (e_max[species] + k_n[medium] * f[species][medium]))
def phi_R(medium, phi_R0max, e_max, k_n, f, species):
    return (e_max[species] * phi_R0max[species] + k_n[medium] * f[species][medium] * phi_max)/(e_max[species] + k_n[medium] * f[species][medium])

def lambd_0(medium, prev_medium, phi_R0max, e_max, k_n, f, species):
    if medium == 'r1' and prev_medium == 'p1': #Should be like standard upshift
        return e_max[species] * (phi_R('r1', phi_R0max, e_max, k_n, f, species)/(phi_R('r1', phi_R0max, e_max, k_n, f, species) + phi_R0max[species])) * phi_R('p1', phi_R0max, e_max, k_n, f, species) * (1 - ((phi_R0max[species]**2) /(phi_R('r1', phi_R0max, e_max, k_n, f, species)**2)))
    if medium == 'p1' and prev_medium == 'r1': #Should be like standard downshift
        return (f[species]['r1-p1']) * phi_P('r1', phi_R0max, e_max, k_n, f, species) * k_n['p1']
    if medium == 'r2' and prev_medium == 'p2': #Should be like standard upshift
        return e_max[species] * (phi_R('r2', phi_R0max, e_max, k_n, f, species)/(phi_R('r2', phi_R0max, e_max, k_n, f, species) + phi_R0max[species])) * phi_R('p2', phi_R0max, e_max, k_n, f, species) * (1 - ((phi_R0max[species]**2) /(phi_R('r2', phi_R0max, e_max, k_n, f, species)**2)))
    if medium == 'p2' and prev_medium == 'r2': #Should be like standard downshift
        return (f[species]['r2-p2']) * phi_P('r2', phi_R0max, e_max, k_n, f, species) * k_n['p2']
    if medium == 'r1' and prev_medium == 'r2': #Change between similar media - phi_R remains same
        return (f[species]['r2-r1']) * phi_P('r2', phi_R0max, e_max, k_n, f, species) * k_n['r1']
    if medium == 'r2' and prev_medium == 'r1': #Change betweeen similar media
        return (f[species]['r1-r2']) * phi_P('r1', phi_R0max, e_max, k_n, f, species) * k_n['r2']
    if medium == 'p1' and prev_medium == 'p2':
        return (1 - f[species]['p2']) * phi_P('p2', phi_R0max, e_max, k_n, f, species) * k_n['p1']
    if medium == 'p2' and prev_medium == 'p1':
        return (1 - f[species]['p1']) * phi_P('p1', phi_R0max, e_max, k_n, f, species) * k_n['p2']
    if medium == prev_medium:
        return lambd(medium, phi_R0max, e_max, k_n, f, species)

def lag_time(medium, prev_medium, phi_R0max, e_max, k_n, f, species):
    T_lag = (1/lambd(medium, phi_R0max, e_max, k_n, f, species)) * np.log(lambd(medium, phi_R0max, e_max, k_n, f, species)/lambd_0(medium, prev_medium, phi_R0max, e_max, k_n, f, species))
    return max(0, T_lag)

def consumption_function(t, phi_R0max, e_max, k_n, f, c, nspecies, medium, prev_medium, initial_biomass): #This is the function we need to find roots of
    lambdas = np.array([lambd(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    lags = np.array([lag_time(medium, prev_medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    growth_amount = initial_biomass * (np.exp(lambdas * np.maximum(t - lags, 0)) - 1)
    return np.sum(growth_amount) - c[medium]

def consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, prev_medium, initial_biomass):
    # Lower bound to search is minimum lag time
    t_lo = np.min([lag_time(medium, prev_medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    f_lo = consumption_function(t_lo, phi_R0max, e_max, k_n, f, c, nspecies, medium, prev_medium, initial_biomass)
    if f_lo > 0:
        # consumption already exceeded at lag (just in case)
        return t_lo

    # Adaptive upper bound
    t_hi = t_lo + 10.0
    for _ in range(600):
        f_hi = consumption_function(t_hi, phi_R0max, e_max, k_n, f, c, nspecies, medium, prev_medium, initial_biomass)
        if f_hi > 0:
            break
        t_hi *= 2
    else:
        return None

    sol = root_scalar(consumption_function, args=(phi_R0max, e_max, k_n, f, c, nspecies, medium, prev_medium, initial_biomass), bracket=[t_lo, t_hi], method="brentq")
    return sol.root if sol.converged else None

def growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, prev_medium, initial_biomass, consumption_val):
    lambdas = np.array([lambd(medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    lags = np.array([lag_time(medium, prev_medium, phi_R0max, e_max, k_n, f, i) for i in range(nspecies)])
    final_biomass = initial_biomass * np.exp(lambdas * (np.maximum(consumption_val - lags,0)))
    return final_biomass #Note that the output of this function is NOT diluted

# Mean consumption time of the random community
k_n = {'r1':5.3, 'r2':5.3, 'p1':0.2, 'p2':0.2}
c = {'r1':1.0, 'r2':1.0, 'p1':1.0, 'p2':1.0}

D = 10
index = 100 #int(sys.argv[1])

# Storage
abundance_history = []
f_history = []
phi_history = []
time_history = []
lineage_history = []

# Lineage + saving
t_global = 0
next_lineage_id = 0
current_lineage_ids = [0]
parent_map = {0: -1}

max_save_species = 4
trajectory_data = []

# initial species
f = [{'p1': 0.99, 'r1': 0.99, 'p2': 0.99, 'r2': 0.99, 'r1-r2': 0.005, 'r1-p1': 0.005, 'r2-r1':0.005, 'r2-p2': 0.005}]
phi_R0max = [0.001]
e_max = [5.5]
def normalize_f(f_dict):
    f_new = f_dict.copy()

    # r1 block
    r1_block = ['r1', 'r1-r2', 'r1-p1']
    total_r1 = sum(f_new[k] for k in r1_block)
    for k in r1_block:
        f_new[k] /= total_r1

    # r2 block
    r2_block = ['r2', 'r2-r1', 'r2-p2']
    total_r2 = sum(f_new[k] for k in r2_block)
    for k in r2_block:
        f_new[k] /= total_r2

    return f_new

f[0] = normalize_f(f[0])
initial_biomass = np.array([0.1])
nspecies = 1

# add initial mutant
parent = 0

def mutate_f(parent_f):
    f_mut = {}
    for key in parent_f:
        val = parent_f[key] + np.random.normal(0, 0.01)
        # hard bounds [0,1]
        val = min(max(val, 1e-6), 1 - 1e-6)
        f_mut[key] = val

    # normalize structured blocks
    f_mut = normalize_f(f_mut)
    return f_mut

f_mut = mutate_f(f[parent])
phi_mut = np.clip(phi_R0max[parent] + np.random.normal(0, 0.005), 0, 0.2)

f.append(f_mut)
phi_R0max.append(phi_mut)
e_max.append(e_max[parent])

next_lineage_id = 1
current_lineage_ids.append(next_lineage_id)
parent_map[next_lineage_id] = current_lineage_ids[parent]
next_lineage_id += 1

initial_biomass = np.append(initial_biomass, 1e-6)
nspecies = 2

# media dynamics
media = ['p1', 'p2', 'r1', 'r2']

T = np.array([
    [10**np.random.uniform(-3, 3), 10**np.random.uniform(-3, 3), 10**np.random.uniform(-3, 3), 0.0],
    [10**np.random.uniform(-3, 3), 10**np.random.uniform(-3, 3), 0.0, 10**np.random.uniform(-3, 3)],
    [10**np.random.uniform(-3, 3), 0.0, 10**np.random.uniform(-3, 3), 10**np.random.uniform(-3, 3)],
    [0.0, 10**np.random.uniform(-3, 3), 10**np.random.uniform(-3, 3), 10**np.random.uniform(-3, 3)]])

pd.DataFrame(T, index=["p1", "p2", "r1", "r2"], columns=["p1", "p2", "r1", "r2"]).to_csv(f"transition_matrix_{index}.csv")

def next_medium(prev_medium, T, media):
    i = media.index(prev_medium)
    probs = T[i] / T[i].sum() #Normalizes T
    return np.random.choice(media, p=probs)

medium = np.random.choice(media)
prev_medium = medium
nextone = next_medium(prev_medium, T, media)

# evolution loop
mutation_counter = 0
threshold = 1e-8
print("got here", flush=True)

while mutation_counter < 1000:
    if mutation_counter % 50 == 0:
        print(mutation_counter, flush=True)
    counter = 0
    biomasses_store_compare = []
    # ecological dynamics
    while counter < 10000:
        if counter > 1 and np.allclose(biomasses_store_compare[-2], biomasses_store_compare[-1], atol=1e-9):
            break
        medium = next_medium(prev_medium, T, media)
        consumption_val = consumption_time(phi_R0max, e_max, k_n, f, c, nspecies, medium, prev_medium, initial_biomass)
        final_biomass = growth_phase(phi_R0max, e_max, k_n, f, c, nspecies, medium, prev_medium, initial_biomass, consumption_val)
        prev_medium = medium
        # record dynamics
        for i in range(nspecies):
            abundance_history.append(final_biomass[i])
            f_history.append(f[i])
            phi_history.append(phi_R0max[i])
            lineage_history.append(current_lineage_ids[i])
            time_history.append(t_global)
        t_global += 1

        initial_biomass = final_biomass / D
        biomasses_store_compare.append(initial_biomass)
        counter += 1

    # filter survivors
    alive = np.where(final_biomass > threshold)[0]
    f = [f[i] for i in alive]
    phi_R0max = [phi_R0max[i] for i in alive]
    e_max = [e_max[i] for i in alive]
    current_lineage_ids = [current_lineage_ids[i] for i in alive]
    final_biomass = final_biomass[alive]
    nspecies = len(alive)

    # save (wide format, up to 4 species)
    row = [mutation_counter]
    for i in range(max_save_species):
        if i < nspecies:
            sid = current_lineage_ids[i]
            pid = parent_map.get(sid, -1)
            row.extend([sid, pid, float(phi_R0max[i]), float(f[i]['r1']), float(f[i]['r2']), float(f[i]['p1']), float(f[i]['p2']), float(f[i]['r1-r2']), float(f[i]['r1-p1']), float(f[i]['r2-r1']), float(f[i]['r2-p2'])])
        else:
            row.extend([np.nan]*11)
    trajectory_data.append(row)

    # add new mutant
    parent = np.random.choice(nspecies)
    parent_id = current_lineage_ids[parent]

    f_mut = mutate_f(f[parent])
    phi_mut = np.clip(phi_R0max[parent] + np.random.normal(0, 0.005), 0, 0.2)

    f.append(f_mut)
    phi_R0max.append(phi_mut)
    e_max.append(e_max[parent])

    new_id = next_lineage_id
    current_lineage_ids.append(new_id)
    parent_map[new_id] = parent_id
    next_lineage_id += 1

    initial_biomass = np.append(final_biomass, 1e-6)
    nspecies += 1

    mutation_counter += 1

print("Reached here without error")
# save csv
columns = ["mutation_step"]
for i in range(1, max_save_species + 1):
    columns.extend([f"id_{i}", f"parent_{i}", f"phi_{i}", f"r1_{i}", f"r2_{i}", f"p1_{i}", f"p2_{i}", f"r1r2_{i}", f"r1p1_{i}", f"r2r1_{i}", f"r2p2_{i}"])
df_traits = pd.DataFrame(trajectory_data, columns=columns)
df_traits.to_csv(f"trait_trajectory_{index}.csv", index=False, float_format="%.3f")
