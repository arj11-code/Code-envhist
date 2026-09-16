import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares

phi_max = 0.5

# use Arial for all plot text, keep it editable in PDF export
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"

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

#This just finds where the gradient is zero. Note that it is NOT doing the second derivative test and it is NOT checking for convergence stability.
def find_ESS(phi_R0max, e_max, k_n, f, f_bounds=(0,1), phi_R0max_bounds=(0,0.4)):
    x0 = [np.mean(f_bounds), np.mean(phi_R0max_bounds)]
    sol = least_squares(selection_gradient, x0=x0, bounds=([f_bounds[0], phi_R0max_bounds[0]],[f_bounds[1], phi_R0max_bounds[1]]), args=(e_max, k_n, f), ftol=1e-12, xtol=1e-12, max_nfev=5000)
    r, phi = sol.x
    grad_norm = np.linalg.norm(sol.fun)
    if sol.success and grad_norm < 1e-6:
        return np.round([r, phi], 6)

k_n = {'r': 8, 'p': 0.2}
c = {'r': 100, 'p': 1}
f = [{},{}]
phi_R0max = [None] * 2
e_max = [5.5]*2
f[0]['p'] = 1
f[1]['p'] = 1
D = 10
# The following ESS values are obtained from the other code which uses least_squares to find points where selection gradient is zero in 2D
ESS_from_func = find_ESS(phi_R0max, e_max, k_n, f, f_bounds=(0,1), phi_R0max_bounds=(0,0.4))
ESS_f_r = ESS_from_func[0]
ESS_phi_R0max = ESS_from_func[1]
print(ESS_f_r, ESS_phi_R0max)
phi_R0max = [ESS_phi_R0max] * 2

# PIP for f_r
f_rspace = np.linspace(0.001,0.999,1000)
n = len(f_rspace)
invasion_matrix = np.zeros((n, n))
for i, resident in enumerate(f_rspace):
    for j, invader in enumerate(f_rspace):
        f[0]['r'] = resident
        f[1]['r'] = invader
        svalue = invasion_tester(phi_R0max, e_max, k_n, f)
        invasion_matrix[i, j] = svalue

# find f_r values where both conditions hold
valid_f_r = []
for k in range(n):
    if np.all(invasion_matrix[k, :] <= 0) and np.all(invasion_matrix[:, k] >= 0):
        print("ESS:", f_rspace[k])

color_matrix = np.zeros((n, n, 3))  # RGB
for i in range(n):       # resident (row)
    for j in range(n):   # invader (col)
        if i == j:
            color_matrix[i, j] = [1, 1, 1]  # no self-replacement
        else:
            inv_ij = invasion_matrix[i, j]  # does j invade i?
            inv_ji = invasion_matrix[j, i]  # does i invade j?
            if inv_ij > 0 and inv_ji > 0:
                color_matrix[i, j] = [68/255, 142/255, 66/255]       # mutual invasion
            elif inv_ij > 0 and inv_ji < 0:
                color_matrix[i, j] = [68/255, 142/255, 66/255] # one-way invasion
            elif inv_ij == 0:
                color_matrix[i,j] = [0/255,0/255,0/255]
            else:
                color_matrix[i, j] = [0/255, 0/255,0/255]       # no invasion

plt.figure(figsize=(1.5, 1.5))
plt.imshow(np.transpose(color_matrix, (1, 0, 2)), aspect="equal", origin="lower")

# decide how many ticks to show
n_ticks = 3
tick_idx = np.linspace(0, n-1, n_ticks, dtype=int)
tick_labels = [f"{f_rspace[i]:.1f}" for i in tick_idx]

plt.xticks(tick_idx, tick_labels, rotation=0)
plt.yticks(tick_idx, tick_labels)
plt.xlabel(r"Resident $f_r$")
plt.ylabel(r"Invader $f_r$")
plt.savefig('Invasionplot_f_r_100.svg', format='svg', dpi=2000, bbox_inches="tight")
plt.show()

# now do the same PIP for phi_R0max, holding f_r fixed at its ESS value
phi_R0max = [None] * 2
e_max = [5.5]*2
f[0]['p'] = 1
f[1]['p'] = 1
f[0]['r'] = ESS_f_r
f[1]['r'] = ESS_f_r

phi_R0maxspace = np.linspace(0.0001,0.1,1000)
n = len(phi_R0maxspace)
invasion_matrix = np.zeros((n, n))
for i, resident in enumerate(phi_R0maxspace):
    for j, invader in enumerate(phi_R0maxspace):
        phi_R0max[0] = resident
        phi_R0max[1] = invader
        svalue = invasion_tester(phi_R0max, e_max, k_n, f)
        invasion_matrix[i, j] = svalue

# find phi_R0max values where both conditions hold
valid_phi_R0max = []
for k in range(n):
    if np.all(invasion_matrix[k, :] <= 0) and np.all(invasion_matrix[:, k] >= 0):
        print("ESS:", phi_R0maxspace[k])

color_matrix = np.zeros((n, n, 3))  # RGB
for i in range(n):       # resident (row)
    for j in range(n):   # invader (col)
        if i == j:
            color_matrix[i, j] = [1, 1, 1]  # no self-replacement
        else:
            inv_ij = invasion_matrix[i, j]  # does j invade i?
            inv_ji = invasion_matrix[j, i]  # does i invade j?
            if inv_ij > 0 and inv_ji > 0:
                color_matrix[i, j] = [68/255, 142/255, 66/255]       # mutual invasion
            elif inv_ij > 0 and inv_ji < 0:
                color_matrix[i, j] = [68/255, 142/255, 66/255] # one-way invasion
            elif inv_ij == 0:
                color_matrix[i,j] = [0/255,0/255,0/255]
            else:
                color_matrix[i, j] = [0/255,0/255,0/255]       # no invasion

plt.figure(figsize=(1.5, 1.5))
plt.imshow(np.transpose(color_matrix, (1, 0, 2)), aspect="equal", origin="lower")

# decide how many ticks to show
n_ticks = 3
tick_idx = np.linspace(0, n-1, n_ticks, dtype=int)
tick_labels = [f"{phi_R0maxspace[i]:.1f}" for i in tick_idx]

plt.xticks(tick_idx, tick_labels)
plt.yticks(tick_idx, tick_labels)
plt.xlabel(r"Resident $\phi_R^\mathrm{0,max}$")
plt.ylabel(r"Invader $\phi_R^\mathrm{0,max}$")
plt.savefig('Invasionplot_phi_R0max_100.svg', format='svg', dpi=2000, bbox_inches="tight")
plt.show()
