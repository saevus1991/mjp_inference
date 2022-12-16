"""
TASEP version of the garcia model with 1 sites = 1 stem footprint ~ 60 sites. 
This leads more than 100 sites and is thus not suitable for master equation based inference.
Simulations can be instructive. 
"""

import numpy as np
import mjp_inference as mjpi
import matplotlib.pyplot as plt

# fix seed
# np.random.seed(2211020943)

# some properties of the physical system
ms2_size = 1.3e3  # kb
pp7_size = 1.5e3  # kb
ms2_offset = 0  # kb
pp7_offset = 4.3e3  # kb
lacz_size = 3e3  # kb
lacy_size = 0.4e3  # kb
num_stems_ms2 = 24  # kb
num_stems_pp7 = 24  # kb
init_rate = 0.3  # 1/s
elong_rate = 30  # kb/s
termin_rate = 0.008  # kb

# some properties of the model
num_sites_ms2 = 24
site_size_ms2 = ms2_size / num_sites_ms2
num_sites_pp7 = 24
site_size_pp7 = pp7_size / num_sites_pp7
num_sites_lacz = int(np.round(2 * lacz_size / (site_size_pp7+site_size_ms2)))
site_size_lacz = lacz_size / num_sites_lacz
num_sites_lacy = int(np.round(2 * lacy_size / (site_size_pp7+site_size_ms2)))
site_size_lacy = lacy_size / num_sites_lacy
print(site_size_pp7, site_size_ms2)
print(num_sites_lacz, site_size_lacz)
print(num_sites_lacy, site_size_lacy)
# quit()
max_pol_per_ms2_site = 1
max_pol_per_pp7_site = 1
max_pol_per_lacz_site = 1
max_pol_per_lacy_site = 1
max_pol_per_termin_site = 20
alpha_ms2 = [0] + [1 for i in range(num_sites_ms2)] + [0 for i in range(num_sites_lacz)] + [0 for i in range(num_sites_pp7)] + [0 for i in range(num_sites_lacy)]
alpha_ms2 = np.array(alpha_ms2).cumsum()
alpha_pp7 = [0] + [0 for i in range(num_sites_ms2)] + [0 for i in range(num_sites_lacz)] + [1 for i in range(num_sites_pp7)] + [0 for i in range(num_sites_lacy)]
alpha_pp7 = np.array(alpha_pp7).cumsum()


# --- MS2 x 24 --- spacer lacZ spacer --- PP7 x 24 --- spacer lacY spacer --- terminator 

# set up model
model = mjpi.MJP('Garcia TASEP')
# add species for 
for i in range(num_sites_ms2):
    model.add_species(f'MS2 site {i}', upper=max_pol_per_ms2_site)
for i in range(num_sites_lacz):
    model.add_species(f'lacZ site {i}', upper=max_pol_per_lacz_site)
for i in range(num_sites_pp7):
    model.add_species(f'PP7 site {i}', upper=max_pol_per_pp7_site)
for i in range(num_sites_lacy):
    model.add_species(f'lacY site {i}', upper=max_pol_per_lacy_site)
model.add_species('Termin site', upper=max_pol_per_termin_site)
# add some parameters
model.add_rate('k_init', value=init_rate)
model.add_rate('k_elong_ms2', value=elong_rate/site_size_ms2)
model.add_rate('k_elong_lacz', value=elong_rate/site_size_lacz)
model.add_rate('k_elong_pp7', value=elong_rate/site_size_pp7)
model.add_rate('k_elong_lacy', value=elong_rate/site_size_lacy)
model.add_rate('k_termin', value=termin_rate)
# add transitions
model.add_event(mjpi.Transition('Initiation', species=['MS2 site 0'], state=[0], target=[1], rate='k_init'))
for i in range(1, num_sites_ms2):
    model.add_event(mjpi.Transition(f'MS2 {i-1}', species=[f'MS2 site {i-1}', f'MS2 site {i}'], state=[1, 0], target=[0, 1], rate='k_elong_ms2'))
model.add_event(mjpi.Transition(f'MS2 {num_sites_ms2-1}', species=[f'MS2 site {num_sites_ms2-1}', 'lacZ site 0'], rate='k_elong_ms2', state=[1, 0], target=[0, 1]))
for i in range(1, num_sites_lacz):
    model.add_event(mjpi.Transition(f'lacZ site {i-1}', species=[f'lacZ site {i-1}', f'lacZ site {i}'], state=[1, 0], target=[0, 1], rate='k_elong_lacz'))
model.add_event(mjpi.Transition(f'lacZ {num_sites_lacz-1}', species=[f'lacZ site {num_sites_lacz-1}', 'PP7 site 0'], rate='k_elong_lacz', state=[1, 0], target=[0, 1]))
for i in range(1, num_sites_pp7):
    model.add_event(mjpi.Transition(f'PP7 {i-1}', species=[f'PP7 site {i-1}', f'PP7 site {i}'], state=[1, 0], target=[0, 1], rate='k_elong_pp7'))
model.add_event(mjpi.Transition(f'PP7 {num_sites_pp7-1}', species=[f'PP7 site {num_sites_pp7-1}', 'lacY site 0'], rate='k_elong_pp7', state=[1, 0], target=[0, 1]))
for i in range(1, num_sites_lacy):
    model.add_event(mjpi.Transition(f'lacY {i-1}', species=[f'lacY site {i-1}', f'lacY site {i}'], state=[1, 0], target=[0, 1], rate='k_elong_lacy'))
model.add_event(mjpi.Event(f'lacY site {num_sites_lacy-1}', input_species=[f'lacY site {num_sites_lacy-1}', 'Termin site'], output_species=[f'lacY site {num_sites_lacy-1}', 'Termin site'], rate='k_elong_lacy', propensity=lambda x: x[0]*(x[1]<max_pol_per_termin_site), change_vec=[-1, 1]))
model.add_event(mjpi.Event('Termination', input_species=['Termin site'], output_species=['Termin site'], rate='k_termin', propensity=lambda x: x[0], change_vec=[-1]))
model.build()

# print(model.rate_array)
print('num_species', model.num_species, "num_sites", model.num_states)
print(alpha_pp7.shape, alpha_ms2.shape)
# print('num_states', model.num_states)
# print('dim', model.dims)
# state = (np.random.rand(model.num_species) * (np.array(model.dims)-1)).round()
# print(state)
# prop = model.propensity(state)
# print(prop)

# set up initial
tspan = np.array([0.0, 2000.0])
initial = np.zeros(model.num_species)

# simulate trajectory
seed = np.random.randint(2**18)
trajectory = mjpi.simulate_full(model, initial, tspan, seed)

# get plot stuff

t_plot = np.linspace(tspan[0], tspan[1], 100)
states_plot = mjpi.discretize_trajectory(trajectory, t_plot)
ms2_plot = np.sum(alpha_ms2[None, :] * states_plot, axis=1)
pp7_plot = np.sum(alpha_pp7[None, :] * states_plot, axis=1)

# print(states_plot[-1])
# haz = model.hazard(states_plot[-1])
# print(haz)

# plot
fig, axs = plt.subplots()
axs.plot(t_plot, ms2_plot, '-r')
axs.plot(t_plot, pp7_plot, '-g')
plt.show()