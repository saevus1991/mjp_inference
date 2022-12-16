"""
Coarse-grained tasep version of the garcia model where sites corresponding to stem loops are smaller and spacers are mapped to larger size. Leads in principle to a smaller parameter space, but will probably produce artifacts in the expected parameter regime.
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
num_sites_ms2 = 4
num_sites_pp7 = 4
num_sites_lacz = 1
max_pol_per_lacz_site = 20
num_sites_lacy = 1
max_pol_per_lacy_site = 1
max_pol_per_termin_site = 30
alpha_ms2 = np.array([0] + [num_stems_ms2/num_sites_ms2 for i in range(num_sites_ms2)] + [0 for i in range(num_sites_pp7+2)]).cumsum()
alpha_pp7 = np.array([0] + [0 for i in range(num_sites_ms2)] + [0] + [num_stems_pp7/num_sites_pp7 for i in range(num_sites_pp7)] + [0]).cumsum()


# def test(x):
#     return( x[0]*float((x[1]==3)))
# # quit()
# x = np.array([4, 3])
# y = test(x)
# print(y)
# quit()

# --- MS2 x 24 --- spacer lacZ spacer --- PP7 x 24 --- spacer lacY spacer --- terminator 

# set up model
model = mjpi.MJP('Garcia TASEP')
# add species for 
for i in range(num_sites_ms2):
    model.add_species(f'MS2 site {i}')
model.add_species('lacZ site', upper=max_pol_per_lacz_site)
for i in range(num_sites_pp7):
    model.add_species(f'PP7 site {i}')
model.add_species('lacY site', upper=max_pol_per_lacy_site)
model.add_species('Termin site', upper=max_pol_per_termin_site)
# add some parameters
model.add_rate('k_init', value=init_rate)
model.add_rate('k_elong_ms2', value=(num_sites_ms2*elong_rate)/ms2_size)
model.add_rate('k_elong_lacz', value=(num_sites_lacz*elong_rate)/lacz_size)
model.add_rate('k_elong_pp7', value=(num_sites_pp7*elong_rate)/pp7_size)
model.add_rate('k_elong_lacy', value=(num_sites_lacy*elong_rate)/lacy_size)
model.add_rate('k_termin', value=termin_rate)
# add transitions
model.add_event(mjpi.Transition('Initiation', species=['MS2 site 0'], state=[0], target=[1], rate='k_init'))
for i in range(1, num_sites_ms2):
    model.add_event(mjpi.Transition(f'MS2 {i-1}', species=[f'MS2 site {i-1}', f'MS2 site {i}'], state=[1, 0], target=[0, 1], rate='k_elong_ms2'))
model.add_event(mjpi.Event(f'MS2 {num_sites_ms2-1}', input_species=[f'MS2 site {num_sites_ms2-1}', f'lacZ site'], output_species=[f'MS2 site {num_sites_ms2-1}', f'lacZ site'], rate='k_elong_ms2', propensity=lambda x: x[0]*(x[1]<max_pol_per_lacz_site), change_vec=[-1, 1]))
model.add_event(mjpi.Event('lacZ', input_species=['lacZ site', 'PP7 site 0'], output_species=['lacZ site', 'PP7 site 0'], rate='k_elong_lacz', propensity=lambda x: x[0]*(1-x[1]), change_vec=[-1, 1]))
for i in range(1, num_sites_pp7):
    model.add_event(mjpi.Transition(f'PP7 {i-1}', species=[f'PP7 site {i-1}', f'PP7 site {i}'], state=[1, 0], target=[0, 1], rate='k_elong_pp7'))
model.add_event(mjpi.Event(f'PP7 {num_sites_ms2-1}', input_species=[f'PP7 site {num_sites_pp7-1}', f'lacY site'], output_species=[f'PP7 site {num_sites_pp7-1}', f'lacY site'], rate='k_elong_pp7', propensity=lambda x: x[0]*(x[1]<max_pol_per_lacy_site), change_vec=[-1, 1]))
model.add_event(mjpi.Event('lacY', input_species=[f'lacY site', 'Termin site'], output_species=[f'lacY site', 'Termin site'], rate='k_elong_lacy', propensity=lambda x: x[0]*(x[1]<max_pol_per_termin_site), change_vec=[-1, 1]))
model.add_event(mjpi.Event('Termination', input_species=['Termin site'], output_species=['Termin site'], rate='k_termin', propensity=lambda x: x[0], change_vec=[-1]))
model.build()

# print(model.num_species, model.num_states)
# print(model.dims)
# quit()
# print(model.rate_array)
# print('num_species', model.num_species)
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
seed = np.random.randint(2**31)
trajectory = mjpi.simulate_full(model, initial, tspan, seed)

# get plot stuff

t_plot = np.linspace(tspan[0], tspan[1], 100)
states_plot = mjpi.discretize_trajectory(trajectory, t_plot)
ms2_plot = np.sum(alpha_ms2[None, :] * states_plot, axis=1)
pp7_plot = np.sum(alpha_pp7[None, :] * states_plot, axis=1)

print(states_plot[-1])
haz = model.hazard(states_plot[-1])
print(haz)

# plot
fig, axs = plt.subplots()
axs.plot(t_plot, ms2_plot, '-r')
axs.plot(t_plot, pp7_plot, '-g')
plt.show()