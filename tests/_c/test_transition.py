import mjp_inference as mjpi
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2210311119)

# set up model
model = mjpi.MJP('3 state')
# add state
model.add_species('X', upper=2)
# add events
model.add_event(mjpi.Transition(name='0->1', species=['X'], state=[0], target=[1], rate=0.1))
model.add_event(mjpi.Transition(name='1->2', species=['X'], state=[1], target=[2], rate=0.05))
model.add_event(mjpi.Transition(name='2->0', species=['X'], state=[2], target=[0], rate=0.025))
model.build()

# simulate
initial = model.default_state
tspan = np.array([0.0, 3000.0])
t_plot = np.linspace(tspan[0], tspan[1], 200)
seed = np.random.randint(2**18)
trajectory = mjpi.simulate_full(model, initial, tspan, seed)
states_plot = mjpi.discretize_trajectory(trajectory, t_plot)

# plot
plt.plot(t_plot, states_plot, '--r')
plt.show()