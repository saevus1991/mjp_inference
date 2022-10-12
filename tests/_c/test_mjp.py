import mjp_inference as mjpi
from numba import cfunc, types, carray
from scipy import LowLevelCallable
import numpy as np


# # set up species
# mrna = mjpi.Species(name='mRNA', lower=0, upper=100, default=0)
# species_dict = {'mRNA': mrna}
sig = types.double(types.CPointer(types.double))

# set up model
model = mjpi.MJP("Simple gene expression model")
# add species
species = mjpi.Species(name='G0', default=1)
model.add_species(species)
species = mjpi.Species(name='G1')
model.add_species(species)
species = mjpi.Species(name='mRNA', upper=100)
model.add_species(species)
species = mjpi.Species(name='Protein', upper=500)
model.add_species(species)
# add events
def hazard(x):
    return(0.001*x[0])
event = mjpi.Event(name='Activation', input_species=['G0'], output_species=['G0', 'G1'], hazard=hazard, change_vec=[-1, 1])
model.add_event(event)
def hazard(x):
    return(0.001*x[0])
event = mjpi.Event(name='Dectivation', input_species=['G1'], output_species=['G0', 'G1'], hazard=hazard, change_vec=[1, -1])
model.add_event(event)
def hazard(x):
    return(0.06*x[0])
event = mjpi.Event(name='Transcription', input_species=['G1'], output_species=['mRNA'], hazard=hazard, change_vec=[1])
model.add_event(event)
def hazard(x):
    return(0.001*x[0])
event = mjpi.Event(name='mRNA Decay', input_species=['mRNA'], output_species=['mRNA'], hazard=hazard, change_vec=[-1])
model.add_event(event)
def hazard(x):
    return(0.01*x[0])
event = mjpi.Event(name='Translation', input_species=['mRNA'], output_species=['Protein'], hazard=hazard, change_vec=[1])
model.add_event(event)
def hazard(x):
    return(0.0009*x[0])
event = mjpi.Event(name='Protein Decay', input_species=['Protein'], output_species=['Protein'], hazard=hazard, change_vec=[-1])
model.add_event(event)
model.build()

print("test")


# test the hazard
state = np.array([2.0, 3.0, 10.0, 50.0])
haz = model.hazard(state)
print(haz)


    # # set up model
    # model = StochasticKineticModel('Simple gene expression')
    # # set up species
    # model.add_species(name='G0')
    # model.add_species(name='G1')
    # model.add_species(name='mRNA', upper=100)
    # model.add_species(name='Protein', upper=500)
    # # create events
    # model.add_event(MassAction(name='Activation', reaction='1 G0 -> 1 G1', rate=0.001, species_dict=model.species_dict))
    # model.add_event(MassAction(name='Deactivation', reaction='1 G1 -> 1 G0', rate=0.001, species_dict=model.species_dict))
    # model.add_event(MassAction(name='Transcription', reaction='1 G1 -> 1 G1 + 1 mRNA', rate=0.06, species_dict=model.species_dict))
    # model.add_event(MassAction(name='mRNA Decay', reaction='1 mRNA -> 0 mRNA', rate=0.001, species_dict=model.species_dict))
    # model.add_event(MassAction(name='Translation', reaction='1 mRNA -> 1 mRNA + 1 Protein', rate=0.01, species_dict=model.species_dict))
    # model.add_event(MassAction(name='Protein Decay', reaction='1 Protein -> 0 Protein', rate=0.0009, species_dict=model.species_dict))

# # define a hazard function
# forward_sig = types.double(types.CPointer(types.double))
# c1 = 0.2

# @cfunc(forward_sig, nopython=True)
# def const_func(x):
#     return(x[0])


# # print(const_func.__dict__.keys())
# # quit()
# # print(const_func._wrapper_address, type(const_func._wrapper_address))
# # x = 5
# # print(const_func.address)
# # quit()

# # print(birth_hazard)
# # quit()

# # set up event
# birth = mjpi.Event("birth", input_species=['mRNA'], output_species=['mRNA'], hazard_callable=LowLevelCallable(const_func.ctypes), change_vec=[1], species_dict=species_dict)