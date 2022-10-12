import mjp_inference as mjpi


# parameters
num_sites = 20
k_init = mjpi.Rate('k_init', 0.1)
k_elong = mjpi.Rate('k_elong', 0.5)
k_termin = mjpi.Rate('k_termin', 0.05)
# set up model
model = mjpi.MJP("Tasep")
# add species
for i in range(num_sites):
    model.add_species(name=f'X_{i}')
# add init
model.add_event(mjpi.Event(name='Initiation', input_species=['X_0'], output_species=['X_0'] , rate=k_init, hazard=lambda x: (1-x[0]), change_vec=[1]))
# add add hops
for i in range(1, num_sites):
    input_species = [f'X_{i-1}', f'X_{i}']
    model.add_event(mjpi.Event(name=f'Hop {i}', input_species=input_species, output_species=input_species, rate=k_elong, hazard=lambda x: x[0]*(1-x[1]), change_vec=[-1, 1]))
# add termination
model.add_event(mjpi.Event(name='Termination', input_species=[f'X_{num_sites-1}'], output_species=[f'X_{num_sites-1}'], rate=k_termin, hazard=lambda x: x[0], change_vec=[-1]))
model.build()

print(model.rate_list)

# set up new model
model = mjpi.MJP("Tasep")
# add species
for i in range(num_sites):
    model.add_species(name=f'X_{i}')
# add rates
model.add_rate('k_init', 0.1)
model.add_rate('k_elong', 0.5)
model.add_rate('k_termin', 0.05)
# add init
model.add_event(mjpi.Event(name='Initiation', input_species=['X_0'], output_species=['X_0'] , rate='k_init', hazard=lambda x: (1-x[0]), change_vec=[1]))
# add add hops
for i in range(1, num_sites):
    input_species = [f'X_{i-1}', f'X_{i}']
    model.add_event(mjpi.Event(name=f'Hop {i}', input_species=input_species, output_species=input_species, rate='k_elong', hazard=lambda x: x[0]*(1-x[1]), change_vec=[-1, 1]))
# add termination
model.add_event(mjpi.Event(name='Termination', input_species=[f'X_{num_sites-1}'], output_species=[f'X_{num_sites-1}'], rate='k_termin', hazard=lambda x: x[0], change_vec=[-1]))
model.build()

print(model.rate_list)
print(model.event_dict['Initiation'].rate.name, model.event_dict['Initiation'].rate.value)
print(model.event_dict['Hop 3'].rate.name, model.event_dict['Hop 3'].rate.value)
print(model.event_dict['Termination'].rate.name, model.event_dict['Termination'].rate.value)