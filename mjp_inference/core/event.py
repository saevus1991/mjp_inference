from typing import Callable, Any
from mjp_inference._c.mjp_inference import Species, Rate
from mjp_inference._c.mjp_inference import Event as _Event
from mjp_inference.core.compile import parse_function
import numpy as np
import re
import mjp_inference.util.functions as func
from numba import cfunc, types
from scipy import LowLevelCallable

__all__ = ['ArrayFun', 'Event', 'Reaction', 'MassAction', 'Transition']


ArrayFun = types.double(types.CPointer(types.double))


class Event(_Event):

    def __init__(self, name: str, input_species: list=None, output_species: list=None, rate=None, propensity: Callable=None, change_vec: list[int]=None):
        # parse rate
        rate = self.parse_rate(name, rate)
        # parse hazard 
        propensity_callable = parse_function(propensity, ArrayFun)   
        # call Event constructor
        _Event.__init__(self, name, input_species=input_species, output_species=output_species, rate=rate, propensity_callable=propensity_callable, change_vec=change_vec)

    def parse_rate(self, name, rate):
        if rate is None:
            rate = Rate(name, 1.0)
        elif isinstance(rate, float):
            rate = Rate(name, rate)
        elif isinstance(rate, str):
            rate = Rate(rate, 1.0)
        return(rate)


class Reaction(Event):

    def __init__(self, name: str, reaction: str=None,  rate: float=None, propensity: Callable=None):
        # store reaction
        self.reaction = reaction
        # parse reaction
        input_species, input_numbers, output_species, change_vec = self.parse_reaction()
        self.input_numbers = input_numbers
        # call Event constructor
        Event.__init__(self, name, input_species=input_species, output_species=output_species, rate=rate, propensity=propensity, change_vec=change_vec)

    def parse_reaction(self):
        # split reaction in left and right side
        substrates, products = re.split(' -> ', self.reaction)
        # extract inputs 
        substrates = re.split(' \+ ', substrates)
        input_species = []
        input_numbers = []
        for substrate in substrates:
            num, species = re.split(' ', substrate)
            input_species.append(species)
            input_numbers.append(int(num))
        # extract outputs
        products = re.split(' \+ ', products)
        output_species = []
        output_numbers = []
        for product in products:
            num, species = re.split(' ', product)
            output_species.append(species)
            output_numbers.append(int(num))
        # create change vector
        all_species = input_species + [species for species in output_species if species not in input_species]
        change_vec = [0 for species in all_species]
        for num, species in zip(input_numbers, input_species):
            ind = all_species.index(species)
            change_vec[ind] -= num
        for num, species in zip(output_numbers, output_species):
            ind = all_species.index(species)
            change_vec[ind] += num
        # clean out zero entries
        output_species = [spec for spec, change in zip(all_species, change_vec) if change != 0]
        change_vec = [change for change in change_vec if change != 0]
        return(input_species, input_numbers, output_species, change_vec)


class MassAction(Reaction):

    def __init__(self, name: str, reaction: str=None,  rate: float=None):
        # store reaction
        self.reaction = reaction
        # parse reaction
        input_species, input_numbers, output_species, change_vec = self.parse_reaction()
        self.input_numbers = input_numbers
        # parse propensity
        propensity = self.parse_propensity()
        # call Event constructor
        Event.__init__(self, name, input_species=input_species, output_species=output_species, rate=rate, propensity=propensity, change_vec=change_vec)

    def parse_propensity(self):
        input_numbers = np.array(self.input_numbers).astype(np.float64)
        size = len(input_numbers)
        # parse factorial functio
        falling_factorial = cfunc(types.double(types.double, types.double), nopython=True)(func.falling_factorial)
        def mass_action(state):
            # initialize with one
            prop = 1.0
            for j in range(size):
                    prop *= falling_factorial(state[j], input_numbers[j])
            return(prop)
        return(mass_action)


class Transition(Event):

    def __init__(self, name: str, species: list[str], state: list[int], target: list[int], rate: float=None):
        input_species = species
        output_species = species
        change_vec = [target_i-state_i for target_i, state_i in zip(target, state)]
        propensity = self.parse_propensity(np.array(state), rate)
        Event.__init__(self, name, input_species=input_species, output_species=output_species, rate=rate, propensity=propensity, change_vec=change_vec)
    
    def parse_propensity(self, state, rate):
        # create prop
        size = len(state)
        def propensity(x):
            equal = True
            for i in range(size):
                if np.abs(x[i]-state[i]) > 1e-12:
                    equal = False
                    break
            if equal:
                return(1.0)
            else:
                return(0.0)
        return(propensity)
