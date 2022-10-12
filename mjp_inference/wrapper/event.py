from typing import Callable, Any
from mjp_inference._c.mjp_inference import Species
from mjp_inference._c.mjp_inference import Event as _Event
import numpy as np
import re
import mjp_inference.util.functions as func
from numba import cfunc, types
from scipy import LowLevelCallable

__all__ = ['ArrayFun', 'Event', 'Reaction', 'MassAction']


ArrayFun = types.double(types.CPointer(types.double))


class Event(_Event):

    def __init__(self, name: str, input_species: list=None, output_species: list=None, hazard: Callable=None, change_vec: list[int]=None):
        # parse hazard
        hazard_compiled = cfunc(ArrayFun, nopython=True)(hazard)
        hazard_callable = LowLevelCallable(hazard_compiled.ctypes)     
        # call Event constructor
        _Event.__init__(self, name, input_species=input_species, output_species=output_species, hazard_callable=hazard_callable, change_vec=change_vec)


class Reaction(_Event):

    def __init__(self, name: str, reaction: str=None,  rate: float=None, propensity: Callable=None):
        # store some stuff
        self.reaction = reaction
        self.rate = rate
        self.propensity = propensity
        # parse reaction
        input_species, input_numbers, output_species, change_vec = self.parse_reaction()
        self.input_numbers = input_numbers
        # parse hazard
        prop_compiled = cfunc(ArrayFun, nopython=True)(propensity)
        hazard_callable = LowLevelCallable(cfunc(ArrayFun, nopython=True)(lambda x: rate * prop_compiled(x)).ctypes)
        # call Event constructor
        _Event.__init__(self, name, input_species=input_species, output_species=output_species, hazard_callable=hazard_callable, change_vec=change_vec)

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
        # store stuff
        self.reaction = reaction
        self.rate = rate
        # parse reaction and hazard
        input_species, input_numbers, output_species, change_vec = self.parse_reaction()
        self.input_numbers = input_numbers
        hazard = self.parse_hazard()
        hazard_callable = LowLevelCallable(hazard.ctypes)
        # call Event constructor
        _Event.__init__(self, name, input_species=input_species, output_species=output_species, hazard_callable=hazard_callable, change_vec=change_vec)

    def parse_hazard(self):
        rate = self.rate
        input_numbers = np.array(self.input_numbers).astype(np.float64)
        size = len(input_numbers)
        # parse factorial functio
        falling_factorial = cfunc(types.double(types.double, types.double), nopython=True)(func.falling_factorial)
        def mass_action(state):
            # initialize with one
            prop = rate
            for j in range(size):
                    prop *= falling_factorial(state[j], input_numbers[j])
            return(prop)
        return(cfunc(ArrayFun, nopython=True)(mass_action))
