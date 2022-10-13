"""C++ implementation of a modelling software for Markov jump processes. Contains a text-based model builder, utilities for stochastic simulation, a krylov-based solver of the master equation, tools for filtering and parameter inference"""
from __future__ import annotations
import mjp_inference
import typing
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "Event",
    "MJP",
    "Rate",
    "Species"
]


class Event():
    @typing.overload
    def __init__(self, event: Event) -> None: ...
    @typing.overload
    def __init__(self, name: str, input_species: typing.List[str], output_species: typing.List[str], rate: Rate, propensity_callable: tuple, change_vec: typing.List[int], species_dict: dict = {}) -> None: ...
    def hazard(self, state: numpy.ndarray[numpy.float64]) -> float: ...
    def propensity(self, state: numpy.ndarray[numpy.float64]) -> float: ...
    @property
    def change_vec(self) -> typing.List[int]:
        """
        :type: typing.List[int]
        """
    @property
    def input_species(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @name.setter
    def name(self, arg1: str) -> None:
        pass
    @property
    def output_species(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @property
    def rate(self) -> Rate:
        """
        :type: Rate
        """
    @property
    def species_dict(self) -> dict:
        """
        :type: dict
        """
    pass
class MJP():
    def __init__(self, name: str) -> None: ...
    @typing.overload
    def add_event(self, event: Event) -> None: ...
    @typing.overload
    def add_event(self, name: str, input_species: typing.List[str], output_species: typing.List[str], rate: float, hazard_callable: tuple, change_vec: typing.List[int]) -> None: ...
    @typing.overload
    def add_rate(self, name: str, value: float) -> None: ...
    @typing.overload
    def add_rate(self, rate: Rate) -> None: ...
    @typing.overload
    def add_species(self, name: str, lower: int = 0, upper: int = 1, default_value: int = 0) -> None: ...
    @typing.overload
    def add_species(self, species: Species) -> None: ...
    def build(self) -> None: ...
    @typing.overload
    def event_index(self, event: str) -> int: ...
    @typing.overload
    def event_index(self, event_list: typing.List[str]) -> typing.List[int]: ...
    def hazard(self, state: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]: ...
    def ind2state(self, ind: int) -> numpy.ndarray[numpy.float64]: ...
    def parse_clusters(self, arg0: typing.List[typing.List[str]]) -> typing.List[typing.List[int]]: ...
    def rate_index(self, rate: str) -> int: ...
    @typing.overload
    def species_index(self, species: str) -> int: ...
    @typing.overload
    def species_index(self, species_list: typing.List[str]) -> typing.List[int]: ...
    def state2ind(self, state: numpy.ndarray[numpy.float64]) -> int: ...
    def update_state(self, state: numpy.ndarray[numpy.float64], event: int) -> numpy.ndarray[numpy.float64]: ...
    @property
    def change_vectors(self) -> typing.List[typing.List[int]]:
        """
        :type: typing.List[typing.List[int]]
        """
    @property
    def default_state(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @property
    def dims(self) -> typing.List[int]:
        """
        :type: typing.List[int]
        """
    @property
    def event_dict(self) -> dict:
        """
        :type: dict
        """
    @property
    def event_list(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @property
    def input_species(self) -> typing.List[typing.List[int]]:
        """
        :type: typing.List[typing.List[int]]
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @name.setter
    def name(self, arg1: str) -> None:
        pass
    @property
    def num_events(self) -> int:
        """
        :type: int
        """
    @property
    def num_species(self) -> int:
        """
        :type: int
        """
    @property
    def num_states(self) -> int:
        """
        :type: int
        """
    @property
    def output_species(self) -> typing.List[typing.List[int]]:
        """
        :type: typing.List[typing.List[int]]
        """
    @property
    def rate_list(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @property
    def species_dict(self) -> dict:
        """
        :type: dict
        """
    @property
    def species_list(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    pass
class Rate():
    def __init__(self, name: str, value: float = 1) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> float:
        """
        :type: float
        """
    pass
class Species():
    def __init__(self, name: str, lower: int = 0, upper: int = 1, default: int = 0) -> None: ...
    @property
    def default(self) -> int:
        """
        :type: int
        """
    @default.setter
    def default(self, arg1: int) -> None:
        pass
    @property
    def dim(self) -> int:
        """
        :type: int
        """
    @property
    def index(self) -> int:
        """
        :type: int
        """
    @index.setter
    def index(self, arg1: int) -> None:
        pass
    @property
    def lower(self) -> int:
        """
        :type: int
        """
    @lower.setter
    def lower(self, arg1: int) -> None:
        pass
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @name.setter
    def name(self, arg1: str) -> None:
        pass
    @property
    def upper(self) -> int:
        """
        :type: int
        """
    @upper.setter
    def upper(self, arg1: int) -> None:
        pass
    pass
