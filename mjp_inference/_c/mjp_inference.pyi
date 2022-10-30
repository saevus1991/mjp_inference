"""C++ implementation of a modelling software for Markov jump processes. Contains a text-based model builder, utilities for stochastic simulation, a krylov-based solver of the master equation, tools for filtering and parameter inference"""
from __future__ import annotations
import mjp_inference
import typing
import numpy
import scipy.sparse
_Shape = typing.Tuple[int, ...]

__all__ = [
    "Event",
    "Krylov",
    "KrylovFilter",
    "KrylovPropagator",
    "KrylovSolver",
    "MEInference",
    "MJP",
    "MasterEquation",
    "NoiseModel",
    "NormalNoise",
    "ObservationModel",
    "Param",
    "Rate",
    "Simulator",
    "Species",
    "Transform",
    "simulate",
    "simulate_batched"
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
class Krylov():
    def __init__(self, generator: scipy.sparse.csr_matrix[numpy.float64], initial: numpy.ndarray[numpy.float64, _Shape[m, 1]], order: int) -> None: ...
    def eval(self, arg0: float) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def expand(self, arg0: int) -> None: ...
    def get_proj(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def get_span(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    pass
class KrylovFilter():
    def __init__(self, transition_model: MEInference, obs_model: ObservationModel, obs_times: numpy.ndarray[numpy.float64, _Shape[m, 1]], observations: numpy.ndarray[numpy.float64, _Shape[m, 1]], initial_dist: numpy.ndarray[numpy.float64, _Shape[m, 1]], rates: numpy.ndarray[numpy.float64, _Shape[m, 1]], obs_param: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def compute_rates_grad(self) -> None: ...
    def get_initial_grad(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def get_obs_param_grad(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def get_rates_grad(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def log_prob(self) -> float: ...
    def log_prob_backward(self) -> None: ...
    pass
class KrylovPropagator():
    def __init__(self, transition_model: MEInference, initial: numpy.ndarray[numpy.float64, _Shape[m, 1]], rates: numpy.ndarray[numpy.float64, _Shape[m, 1]], time: float) -> None: ...
    def backward(self, grad_output: numpy.ndarray[numpy.float64]) -> None: ...
    def compute_rates_grad(self) -> None: ...
    def eval(self, time: float) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def get_initial_grad(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def get_rates_grad(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def get_time_grad(self) -> float: ...
    def propagate(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    pass
class KrylovSolver():
    @typing.overload
    def __init__(self, transition_model: MEInference, initial: numpy.ndarray[numpy.float64, _Shape[m, 1]], rates: numpy.ndarray[numpy.float64, _Shape[m, 1]], obs_times: float) -> None: ...
    @typing.overload
    def __init__(self, transition_model: MEInference, initial: numpy.ndarray[numpy.float64, _Shape[m, 1]], rates: numpy.ndarray[numpy.float64, _Shape[m, 1]], obs_times: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def backward(self, krylov_order: int, grad_output: numpy.ndarray[numpy.float64]) -> None: ...
    def compute_rates_grad(self) -> None: ...
    def forward(self, krylov_order: int = 1) -> numpy.ndarray[numpy.float64]: ...
    def get_initial_grad(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def get_rates_grad(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    pass
class MasterEquation():
    def __init__(self, model: MJP) -> None: ...
    def forward(self, arg0: float, arg1: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]: ...
    @property
    def generator(self) -> scipy.sparse.csr_matrix[numpy.float64]:
        """
        :type: scipy.sparse.csr_matrix[numpy.float64]
        """
    @property
    def hazard_generators(self) -> typing.List[scipy.sparse.csr_matrix[numpy.float64]]:
        """
        :type: typing.List[scipy.sparse.csr_matrix[numpy.float64]]
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
    def propensity(self, state: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]: ...
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
    def num_rates(self) -> int:
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
    def rate_array(self) -> numpy.ndarray[numpy.float64]:
        """
        :type: numpy.ndarray[numpy.float64]
        """
    @property
    def rate_dict(self) -> dict:
        """
        :type: dict
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
class MEInference(MasterEquation):
    def __init__(self, model: MJP) -> None: ...
    @property
    def param_generators(self) -> typing.List[scipy.sparse.csr_matrix[numpy.float64]]:
        """
        :type: typing.List[scipy.sparse.csr_matrix[numpy.float64]]
        """
    @property
    def propensity_generators(self) -> typing.List[scipy.sparse.csr_matrix[numpy.float64]]:
        """
        :type: typing.List[scipy.sparse.csr_matrix[numpy.float64]]
        """
    pass
class NoiseModel():
    def __init__(self, param_list: typing.List[Param]) -> None: ...
    def log_prob(self, obs: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> float: ...
    def log_prob_grad(self, obs: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]: ...
    def sample(self, seed: int = 3812452470) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    @property
    def param_list(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    pass
class NormalNoise(NoiseModel):
    def __init__(self, mu: numpy.ndarray[numpy.float64, _Shape[m, 1]], sigma: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    pass
class ObservationModel():
    def __init__(self, transition_model: MJP, noise_type: str) -> None: ...
    @typing.overload
    def add_param(self, name: str, value: float) -> None: ...
    @typing.overload
    def add_param(self, param: Param) -> None: ...
    def add_transform(self, transform: Transform) -> None: ...
    def build(self) -> None: ...
    def log_prob(self, time: float, state: numpy.ndarray[numpy.float64, _Shape[m, 1]], param: numpy.ndarray[numpy.float64, _Shape[m, 1]], obs: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> float: ...
    def log_prob_grad(self, time: float, state: numpy.ndarray[numpy.float64, _Shape[m, 1]], param: numpy.ndarray[numpy.float64, _Shape[m, 1]], obs: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def log_prob_grad_vec(self, time: float, param: numpy.ndarray[numpy.float64, _Shape[m, 1]], obs: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def log_prob_vec(self, time: float, param: numpy.ndarray[numpy.float64, _Shape[m, 1]], obs: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def sample(self, time: float, state: numpy.ndarray[numpy.float64, _Shape[m, 1]], param: numpy.ndarray[numpy.float64, _Shape[m, 1]], seed: int) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    @property
    def noise_param_list(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @property
    def noise_type(self) -> str:
        """
        :type: str
        """
    @property
    def num_param(self) -> int:
        """
        :type: int
        """
    @property
    def param_array(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def param_list(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @property
    def param_map(self) -> typing.List[Param]:
        """
        :type: typing.List[Param]
        """
    @property
    def param_parser(self) -> str:
        """
        :type: str
        """
    pass
class Param():
    def __init__(self, name: str, value: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @name.setter
    def name(self, arg1: str) -> None:
        pass
    @property
    def value(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
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
class Simulator():
    def __init__(self, model: MJP, initial_state: numpy.ndarray[numpy.float64, _Shape[m, 1]], tspan: numpy.ndarray[numpy.float64, _Shape[m, 1]], seed: int, max_events: int = 100000, max_event_handler: str = 'warning') -> None: ...
    @typing.overload
    def simulate(self) -> dict: ...
    @typing.overload
    def simulate(self, t_eval: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]: ...
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
class Transform():
    def __init__(self, name: str, transform_callable: tuple, output_dim: int = 1, grad_callable: tuple = ()) -> None: ...
    def grad_param(self, time: float, state: numpy.ndarray[numpy.float64, _Shape[m, 1]], param: numpy.ndarray[numpy.float64, _Shape[m, 1]], grad_output: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def transform(self, time: float, state: numpy.ndarray[numpy.float64, _Shape[m, 1]], param: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def output_dim(self) -> int:
        """
        :type: int
        """
    pass
@typing.overload
def simulate(initial_state: numpy.ndarray[numpy.float64], transition_model: MJP, obs_model: ObservationModel, t_eval: numpy.ndarray[numpy.float64], seed: int, max_events: int = 100000, max_event_handler: str = 'warning') -> numpy.ndarray[numpy.float64]:
    pass
@typing.overload
def simulate(initial_state: numpy.ndarray[numpy.float64], transition_model: MJP, tspan: numpy.ndarray[numpy.float64], seed: int, max_events: int = 100000, max_event_handler: str = 'warning') -> dict:
    pass
def simulate_batched(initial_dist: numpy.ndarray[numpy.float64], rates: numpy.ndarray[numpy.float64], transition_model: MJP, obs_model: ObservationModel, t_obs: numpy.ndarray[numpy.float64], obs_param: numpy.ndarray[numpy.float64], t_span: numpy.ndarray[numpy.float64], seed: int, num_samples: int = -1, num_workers: int = -1, max_events: int = 100000, max_event_handler: str = 'warning') -> numpy.ndarray[numpy.float64]:
    pass
