import numpy as np
import torch
import pyro
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions.transforms import AffineTransform
from mjp_inference._c.mjp_inference import ObservationModel
import mjp_inference.core as mjpi
import mjp_inference.wrapper.torch as mjpt


__all__ = ['CTHMM']


class CTHMM(TorchDistribution):
    """
    A continuous time Hidden Markov model with marginalized latent states.
    """

    def __init__(self, master_equation: mjpi.MEInference, obs_model: ObservationModel, obs_times: torch.Tensor, initial_dist: torch.Tensor, rates: torch.Tensor, obs_param: torch.Tensor, initial_map: torch.Tensor=None, rates_map: torch.Tensor=None, obs_param_map: torch.Tensor=None, validate_args: bool=False, mode: str='krylov', num_workers: int=1):
        self.master_equation = master_equation
        self.obs_model = obs_model
        self.obs_times = obs_times
        self.initial_dist = torch.empty(initial_dist.shape)
        thresh_index = initial_dist < 1e-18
        self.initial_dist[~thresh_index] = initial_dist[~thresh_index]
        self.initial_dist[thresh_index] = 1e-18
        self.rates = rates
        self.obs_param = obs_param
        self.initial_map = initial_map
        self.rates_map = rates_map
        self.obs_param_map = obs_param_map
        self.num_intervals = len(obs_times)
        self.me_mode = mode
        self.num_workers = num_workers
        # set shapes
        batch_shape, event_shape = self.get_shapes()
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

    def get_shapes(self):
        obs_dim = self.obs_model.obs_dim
        if isinstance(self.obs_times, list) and self.rates_map is not None:
            event_shape = (len(self.obs_times), self.num_intervals, obs_dim)
            batch_shape = torch.Size()
        elif isinstance(self.obs_times, torch.Tensor) or (isinstance(self.obs_times, list) and self.rates_map is None):
            event_shape = (self.num_intervals, obs_dim)
            params = [self.initial_dist, self.rates, self.obs_param]
            # set batch size
            shape_list = [param.shape[0] for param in params if param.ndim > 1]
            if isinstance(self.obs_times, list):
                shape_list += [len(self.obs_times)]
            if len(shape_list) == 0:
                batch_shape = torch.Size()
            else:
                shape_set = set(shape_list)
                if len(shape_set) == 1:
                    batch_shape = (shape_list[0],)
                elif len(shape_set) == 2 and 1 in shape_set:
                    batch_shape = (np.max(np.array(shape_list)),)
                else:
                    raise ValueError('Incompatible parameter shapes')
        else:
            raise ValueError(f'Error: Could not infer event shapes')
        return(batch_shape, event_shape)

    def sample(self, sample_shape: torch.Size=torch.Size()):
        #TODO: adapt to mapped arguments 
        shape = sample_shape + self.batch_shape + self.event_shape
        if isinstance(self.obs_times, list):
            if self.initial_dist.ndim != 1:
                raise NotImplementedError
            if self.rates.ndim != 1:
                raise NotImplementedError
            if self.obs_param.ndim != 1:
                raise NotImplementedError
            observations = []
            for i in range(len(self.obs_times)):
                # get seed
                seed = torch.randint(2**16, (1,)).item()
                # sample initial
                ind = torch.distributions.categorical.Categorical(self.initial_dist).sample()
                initial = self.master_equation.ind2state(ind.item())
                # create observation
                obs_tmp = mjpi.simulate(self.transition_model, self.obs_model, initial, self.rates.detach().numpy, self.rates.detach().numpy(), self.obs_param.detach().numpy(), self.obs_times[i].numpy(), seed)
                observations.append(torch.from_numpy(obs_tmp))
            return(observations)
        if len(shape) == 3:
            tspan = self.obs_times[[0, -1]]
            seed = torch.randint(2**16, (1,)).item()
            if sample_shape == torch.Size() and self.batch_shape != torch.Size():
                rates = self.rates
            elif sample_shape != torch.Size() and self.batch_shape == torch.Size():
                rates = self.rates.repeat(sample_shape + (1,))
            observations = mjpi.simulate_batched(self.master_equation.model, self.obs_model, self.initial_dist.detach().numpy(), rates.detach().numpy(), self.obs_param.detach().numpy(), self.obs_times.detach().numpy(), seed, num_workers=self.num_workers)
            return(torch.from_numpy(np.stack(observations, axis=0)))
        elif len(shape) == 2:
            # sample initial
            ind = torch.distributions.categorical.Categorical(self.initial_dist).sample()
            initial = self.master_equation.model.ind2state(ind.item())
            # get oservations
            seed = np.random.randint(2**16)
            observations = mjpi.simulate(self.master_equation.model, self.obs_model, initial, self.rates.detach().numpy(), self.obs_param.detach().numpy(), self.obs_times.numpy(), seed)
            return(torch.from_numpy(observations))
        else:
            raise ValueError(f'Cannot handle combination of sample shape {sample_shape} and batch shape {self.batch_shape}')

    def log_prob(self, obs_data: torch.Tensor):
        """
        Takes one vector of observations as input and computes log_prob
        """
        log_prob = mjpt.filter(self.initial_dist, self.rates, self.master_equation, self.obs_model, self.obs_times, obs_data, self.obs_param, initial_map=self.initial_map, rates_map=self.rates_map, obs_param_map=self.obs_param_map, mode=self.me_mode, num_workers=self.num_workers)
        return(log_prob)

        