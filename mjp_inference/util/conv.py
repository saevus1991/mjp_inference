import numpy as np
from mjp_inference._c.mjp_inference import MJP, ObservationModel
from typing import Union
from scipy.interpolate import interp1d
from typing import Callable


__all__ = ["marginals2means", "eval_marginals", "discretize_trajectory"]


def eval_marginals(model: MJP, prob: np.ndarray, species: list = None, keepdims: bool = False):
    """
    Compute marginal probabilites from cluster probs given a species map and a species list
    Input:
        cluster_probs: 
    """
    # preparations
    if species is None:
        species = model.species_list
    dims = model.dims
    marginals = {}
    # check if there is a time dimension
    if prob.ndim == 1:
        raise NotImplementedError
    if prob.ndim == 2:
        prob = prob.reshape(prob.shape[:1]+tuple(dims))
        for spec in species:
            ind = model.species_index(spec)
            marg_ind = tuple([j+1 for j in range(model.num_species) if j != ind])
            marginals[spec] = prob.sum(axis=marg_ind, keepdims=keepdims)
    else:
        raise ValueError(f'Can\'t handle prob of shape {prob.shape}')
    return(marginals)


def marginals2means(model: MJP, marginals: dict, species: list=None, format: str='dict'):
    if species is None:
        species = [species.name for species in model.species_list]
    cc_means = {}
    for i, spec in enumerate(species):
        states = np.arange(model.species_dict[spec].lower, model.species_dict[spec].upper+1)
        cc_means[spec] = (marginals[spec] * states[None, :]).sum(axis=1)
    if format == 'dict':
        return(cc_means)
    elif format == 'array':
        cc_means = np.concatenate([val for _, val in cc_means.items()])
        return(cc_means)
    else:
        raise ValueError(f'Unsupported format {format}')


def discretize_trajectory(trajectory: dict, sample_times: np.ndarray, obs_model:ObservationModel=None, converter: Callable=None):
    """ 
    Discretize a trajectory of a jump process by linear interpolation 
    at the support points given in sample times
    Input
        trajectory: a dict with keys 'initial', 'tspan', 'times', 'states'
        sample_times: np.array containin the sample times
        obs_model: an observation model from which to sample the state
        converter: a converter function to translate the state representation to something that the obs_model understands
    """
    if isinstance(trajectory['states'], np.ndarray) and converter is None:
        initial = np.array(trajectory['initial'])
        if (len(trajectory['times']) == 0):
            times = trajectory['tspan']
            states = np.stack([initial, initial])
        elif (trajectory['times'][-1] < trajectory['tspan'][1]):
            delta = (trajectory['tspan'][1]-trajectory['tspan'][0])/1e-3
            times = np.concatenate([trajectory['tspan'][0:1], trajectory['times'], trajectory['tspan'][1:]+delta])
            states = np.concatenate([initial.reshape(1, -1), trajectory['states'], trajectory['states'][-1:, :]])
        else:
            times = np.concatenate([trajectory['tspan'][0:1], trajectory['times']])
            states = np.concatenate([initial.reshape(1, -1), trajectory['states']])
        sample_states = interp1d(times, states, kind='zero', axis=0)(sample_times)
    elif isinstance(trajectory['states'], list) or converter is not None:
        if converter is None:
            converter = lambda x: x
        state = trajectory['initial'].copy()
        time = trajectory['tspan'][0]
        cnt = 0
        sample_states = []
        for i, t_i in enumerate(trajectory['times']):
            while cnt < len(sample_times) and time <= sample_times[cnt] and t_i > sample_times[cnt]:
                sample_states.append(converter(state))
                cnt += 1
            time = t_i
            state = trajectory['states'][i]
            if cnt == len(sample_times):
                break
        # convert to np array if possible
        # try:
        #     sample_states = np.concatenate(sample_states, axis=0)
        # except ValueError:
        #     pass
    else:
        msg = "Unsupported type {type(trajectory['states'])} for trajectory states"
        raise ValueError(msg)
    # sample observations
    if obs_model is not None:
        obs_states = np.zeros((len(sample_states), obs_model.obs_dim))
        for i in range(len(sample_times)):
            seed = np.random.randint(2**17)
            obs_states[i] = obs_model.sample(sample_times[i], sample_states[i], obs_model.param_array, seed)
        sample_states = obs_states
    return(sample_states)