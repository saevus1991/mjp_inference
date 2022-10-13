import numpy as np
from mjp_inference._c.mjp_inference import MJP
from typing import Union

__all__ = ["marginals2means", "eval_marginals"]


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
        prob = prob.reshape((1,)+tuple(dims))
        for spec in species:
            ind = model.species_index(spec)
            marg_ind = (0,) + tuple([j+1 for j in range(model.num_species) if j != ind])
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