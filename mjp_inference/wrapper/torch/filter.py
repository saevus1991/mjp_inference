import numpy as np
import torch
from scipy.integrate import solve_ivp
import multiprocessing as mp
import mjp_inference as mjpi

__all__ = ['filter']


def forward_backward_filter_ode(initial_dist: np.ndarray, rates: np.ndarray, master_equation: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: np.ndarray, observations: np.ndarray, obs_param: np.ndarray, get_gradient: bool=False):
    """
    Numpy function computing data log likelihood via forward filtering and gradient via backward filtering
    """
    # set up transition model
    master_equation.update_generator(rates)
    def odefun(t, x): return(master_equation.forward(t, x))
    # prepare containers
    forward_fun = []
    num_steps = len(obs_times)
    dim = len(initial_dist)
    time = 0.0
    norm = np.zeros(num_steps)
    inds = np.zeros((num_steps, dim), dtype=np.bool)
    tspans = np.stack([np.concatenate([np.array([0.0]), obs_times[:-1]]), obs_times], axis=1)
    llh = np.zeros((num_steps, dim))
    states = np.zeros((num_steps, dim))
    # initialize state
    state = initial_dist.copy()
    # iterate over observations
    for i, (t_i, obs_i) in enumerate(zip(obs_times, observations)):
        tspan = tspans[i]
        # solve forward #TODO: add solver method as argument
        sol = solve_ivp(odefun, tspan, state, t_eval=tspan[[1]], dense_output=True, method='RK45', rtol=1e-8, atol=1e-10)
        forward_fun.append(sol['sol'])
        state = sol['y'][:, -1]
        inds[i] = state < 1e-10
        state[inds[i]] = 1e-10
        states[i] = state
        # observation update
        llh[i] = obs_model.llh_vec(t_i, obs_i)
        state = np.log(state) + llh[i]
        max_state = np.max(state)
        state = np.exp(state-max_state)
        norm_tmp = state.sum()
        state = state / norm_tmp
        # update running quantities
        norm[i] = max_state + np.log(norm_tmp)
    # compute final output
    log_prob = norm.sum()
    # compute backward pass if required
    if get_gradient:
        # initialize rates gradient
        rates_grad = np.zeros(rates.shape)
        obs_param_grad = np.zeros(obs_param.shape)
        # compute gradient of log prob
        for i in range(num_steps-1, -1, -1):
            # pass through obs update
            if i == num_steps - 1:
                d_llh = np.exp(np.log(states[i]) + llh[i] - norm[i])
                state = np.exp(llh[i] - norm[i])
                obs_param_grad += d_llh @ obs_model.llh_vec_grad(obs_times[i], observations[i])
            else: 
                tmp = np.sum(state * np.exp(np.log(states[i]) + llh[i] - norm[i]))
                d_llh = state * np.exp(np.log(states[i]) + llh[i] - norm[i]) - tmp * np.exp(np.log(states[i]) + llh[i] - norm[i]) + np.exp(np.log(states[i]) + llh[i] - norm[i])
                state = state * np.exp(llh[i] - norm[i]) - tmp * np.exp(llh[i] - norm[i]) + np.exp(llh[i] - norm[i])
                obs_param_grad += d_llh @ obs_model.llh_vec_grad(obs_times[i], observations[i])
            # remove threshold entries
            state[inds[i]] = 0.0
            tspan = tspans[i]
            # solve backward
            def odefun(t, x):
                forward_tmp = forward_fun[i](t)
                return(master_equation.augmented_backward(t, x, forward_tmp, rates))
            terminal = np.concatenate([state, np.zeros(rates_grad.shape)])
            sol = solve_ivp(odefun, tspan[-1::-1], terminal, t_eval=tspan[[0]], method='RK45', rtol=1e-8, atol=1e-10)
            res = sol['y'].flatten()
            # get output
            state = res[:len(state)]
            rates_grad += -res[len(state):]
    if get_gradient:
        return(log_prob, state, rates_grad, obs_param_grad)
    else:
        return(log_prob)


def forward_backward_filter_krylov(initial_dist: np.ndarray, rates: np.ndarray, master_equation: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: np.ndarray, observations: np.ndarray, obs_param: np.ndarray, get_gradient: bool=False):
    """
    Numpy function computing data log likelihood via forward filtering and gradient via backward filtering
    """
    # set up compiled filter
    filt = mjpi.KrylovFilter(master_equation, obs_model, obs_times, observations, initial_dist
, rates, obs_param)
    if not get_gradient:
        log_prob = filt.llh()
        return(log_prob)
    else:
        log_prob = filt.llh()
        filt.llh_backward()
        filt.compute_rates_grad()
        rates_grad = filt.get_rates_grad()
        obs_param_grad = filt.get_obs_param_grad()
        initial_grad = filt.get_initial_grad()
        return(log_prob, initial_grad, rates_grad, obs_param_grad)


def forward_backward_filter(initial_dist: np.ndarray, rates: np.ndarray, master_equation: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: np.ndarray, observations: np.ndarray, obs_param: np.ndarray, mode: str='ode', get_gradient: bool=False):
    if mode == 'ode':
        return(forward_backward_filter_ode(initial_dist
    , rates, master_equation, obs_model, obs_times, observations, obs_param, get_gradient=get_gradient))
    elif mode == 'krylov':
        return(forward_backward_filter_krylov(initial_dist
    , rates, master_equation, obs_model, obs_times, observations, obs_param, get_gradient=get_gradient))
    else:
        msg = f'Invalid value {mode} for keyword argument mode'
        raise ValueError(msg)


def forward_backward_filter_batch(initial_dist: np.ndarray, rates: np.ndarray, transition_model: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: np.ndarray, observations: np.ndarray, obs_param: np.ndarray, mode: str='ode', get_gradient: bool=False):
    """
    Numpy function computing data log likelihood via forward filtering and gradient via backward filtering
    """
    # preparations
    if observations.ndim == 2:
        batch_size = 1
        observations = observations.reshape((1,)+observations.shape)
    batch_size = observations.shape[0]
    if initial_dist.ndim == 1:
        initial_dist = np.tile(initial_dist, [batch_size, 1])
    if obs_param.ndim == 1:
        obs_param = np.tile(obs_param, [batch_size, 1])
    if rates.ndim == 1:
        rates = np.tile(rates, [batch_size, 1])
    if get_gradient:
        log_prob = np.zeros(batch_size)
        initial_grad = np.zeros(initial_dist.shape)
        rates_grad = np.zeros(rates.shape)
        obs_param_grad = np.zeros(obs_param.shape)
        for i in range(batch_size):
            tmp = forward_backward_filter(initial_dist
        [i], rates[i], transition_model, obs_model, obs_times, observations[i], obs_param[i], mode, get_gradient)
            log_prob[i] = tmp[0]
            initial_grad[i] = tmp[1]
            rates_grad[i] = tmp[2]
            obs_param_grad[i] = tmp[3]
        return(log_prob, initial_grad, rates_grad, obs_param_grad)
    else:
        log_prob = np.zeros(batch_size)
        for i in range(batch_size):
            log_prob[i] = forward_backward_filter(initial_dist
        [i], rates[i], transition_model, obs_model, obs_times, observations[i], obs_param[i], mode, get_gradient)
        return(log_prob)


def fun(arg: tuple):
    # split arfuments
    initial_dist, rates, master_equation, obs_model, obs_times, observations, obs_param, mode, get_gradient = arg
    # call wrapped function
    return(forward_backward_filter(initial_dist, rates, master_equation, obs_model, obs_times, observations, obs_param, mode, get_gradient))
        

def forward_backward_filter_mp(initial_dist: np.ndarray, rates: np.ndarray, master_equation: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: np.ndarray, observations: np.ndarray, obs_param: np.ndarray, mode: str='ode', get_gradient: bool=False, num_workers: int=1):
    """
    Numpy function computing data log likelihood via forward filtering and gradient via backward filtering
    """
    # preparations
    if observations.ndim == 2:
        batch_size = 1
        observations = observations.reshape((1,)+observations.shape)
    batch_size = observations.shape[0]
    if initial_dist.ndim == 1:
        initial_dist = np.tile(initial_dist, [batch_size, 1])
    if obs_param.ndim == 1:
        obs_param = np.tile(obs_param, [batch_size, 1])
    if rates.ndim == 1:
        rates = np.tile(rates, [batch_size, 1])
    arg = [[initial_dist[i], rates[i], master_equation, obs_model, obs_times, observations[i], obs_param[i], mode, get_gradient] for i in range(batch_size)]
    # compute paralell result
    with mp.Pool(num_workers) as p:
        res = p.map(fun, arg)
    if get_gradient:
        log_prob = np.array([res_el[0] for res_el in res])
        initial_grad = np.stack([res_el[1] for res_el in res])
        rates_grad = np.stack([res_el[2] for res_el in res])
        obs_param_grad = np.stack([res_el[3] for res_el in res])
        return(log_prob, initial_grad, rates_grad, obs_param_grad)
    else:
        log_prob = np.array(res)
        return(log_prob)


class FilterODE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, initial_dist
: torch.Tensor, rates: torch.Tensor, transition_model: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: torch.Tensor, observations: torch.Tensor, obs_param: torch.Tensor, mode: str):
        # preparations
        get_gradient = initial_dist.requires_grad or rates.requires_grad
        if get_gradient:
            log_prob, initial_grad, rates_grad, obs_param_grad = forward_backward_filter_batch(initial_dist.numpy(), rates.numpy(), transition_model, obs_model, obs_times.numpy(), observations.numpy(), obs_param.numpy(), mode=mode, get_gradient=True)
            ctx.save_for_backward(torch.from_numpy(initial_grad), torch.from_numpy(rates_grad), torch.from_numpy(obs_param_grad))
            ctx.initial_dim = initial_dist.ndim
            ctx.rates_dim = rates.ndim
            ctx.obs_param_dim = obs_param.ndim
            return(torch.from_numpy(log_prob))
        else:
            log_prob = forward_backward_filter_batch(initial_dist.numpy(), rates.numpy(), transition_model, obs_model, obs_times.numpy(), observations.numpy(), obs_param.numpy(), mode=mode, get_gradient=False)
            return(torch.from_numpy(log_prob))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # return precomputed gradients
        initial_grad, rates_grad, obs_param_grad = ctx.saved_tensors
        if ctx.initial_dim == 2:
            initial_grad = grad_output[:, None] * initial_grad
        else:
            initial_grad = grad_output @ initial_grad
        if ctx.obs_param_dim == 2:
            obs_param_grad = grad_output[:, None] * obs_param_grad
        else:
            obs_param_grad = grad_output @ obs_param_grad
        if ctx.rates_dim == 2:
            rates_grad = grad_output[:, None] * rates_grad
        else:
            rates_grad = grad_output @ rates_grad
        return(initial_grad, rates_grad, None, None, None, None, obs_param_grad, None)


class FilterODEMP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, initial_dist
: torch.Tensor, rates: torch.Tensor, master_equation: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: torch.Tensor, observations: torch.Tensor, obs_param: torch.Tensor, mode: str, num_workers: int):
        # preparations
        get_gradient = initial_dist.requires_grad or rates.requires_grad or obs_param.requires_grad
        if get_gradient:
            log_prob, initial_grad, rates_grad, obs_param_grad = forward_backward_filter_mp(initial_dist.numpy(), rates.numpy(), master_equation, obs_model, obs_times.numpy(), observations.numpy(), obs_param.numpy(), mode=mode, get_gradient=True, num_workers=num_workers)
            ctx.save_for_backward(torch.from_numpy(initial_grad), torch.from_numpy(rates_grad), torch.from_numpy(obs_param_grad))
            ctx.initial_dim = initial_dist.ndim
            ctx.rates_dim = rates.ndim
            ctx.obs_param_dim = obs_param.ndim
            return(torch.from_numpy(log_prob))
        else:
            log_prob = forward_backward_filter_mp(initial_dist.numpy(), rates.numpy(), master_equation, obs_model, obs_times.numpy(), observations.numpy(), obs_param.numpy(), mode=mode, get_gradient=False, num_workers=num_workers)
            return(torch.from_numpy(log_prob))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # return precomputed gradients
        initial_grad, rates_grad, obs_param_grad = ctx.saved_tensors
        if ctx.initial_dim == 2:
            initial_grad = grad_output[:, None] * initial_grad
        else:
            initial_grad = grad_output @ initial_grad
        if ctx.obs_param_dim == 2:
            obs_param_grad = grad_output[:, None] * obs_param_grad
        else:
            obs_param_grad = grad_output @ obs_param_grad
        if ctx.rates_dim == 2:
            rates_grad = grad_output[:, None] * rates_grad
        else:
            rates_grad = grad_output @ rates_grad
        return(initial_grad, rates_grad, None, None, None, None, obs_param_grad, None, None)


class FilterKrylov(torch.autograd.Function):

    @staticmethod
    def forward(ctx, initial_dist
: torch.Tensor, rates: torch.Tensor, master_equation: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: torch.Tensor, observations: torch.Tensor, obs_param: torch.Tensor, mode: str, num_workers: int):
        # preparations
        get_gradient = initial_dist.requires_grad or rates.requires_grad or obs_param.requires_grad
        if get_gradient:
            log_prob, initial_grad, rates_grad, obs_param_grad = mjpi.batched_filter(initial_dist.detach().numpy(), rates.detach().numpy(), master_equation, obs_model, obs_times.numpy(), observations.numpy(), obs_param.detach().numpy(), get_gradient=True, num_workers=num_workers, backend=mode)
            check = rates_grad.sum() + obs_param_grad.sum()
            if np.isnan(check):
                torch.set_printoptions(threshold=10_000)
                print("Nan encountered in gradient")
                print(rates)
                print(obs_param)
                quit()
            ctx.save_for_backward(torch.from_numpy(initial_grad), torch.from_numpy(rates_grad), torch.from_numpy(obs_param_grad))
            ctx.initial_dim = initial_dist.ndim
            ctx.rates_dim = rates.ndim
            ctx.obs_param_dim = obs_param.ndim
            return(torch.from_numpy(log_prob))
        else:
            log_prob = mjpi.batched_filter(initial_dist.detach().numpy(), rates.detach().numpy(), master_equation, obs_model, obs_times.numpy(), observations.numpy(), obs_param.numpy(), get_gradient=False, num_workers=num_workers, backend=mode)[0]
            return(torch.from_numpy(log_prob))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # return precomputed gradients
        initial_grad, rates_grad, obs_param_grad = ctx.saved_tensors
        if ctx.initial_dim == 2:
            initial_grad = grad_output[:, None] * initial_grad
        else:
            initial_grad = grad_output @ initial_grad
        if ctx.obs_param_dim == 2:
            obs_param_grad = grad_output[:, None] * obs_param_grad
        else:
            obs_param_grad = grad_output @ obs_param_grad
        if ctx.rates_dim == 2:
            rates_grad = grad_output[:, None] * rates_grad
        else:
            rates_grad = grad_output @ rates_grad
        return(initial_grad, rates_grad, None, None, None, None, obs_param_grad, None, None)


class FilterKrylovList(torch.autograd.Function):

    @staticmethod
    def forward(ctx, initial_dist
: torch.Tensor, rates: torch.Tensor, master_equation: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: torch.Tensor, observations: torch.Tensor, obs_param: torch.Tensor, mode: str, num_workers: int):
        # preparations
        # TODO: check if this an be fused with the regular filter
        get_gradient = initial_dist.requires_grad or rates.requires_grad or obs_param.requires_grad
        if get_gradient:
            log_prob, initial_grad, rates_grad, obs_param_grad = mjpi.batched_filter_list(initial_dist.detach().numpy(), rates.detach().numpy(), master_equation, obs_model, [times.numpy() for times in obs_times], [obs.numpy() for obs in observations], obs_param.detach().numpy(), get_gradient=True, num_workers=num_workers, backend=mode)
            check = rates_grad.sum() + obs_param_grad.sum()
            if np.isnan(check):
                torch.set_printoptions(threshold=10_000)
                print("Nan encountered in gradient")
                print(rates)
                print(obs_param)
                quit()
            ctx.save_for_backward(torch.from_numpy(initial_grad), torch.from_numpy(rates_grad), torch.from_numpy(obs_param_grad))
            ctx.initial_dim = initial_dist.ndim
            ctx.rates_dim = rates.ndim
            ctx.obs_param_dim = obs_param.ndim
            return(torch.from_numpy(log_prob))
        else:
            log_prob = mjpi.batched_filter_list(initial_dist.detach().numpy(), rates.detach().numpy(), master_equation, obs_model, [times.numpy() for times in obs_times], [obs.numpy() for obs in observations], obs_param.numpy(), get_gradient=False, num_workers=num_workers, backend=mode)[0]
            return(torch.from_numpy(log_prob))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # return precomputed gradients
        initial_grad, rates_grad, obs_param_grad = ctx.saved_tensors
        if ctx.initial_dim == 2:
            initial_grad = grad_output[:, None] * initial_grad
        else:
            initial_grad = grad_output @ initial_grad
        if ctx.obs_param_dim == 2:
            obs_param_grad = grad_output[:, None] * obs_param_grad
        else:
            obs_param_grad = grad_output @ obs_param_grad
        if ctx.rates_dim == 2:
            rates_grad = grad_output[:, None] * rates_grad
        else:
            rates_grad = grad_output @ rates_grad
        return(initial_grad, rates_grad, None, None, None, None, obs_param_grad, None, None)


def filter(initial_dist: torch.Tensor, rates: torch.Tensor, master_equation: mjpi.MEInference, obs_model: mjpi.ObservationModel, obs_times: torch.Tensor, observations: torch.Tensor, obs_param: torch.Tensor, initial_map: torch.Tensor=None, rates_map: torch.Tensor=None, obs_param_map: torch.Tensor=None, mode:str='krylov', num_workers: int=1):
    if isinstance(obs_times, list) and initial_map is None:
        return(FilterKrylovList.apply(initial_dist, rates, master_equation, obs_model, obs_times, observations, obs_param, mode, num_workers))
    elif isinstance(obs_times, torch.Tensor):
        if mode == 'krylov':
            return(FilterKrylov.apply(initial_dist, rates, master_equation, obs_model, obs_times, observations, obs_param, mode, num_workers))
        if mode == 'ODE':
            if num_workers == 1:
                return(FilterODE.apply(initial_dist, rates, master_equation, obs_model, obs_times, observations, obs_param, mode))
            else:
                return(FilterODEMP.apply(initial_dist, rates, master_equation, obs_model, obs_times, observations, obs_param, mode, num_workers))
    else:
        raise ValueError('Could not parse filter arguments')