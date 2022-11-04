from doctest import master
import numpy as np
from scipy.integrate import solve_ivp
import torch
from mjp_inference._c import mjp_inference as mjpi


__all__ = ['markov_transition', 'MarkovTransitionODE', 'MarkovTransitionKrylov']


class MarkovTransitionODE(torch.autograd.Function):
    """
    Markov model for a given transition model. Uses adjoint sensitivity for computing gradients
    """

    @staticmethod
    def forward(ctx, initial: torch.Tensor, rates: torch.Tensor, tspan: torch.Tensor, master_equation: mjpi.MEInference):
        # convert stuff to numpy 
        initial_np = initial.detach().numpy()
        rates_np = rates.detach().numpy()
        tspan_np = tspan.detach().numpy()
        # update the rates of the underlying model
        if np.linalg.norm(master_equation.model.rates_array - rates_np) > 1e-10:
            master_equation.update_generator(rates_np)
        ctx.master_equation = master_equation
        # define ode function
        # def odefun(t, x): return(transition_model.forward(t, x, rates_np))
        # compute solution
        sol = solve_ivp(master_equation.forward, tspan_np, initial_np, t_eval=tspan_np[[1]], dense_output=True, method='RK45', rtol=1e-8, atol=1e-10)
        ctx.forward_fun = sol['sol']
        terminal = torch.from_numpy(sol['y'][:, -1])
        terminal_ind = terminal < 1e-10
        terminal[terminal_ind] = 1e-10
        ctx.save_for_backward(initial, rates, tspan, terminal_ind)
        return(terminal)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # recover stored
        transition_model = ctx.transition_model
        forward_fun = ctx.forward_fun
        initial, rates, tspan, terminal_ind = ctx.saved_tensors
        # filter grad output
        grad_output[terminal_ind] = 0.0
        # convert to numpy
        initial_np = initial.detach().numpy()
        rates_np = rates.detach().numpy()
        tspan_np = tspan.detach().numpy()
        # ensure rate consistency
        if np.linalg.norm(transition_model.get_rates() - rates_np) > 1e-10:
            transition_model.update_generator(rates_np)
        # define ode function
        def odefun(t, x):
            forward_tmp = forward_fun(t)
            return(transition_model.augmented_backward(t, x, forward_tmp, rates_np))
        # compute backward solution
        terminal = np.concatenate([grad_output.numpy(), np.zeros(len(rates))])
        sol = solve_ivp(odefun, tspan_np[-1::-1], terminal, t_eval=tspan_np[[0]], method='RK45', rtol=1e-8, atol=1e-10)
        res = torch.from_numpy(sol['y'].flatten())
        # get output
        initial_grad = res[:len(initial)]
        rates_grad = -res[len(initial):]
        return(initial_grad, rates_grad, None, None)


class MarkovTransitionKrylov(torch.autograd.Function):
    """
    Markov model for a given transition model. Uses adjoint sensitivity with krylov exponentials for forward and backward solution.
    """

    @staticmethod
    def forward(ctx, initial: torch.Tensor, rates: torch.Tensor, tspan: torch.Tensor, master_equation: mjpi.MEInference):
        # set up compiled propagator
        time = tspan[1]-tspan[0]
        propagator = mjpi.KrylovPropagator(master_equation, initial.detach().numpy(), rates.detach().numpy(), time.item())
        terminal = propagator.propagate()
        if initial.requires_grad or rates.requires_grad or time.requires_grad:
            ctx.save_for_backward(initial, rates, time)
            ctx.propagator = propagator
        return(torch.from_numpy(terminal))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # load from forward
        initial, rates, time = ctx.saved_tensors
        propagator = ctx.propagator
        # initialize all gradients
        initial_grad = None
        rates_grad = None
        time_grad = None
        transition_model_grad = None
        # compute backward and set grads as required
        propagator.backward(grad_output.detach().numpy())
        if initial.requires_grad:
            initial_grad = torch.from_numpy(propagator.get_initial_grad())
        if rates.requires_grad:
            propagator.compute_rates_grad()
            rates_grad = torch.from_numpy(propagator.get_rates_grad())
        if time.requires_grad:
            time_grad = torch.tensor(propagator.get_time_grad()) * torch.tensor([-1, 1])
        return(initial_grad, rates_grad, time_grad, transition_model_grad)



def markov_transition(initial: torch.Tensor, rates: torch.Tensor, tspan: torch.Tensor, master_equation: mjpi.MEInference, method: str='krylov') -> torch.Tensor:
    if method == 'ode':
        return(MarkovTransitionODE.apply(initial, rates, tspan, master_equation))
    elif method == 'krylov':
        return(MarkovTransitionKrylov.apply(initial, rates, tspan, master_equation))
    else:
        raise ValueError(f'Uknown backend {method}')