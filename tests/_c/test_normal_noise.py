import numpy as np
import mjp_inference as mjpi
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
from mjp_inference.util.diff import num_derivative

# set parameters
mu = np.array([5.0, 3.2, 8.1])
sigma = 2.3*np.ones(3)

# set up gauss noise
noise_model = mjpi.NormalNoise(mu=mu, sigma=sigma)

parameters = [mu, sigma]

# get samples
num_samples = 1000
samples = []
for i in range(num_samples):
    seed = np.random.randint(2**17)
    samples.append(noise_model.sample(seed))
samples = np.stack(samples, axis=0)

fig, axs = plt.subplots(1, 3)
num_steps = 200
for i, ax in enumerate(axs):
    ax.hist(samples[:, i], density=True)
    x_min, x_max = ax.get_xlim()
    x_vec = np.linspace(x_min, x_max, num_steps)
    y_vec = gaussian_kde(samples[:, i])(x_vec)
    ax.plot(x_vec, y_vec, '-', color='tab:red')
plt.show()

# check llh
normal_torch = torch.distributions.Normal(loc=torch.from_numpy(mu), scale=torch.from_numpy(sigma))
check = 0.0
for i in range(num_samples):
    llh1 = normal_torch.log_prob(torch.from_numpy(samples[i])).sum().item()
    llh2 = noise_model.log_prob(samples[i])
    check += np.abs((llh1-llh2))
print(f'Log prob check', check)

# compute gradients
obs = samples[0]
llh = noise_model.log_prob(obs)
mu_grad, sigma_grad = noise_model.log_prob_grad(obs)

# compute numerical gradients
def fun(x):
    noise_model = mjpi.NormalNoise(mu=x, sigma=sigma)
    res = np.array([noise_model.log_prob(obs)])
    return(res)

print("test mu gradient")
mu_grad_num = num_derivative(fun, mu).squeeze()
print(np.stack([mu_grad, mu_grad_num], axis=1))

def fun(x):
    noise_model = mjpi.NormalNoise(mu=mu, sigma=x)
    res = np.array([noise_model.log_prob(obs)])
    return(res)

sigma_grad_num = num_derivative(fun, sigma).squeeze()
print("test sigma gradient")
print(np.stack([sigma_grad, sigma_grad_num], axis=1))
