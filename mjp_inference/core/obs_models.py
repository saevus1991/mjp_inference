from mjp_inference._c.mjp_inference import MJP, ObservationModel
from mjp_inference.core.transform import Transform


__all__ = ['NormalObs']


class NormalObs(ObservationModel):

    def __init__(self, model: MJP, sigma: float, observed_species: str):
        # set up model
        ObservationModel.__init__(self, model, noise_type='normal')
        # add parameter
        self.add_param(name='sigma', value=sigma)
        # add transforms
        mu, mu_grad = self.mu_transform(model, observed_species)
        sigma, sigma_grad = self.sigma_transform(model, observed_species)
        self.add_transform(Transform('mu', mu, transform_grad=mu_grad))
        self.add_transform(Transform('sigma', sigma, transform_grad=sigma_grad))
        self.build()

    def mu_transform(self, model, observed_species):
        ind = model.species_index(observed_species)
        def mu(time, state, param, transformed):
            transformed[0] = state[ind]
        def mu_grad(time, state, param, grad_output, grad):
            grad[0] = 0.0
        return(mu, mu_grad)

    def sigma_transform(self, model, observed_species):
        def sigma(time, state, param, transformed):
            transformed[0] = param[0]
        def sigma_grad(time, state, param, grad_output, grad):
            grad[0] = grad_output[0]
        return(sigma, sigma_grad)