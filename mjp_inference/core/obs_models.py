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
        mu_transform = self.mu_transform(model, observed_species)
        sigma_transform = self.sigma_transform(model, observed_species)
        self.add_transform(Transform('mu', mu_transform))
        self.add_transform(Transform('sigma', sigma_transform))
        self.build()

    def mu_transform(self, model, observed_species):
        ind = model.species_index(observed_species)
        def mu(time, state, param, transformed):
            transformed[0] = state[ind]
        return(mu)

    def sigma_transform(self, model, observed_species):
        def sigma(time, state, param, transformed):
            transformed[0] = param[0]
        return(sigma)