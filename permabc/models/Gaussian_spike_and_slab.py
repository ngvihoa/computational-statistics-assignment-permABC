import jax.numpy as jnp
from jax import random
import numba as nb  # Import Numba for acceleration
from scipy.stats import invgamma, norm
from permABC.model import model
from permABC.utils_functions import Theta  # Import Theta from utils_functions

class GaussianWithSpikeSlab(model):
    def __init__(self, K, n_obs=1, sigma_spike=0.1, sigma_slab=2.0, a_ig=2, b_ig=2):
        """
        Gaussian Spike-and-Slab model.

        Parameters:
        - K (int): Number of silos.
        - n_obs (int): Number of observations per silo.
        - sigma_spike (float): Standard deviation of the spike component.
        - sigma_slab (float): Standard deviation of the slab component.
        - a_ig, b_ig (float): Hyperparameters for inverse-gamma prior on σ².
        """
        super().__init__(K)
        self.n_obs = n_obs
        self.sigma_spike = sigma_spike
        self.sigma_slab = sigma_slab
        self.a_ig = a_ig
        self.b_ig = b_ig

        # Parameter support
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])
        self.support_par_glob = jnp.array([[0, jnp.inf]])

        self.dim_loc = 1  # μ_k is a scalar
        self.dim_glob = 1  # σ² is a scalar
        self.loc_name = ["$\\mu_{"]
        self.glob_name = ["$\\sigma^2$"]

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate prior samples for μ_k and σ².

        Returns:
        - Theta dataclass containing:
          - "loc": Sampled μ_k values (n_particles, n_silos, 1).
          - "glob": Sampled σ² values (n_particles, 1).
        """
        if n_silos == 0:
            n_silos = self.K
        key, key_sigma2, key_choice, key_spike, key_slab = random.split(key, 5)

        # σ² ~ Inverse-Gamma(a, b)
        sigma2 = 1 / random.gamma(key_sigma2, self.a_ig, shape=(n_particles, 1)) * self.b_ig

        # Spike-or-slab selection (Bernoulli 0.5)
        spike_or_slab = random.choice(key_choice, a=2, shape=(n_particles, n_silos, 1))

        # μ_k ~ Mixture of two Gaussians
        mus = (
            random.normal(key_spike, shape=(n_particles, n_silos, 1)) * self.sigma_spike * spike_or_slab
            + random.normal(key_slab, shape=(n_particles, n_silos, 1)) * self.sigma_slab * (1 - spike_or_slab)
        )

        return Theta(loc=mus, glob=sigma2)

    def prior_logpdf(self, thetas: Theta):
        """
        Compute the log-density of the prior.

        Returns:
        - Log-density values of the prior.
        """
        log_pdf_sigma2 = invgamma.logpdf(thetas.glob, a=self.a_ig, scale=self.b_ig)
        log_pdf_mu = jnp.log(
            0.5 * norm.pdf(thetas.loc, scale=self.sigma_spike)
            + 0.5 * norm.pdf(thetas.loc, scale=self.sigma_slab)
        )

        return jnp.sum(log_pdf_mu, axis=1).squeeze() + log_pdf_sigma2.reshape(-1)


    def data_generator(self, key, thetas: Theta):
        """
        Generate simulated observations.

        Returns:
        - Simulated data.
        """
        key, key_data = random.split(key)
        return random.normal(key_data, shape=(thetas.loc.shape[0], thetas.loc.shape[1], self.n_obs))*jnp.sqrt(thetas.glob)[:, :, None] + thetas.loc

 