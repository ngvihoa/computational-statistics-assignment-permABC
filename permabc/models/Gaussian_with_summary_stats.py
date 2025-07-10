import jax.numpy as jnp
from jax import random, vmap
from scipy.stats import invgamma
from model import model
from utils_functions import Theta  # Import Theta from utils_functions

class GaussianWithSummaryStats(model):
    def __init__(self, K, n_obs=1, mu_0=0, sigma_0=5, alpha=2, beta=2):
        """
        Gaussian model with summary statistics.

        Parameters:
        - K (int): Number of silos.
        - n_obs (int): Number of observations per silo.
        - mu_0 (float): Prior mean of μ_k.
        - sigma_0 (float): Prior standard deviation of μ_k.
        - alpha, beta (float): Hyperparameters of the inverse-gamma prior for σ².
        """
        super().__init__(K)
        self.n_obs = n_obs
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.alpha = alpha
        self.beta = beta

        # Parameter support ranges (for potential truncation)
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])
        self.support_par_glob = jnp.array([[0, jnp.inf]])

        self.dim_loc = 1  # μ_k is a scalar
        self.dim_glob = 1  # σ² is a scalar
        self.loc_name = ["$\mu_{"]
        self.glob_name = ["$\\sigma^2$"]

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate prior samples for μ and σ².

        Parameters:
        - key: PRNG key for randomness.
        - n_particles (int): Number of particles to generate.
        - n_silos (int): Number of silos (if 0, defaults to K).

        Returns:
        - Theta dataclass containing:
          - "loc": Sampled μ_k values (n_particles, n_silos, 1).
          - "glob": Sampled σ² values (n_particles, 1).
        """
        if n_silos == 0:
            n_silos = self.K
        key, key_mu, key_sigma = random.split(key, 3)

        # μ_k ~ Normal(mu_0, sigma_0^2)
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * self.sigma_0 + self.mu_0

        # σ² ~ Inverse-Gamma(alpha, beta)
        sigma2 = 1 / random.gamma(key_sigma, self.alpha, shape=(n_particles, 1)) * self.beta

        return Theta(loc=mus, glob=sigma2)

    def prior_logpdf(self, thetas: Theta):
        """
        Compute the log-density of the prior.

        Parameters:
        - thetas: Theta dataclass containing prior samples.

        Returns:
        - Log-density values of the prior.
        """
        log_pdf_mu = -0.5 * ((thetas.loc - self.mu_0) / self.sigma_0) ** 2
        log_pdf_sigma2 = invgamma.logpdf(thetas.glob, a=self.alpha, scale=self.beta)

        return jnp.sum(log_pdf_mu, axis=1).squeeze() + log_pdf_sigma2.reshape(-1)

    def data_generator(self, key, thetas: Theta):
        """
        Generate simulated observations.

        Parameters:
        - key: PRNG key for randomness.
        - thetas: Theta dataclass containing prior samples.

        Returns:
        - Summary statistics of the generated data.
        """
        n_particles = thetas.loc.shape[0]
        n_silos = thetas.loc.shape[1]

        mus = thetas.loc
        sigmas = jnp.sqrt(thetas.glob)[:, :, jnp.newaxis]

        key, key_data = random.split(key)
        zs = random.normal(key_data, shape=(n_particles, n_silos, self.n_obs)) * sigmas + mus

        return self.summary(zs)

    def summary(self, z):
        """
        Compute summary statistics: mean and variance per silo.

        Parameters:
        - z: Simulated observations.

        Returns:
        - Concatenated mean and variance statistics.
        """
        means = jnp.mean(z, axis=2)
        var = jnp.var((z - means[:, :, None]).reshape(z.shape[0], -1), axis=1)
        return jnp.concatenate([means, var[:, None]], axis=1)

