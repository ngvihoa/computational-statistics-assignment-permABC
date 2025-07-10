import jax.numpy as jnp
from jax import random
import numba as nb  # Import Numba for acceleration
from scipy.stats import norm
from permABC.model import model
from permABC.utils_functions import Theta  # Import Theta from utils_functions

@nb.njit
def gk_transformation(Z, alpha, B, g, k):
    """
    Apply the G-and-K transformation to a standard normal variable.

    Parameters:
    - Z: Standard normal variable.
    - alpha, B, g, k: G-and-K parameters.

    Returns:
    - Transformed variable.
    """
    c = 0.8
    return alpha + B * (1 + c * (1 - jnp.exp(-g * Z)) / (1 + jnp.exp(-g * Z))) * (1 + Z ** 2) ** k * Z

class GaussianWithGK(model):
    def __init__(self, K, n_obs, low_alpha, high_alpha, low_B, high_B, low_g, high_g, low_k, high_k):
        """
        Gaussian model with G-and-K transformation.

        Parameters:
        - K (int): Number of silos.
        - n_obs (int): Number of observations per silo.
        - low_alpha, high_alpha (float): Bounds for α (location).
        - low_B, high_B (float): Bounds for B (scale).
        - low_g, high_g (float): Bounds for g (skewness).
        - low_k, high_k (float): Bounds for k (kurtosis).
        """
        super().__init__(K)
        self.n_obs = n_obs

        # Parameter support
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])
        self.support_par_glob = jnp.array([[low_alpha, high_alpha], [low_B, high_B], [low_g, high_g], [low_k, high_k]])

        self.dim_loc = 1  # μ_k is a scalar
        self.dim_glob = 4  # α, B, g, k
        self.loc_name = ["$\\mu_{"]
        self.glob_name = ["$\\alpha$", "$B$", "$g$", "$k$"]

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate prior samples for μ_k and G-and-K parameters.

        Returns:
        - Theta dataclass containing:
          - "loc": Sampled μ_k values (n_particles, n_silos, 1).
          - "glob": Sampled G-and-K parameters (n_particles, 4).
        """
        if n_silos == 0:
            n_silos = self.K
        key, key_alpha, key_B, key_g, key_k, key_mu = random.split(key, 5)

        # α, B, g, k ~ Uniform(bounds)
        alpha = random.uniform(key_alpha, shape=(n_particles, 1), minval=self.support_par_glob[0, 0], maxval=self.support_par_glob[0, 1])
        B = random.uniform(key_B, shape=(n_particles, 1), minval=self.support_par_glob[1, 0], maxval=self.support_par_glob[1, 1])
        g = random.uniform(key_g, shape=(n_particles, 1), minval=self.support_par_glob[2, 0], maxval=self.support_par_glob[2, 1])
        k = random.uniform(key_k, shape=(n_particles, 1), minval=self.support_par_glob[3, 0], maxval=self.support_par_glob[3, 1])

        # μ_k ~ Normal(α, B)
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * B + alpha[:, None, None]

        return Theta(loc=mus, glob=jnp.hstack((alpha, B, g, k)))

    def prior_logpdf(self, thetas: Theta):
        """
        Compute the log-density of the prior.

        Returns:
        - Log-density values of the prior.
        """
        log_pdf_mu = norm.logpdf(thetas.loc, loc=thetas.glob[:, 0][:, None, None], scale=thetas.glob[:, 1][:, None, None])
        return jnp.sum(log_pdf_mu, axis=1).squeeze()

    def data_generator(self, key, thetas: Theta):
        """
        Generate simulated observations using the G-and-K transformation.

        Returns:
        - Simulated data.
        """
        key, key_data = random.split(key)
        Z = random.normal(key_data, shape=(thetas.loc.shape[0], self.K, self.n_obs))

        alpha, B, g, k = thetas.glob[:, 0], thetas.glob[:, 1], thetas.glob[:, 2], thetas.glob[:, 3]
        return gk_transformation(Z, alpha[:, None, None], B[:, None, None], g[:, None, None], k[:, None, None])

   