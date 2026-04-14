"""
Gaussian model without summary statistics.

This module implements a Gaussian mixture model where each component has
its own mean parameter (μ_k) and shares a global variance parameter (σ²).
The model works directly with raw observations without summary statistics.
"""

import jax.numpy as jnp
from jax import random, vmap
from scipy.stats import invgamma, norm
import numpy as np

# Import from package structure
try:
    from . import ModelBase
    from ..utils.functions import Theta
except ImportError:
    # Fallback for old structure
    try:
        from models import model as ModelBase
        from utils_functions import Theta
    except ImportError:
        from . import model as ModelBase
        from ..utils.functions import Theta


class GaussianWithNoSummaryStats(ModelBase):
    """
    Gaussian model with component-specific means and shared variance.
    
    This model assumes:
    - μ_k ~ Normal(μ₀, σ₀²) for each component k
    - σ² ~ InverseGamma(α, β) shared across components
    - X_k,i ~ Normal(μ_k, σ²) for observations
    
    The model works with raw sorted observations rather than summary statistics,
    making it suitable for cases where the full data distribution matters.
    
    Parameters
    ----------
    K : int
        Number of components/groups.
    n_obs : int, default=1
        Number of observations per component.
    mu_0 : float, default=0
        Prior mean for component means μ_k.
    sigma_0 : float, default=5
        Prior standard deviation for component means μ_k.
    alpha : float, default=2
        Shape parameter for InverseGamma prior on σ².
    beta : float, default=2
        Scale parameter for InverseGamma prior on σ².
    """
    
    def __init__(self, K, n_obs=1, mu_0=0, sigma_0=5, alpha=2, beta=2):
        """Initialize the Gaussian model without summary statistics."""
        super().__init__(K)
        
        # Model parameters
        self.n_obs = n_obs
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.alpha = alpha
        self.beta = beta

        # Parameter support ranges for potential constraints
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])  # μ_k can be any real number
        self.support_par_glob = jnp.array([[0, jnp.inf]])        # σ² must be positive

        # Parameter dimensions and names for plotting/display
        self.dim_loc = 1  # μ_k is a scalar per component
        self.dim_glob = 1  # σ² is a single global parameter
        self.loc_name = ["$\\mu_{"]      # LaTeX name for local parameters
        self.glob_name = ["$\\sigma^2$"]  # LaTeX name for global parameters

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate samples from the prior distribution.
        
        Uses NumPy random for sampling to avoid JAX recompilation
        when called with varying n_particles across SMC iterations.
        """
        if n_silos == 0:
            n_silos = self.K

        rng = np.random.default_rng(int(key[0]))

        mus = rng.standard_normal((n_particles, n_silos, 1)) * self.sigma_0 + self.mu_0

        sigma2 = 1.0 / rng.gamma(self.alpha, scale=1.0 / self.beta, size=(n_particles, 1))
        
        return Theta(loc=mus, glob=sigma2)

    def prior_logpdf(self, thetas):
        """
        Compute log probability density of the prior distribution.
        """
        log_pdf_mu = norm.logpdf(np.asarray(thetas.loc), loc=self.mu_0, scale=self.sigma_0)
        log_pdf_mu_sum = np.sum(log_pdf_mu, axis=(1, 2))
        
        log_pdf_sigma2 = invgamma.logpdf(np.asarray(thetas.glob), a=self.alpha, scale=self.beta)
        return log_pdf_mu_sum + log_pdf_sigma2.reshape(-1)

    def data_generator(self, key, thetas):
        """
        Generate simulated observations from the model.
        
        For each component k, generates n_obs observations from Normal(μ_k, σ²).
        The observations are sorted within each component to make them
        permutation-invariant.
        
        Uses NumPy for random generation and sorting to avoid JAX
        JIT recompilation when particle count varies across SMC iterations.
        """
        n_particles = thetas.loc.shape[0]
        n_silos = thetas.loc.shape[1]

        mus = np.asarray(thetas.loc)
        sigmas = np.sqrt(np.asarray(thetas.glob))[:, None, :]

        rng = np.random.default_rng(int(key[0]))
        zs = rng.standard_normal((n_particles, n_silos, self.n_obs)) * sigmas + mus

        zs.sort(axis=2)
        return zs

    def summary(self, z):
        """
        Apply summary transformation to raw data.
        
        For this model, the summary is simply sorting the observations
        within each component, which makes the data permutation-invariant
        within components.
        
        Parameters
        ----------
        z : jnp.ndarray
            Raw simulated observations.
            
        Returns
        -------
        jnp.ndarray
            Sorted observations along the last axis.
        """
        return jnp.sort(z, axis=2)
    
    def prior_generator_jax(self, key, n_particles, n_silos=0):
        """
        JAX-native version of prior generator.
        
        This version returns JAX arrays directly instead of wrapping in Theta,
        useful for performance-critical applications.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        n_particles : int
            Number of particles to generate.
        n_silos : int, default=0
            Number of components (if 0, defaults to K).
            
        Returns
        -------
        tuple
            (mus, sigma2) where:
            - mus: Component means of shape (n_particles, n_silos, 1)
            - sigma2: Global variance of shape (n_particles, 1)
        """
        if n_silos == 0:
            n_silos = self.K
            
        key, key_mu, key_sigma = random.split(key, 3)

        # Sample component means
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * self.sigma_0 + self.mu_0

        # Sample global variance
        sigma2 = self.beta / random.gamma(key_sigma, self.alpha, shape=(n_particles, 1))

        return mus, sigma2
    
    def data_generator_jax(self, key, thetas_loc, thetas_glob):
        """
        JAX-native version of data generator.
        
        This version works directly with JAX arrays for better performance
        in JIT-compiled contexts.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas_loc : jnp.ndarray
            Local parameters (component means).
        thetas_glob : jnp.ndarray
            Global parameters (variance).
            
        Returns
        -------
        jnp.ndarray
            Sorted simulated observations.
        """
        n_particles = thetas_loc.shape[0]
        n_silos = thetas_loc.shape[1]

        # Extract parameters
        mus = thetas_loc
        sigmas = jnp.sqrt(thetas_glob)[:, None, :]

        # Generate and sort observations
        key, key_data = random.split(key)
        zs = random.normal(key_data, shape=(n_particles, n_silos, self.n_obs)) * sigmas + mus

        return jnp.sort(zs, axis=2)