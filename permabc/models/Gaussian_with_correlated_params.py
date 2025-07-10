"""
Gaussian model with correlated local and global parameters.

This module implements a hierarchical Gaussian model where component means
are correlated with a global parameter. This creates dependence between
components that can be challenging for standard ABC methods.
"""

import jax.numpy as jnp
from jax import random
import numpy as np
from jax.scipy.stats import norm

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


class GaussianWithCorrelatedParams(ModelBase):
    """
    Gaussian model with correlated local and global parameters.
    
    This model implements a hierarchical structure where:
    - α ~ Normal(0, σ_α²) is a global parameter
    - μ_k ~ Normal(0, σ_μ²) are component-specific parameters  
    - X_k,i ~ Normal(μ_k + α, 1) are the observations
    
    The correlation between μ_k and α through the observation model
    creates interesting posterior dependencies that test the effectiveness
    of permutation-based inference methods.
    
    Parameters
    ----------
    K : int
        Number of components/groups.
    n_obs : int, default=1
        Number of observations per component.
    sigma_mu : float, default=1
        Standard deviation for component means μ_k.
    sigma_alpha : float, default=1
        Standard deviation for global parameter α.
    """
    
    def __init__(self, K, n_obs=1, sigma_mu=1, sigma_alpha=1):
        """Initialize the correlated Gaussian model."""
        super().__init__(K)
        
        # Model hyperparameters
        self.n_obs = n_obs
        self.sigma_mu = sigma_mu
        self.sigma_alpha = sigma_alpha

        # Parameter support ranges (unbounded for this model)
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])   # μ_k can be any real number
        self.support_par_glob = jnp.array([[-jnp.inf, jnp.inf]])  # α can be any real number

        # Parameter dimensions and names
        self.dim_loc = 1   # μ_k is a scalar per component
        self.dim_glob = 1  # α is a single global parameter
        self.loc_name = ["$\\mu_{"]   # LaTeX name for local parameters
        self.glob_name = ["$\\alpha$"]  # LaTeX name for global parameter

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate samples from the prior distribution.
        
        Samples both local and global parameters independently from
        their respective Normal priors.
        
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
        Theta
            Parameter samples with:
            - loc: Component means μ_k of shape (n_particles, n_silos, 1)
            - glob: Global parameter α of shape (n_particles, 1)
        """
        if n_silos == 0:
            n_silos = self.K
            
        key, key_alpha, key_mu = random.split(key, 3)

        # Sample global parameter: α ~ Normal(0, σ_α²)
        alphas = random.normal(key_alpha, shape=(n_particles, 1)) * self.sigma_alpha

        # Sample component means: μ_k ~ Normal(0, σ_μ²)
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * self.sigma_mu 

        return Theta(loc=mus, glob=alphas)

    def prior_logpdf(self, thetas):
        """
        Compute log probability density of the prior distribution.
        
        Since the priors are independent, this is the sum of log densities
        for all μ_k parameters and the α parameter.
        
        Parameters
        ----------
        thetas : Theta
            Parameter values to evaluate.
            
        Returns
        -------
        np.ndarray
            Log probability densities for each particle.
        """
        # Log PDF for component means: sum over all components
        log_pdf_mu = norm.logpdf(thetas.loc, loc=0, scale=self.sigma_mu)
        log_pdf_mu_sum = jnp.sum(log_pdf_mu, axis=(1, 2))
        
        # Log PDF for global parameter
        log_pdf_alpha = norm.logpdf(thetas.glob, loc=0, scale=self.sigma_alpha)
        
        return log_pdf_mu_sum + log_pdf_alpha.reshape(-1)

    def data_generator(self, key, thetas):
        """
        Generate simulated observations from the hierarchical model.
        
        For each component k, generates n_obs observations from 
        Normal(μ_k + α, 1). The correlation between local and global
        parameters creates interesting posterior dependencies.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas : Theta
            Parameter values for simulation.
            
        Returns
        -------
        np.ndarray
            Simulated data of shape (n_particles, n_silos, n_obs).
        """
        key, key_data = random.split(key)
        
        # Extract parameters
        mus = thetas.loc[:, :, 0]      # Component means: (n_particles, n_silos)
        alphas = thetas.glob[:, 0]     # Global parameter: (n_particles,)

        # Generate observations: X_k,i ~ Normal(μ_k + α, 1)
        # Broadcasting: alphas[:, None, None] adds global effect to each component
        observations = (
            random.normal(key_data, shape=(mus.shape[0], mus.shape[1], self.n_obs)) + 
            mus[:, :, None] + 
            alphas[:, None, None]
        )

        return np.array(observations)

    def prior_generator_jax(self, key, n_particles, n_silos=0):
        """
        JAX-native version of prior generator.
        
        Returns JAX arrays directly for performance-critical applications.
        
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
            (mus, alphas) where:
            - mus: Component means of shape (n_particles, n_silos, 1)
            - alphas: Global parameter of shape (n_particles, 1)
        """
        if n_silos == 0:
            n_silos = self.K
            
        key, key_mu, key_alpha = random.split(key, 3)

        # Sample global parameter
        alphas = random.normal(key_alpha, shape=(n_particles, 1)) * self.sigma_alpha

        # Sample component means
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * self.sigma_mu 

        return mus, alphas
    
    def data_generator_jax(self, key, thetas_loc, thetas_glob):
        """
        JAX-native version of data generator.
        
        Works directly with JAX arrays for better performance in 
        JIT-compiled contexts.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas_loc : jnp.ndarray
            Local parameters (component means).
        thetas_glob : jnp.ndarray
            Global parameters (α).
            
        Returns
        -------
        jnp.ndarray
            Simulated observations.
        """
        key, key_data = random.split(key)
        
        # Extract parameters
        mus = thetas_loc[:, :, 0]
        alphas = thetas_glob[:, 0]

        # Generate correlated observations
        return (
            random.normal(key_data, shape=(mus.shape[0], mus.shape[1], self.n_obs)) + 
            mus[:, :, None] + 
            alphas[:, None, None]
        )