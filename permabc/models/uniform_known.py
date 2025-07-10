"""
Simple uniform model for demonstrating permutation effects.

This module implements a basic uniform distribution model that is particularly
useful for demonstrating the benefits of permutation-enhanced ABC. The model
creates clear label-switching issues that permABC can resolve.
"""

import jax.numpy as jnp
from jax import random, vmap
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


class Uniform_known(ModelBase):
    """
    Simple uniform model with known global parameters.
    
    This model is designed to clearly demonstrate permutation issues in ABC:
    - μ_k ~ Uniform(low, high) for each component k
    - X_k,i ~ Uniform(μ_k - 1, μ_k + 1) for observations
    
    The symmetric nature of the uniform distribution makes label-switching
    particularly problematic for standard ABC, while permABC can handle it
    naturally by finding optimal component assignments.
    
    Parameters
    ----------
    K : int
        Number of components/groups.
    n_obs : int, default=1
        Number of observations per component.
    low_loc : float, default=-2.0
        Lower bound for component means μ_k.
    high_loc : float, default=2.0
        Upper bound for component means μ_k.
    beta : float, default=1.0
        Spread parameter for observations (currently unused).
    """
    
    def __init__(self, K, n_obs=1, low_loc=-2.0, high_loc=2.0, beta=1.0):
        """Initialize the uniform model."""
        super().__init__(K)
        
        # Model parameters
        self.n_obs = n_obs
        self.low_loc = low_loc
        self.high_loc = high_loc
        self.beta = beta

        # Parameter support ranges
        self.support_par_loc = jnp.array([[low_loc, high_loc]])
        self.support_par_glob = None  # No global parameters

        # Parameter dimensions and names
        self.dim_loc = 1   # μ_k is a scalar per component
        self.dim_glob = 0  # No global parameters
        self.loc_name = ["$\\mu_{"]  # LaTeX name for local parameters
        self.glob_name = []          # No global parameter names
        
    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate samples from the prior distribution.
        
        Samples component means uniformly from the specified range.
        This creates a symmetric prior that makes permutation issues
        particularly evident.
        
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
            - loc: Component means μ_k of shape (n_particles, K, 1)
            - glob: None (no global parameters)
        """
        if n_silos == 0:
            n_silos = self.K
            
        # Sample component means uniformly
        mus = random.uniform(
            key, 
            shape=(n_particles, n_silos, 1), 
            minval=self.low_loc, 
            maxval=self.high_loc
        )
        
        return Theta(loc=mus, glob=None)
    
    def data_generator(self, key, thetas):
        """
        Generate simulated observations from the uniform model.
        
        For each component k with mean μ_k, generates observations from
        Uniform(μ_k - 1, μ_k + 1). This creates overlapping distributions
        that make component identification challenging.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas : Theta
            Parameter values for simulation.
            
        Returns
        -------
        np.ndarray
            Simulated data of shape (n_particles, K, n_obs).
        """
        n_particles = thetas.loc.shape[0]
        mus = thetas.loc  # Component means
        
        # Generate observations: X_k,i ~ Uniform(μ_k - 1, μ_k + 1)
        observations = random.uniform(
            key, 
            shape=(n_particles, self.K, self.n_obs), 
            minval=mus - 1, 
            maxval=mus + 1
        )
        
        return np.array(observations)
    
    def prior_logpdf(self, thetas):
        """
        Compute log probability density of the prior distribution.
        
        For the uniform distribution, this is constant within the support
        and -∞ outside.
        
        Parameters
        ----------
        thetas : Theta
            Parameter values to evaluate.
            
        Returns
        -------
        np.ndarray
            Log probability densities (constant for uniform prior).
        """
        # For uniform prior over finite interval, log density is constant
        n_particles = thetas.loc.shape[0]
        log_density_per_component = -jnp.log(self.high_loc - self.low_loc)
        return jnp.full(n_particles, self.K * log_density_per_component)
    
    def prior_generator_jax(self, key, n_particles, n_silos=0):
        """
        JAX-native version of prior generator.
        
        Returns JAX arrays directly for performance-critical applications.
        Since this model has no global parameters, the second return value
        is a dummy array.
        
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
            (mus, dummy_global) where:
            - mus: Component means of shape (n_particles, K, 1)
            - dummy_global: Dummy array of shape (n_particles, 1)
        """
        if n_silos == 0:
            n_silos = self.K
            
        # Sample component means
        mus = random.uniform(
            key, 
            shape=(n_particles, n_silos, 1), 
            minval=self.low_loc, 
            maxval=self.high_loc
        )
        
        # Return dummy global parameters for consistency
        dummy_global = jnp.ones((n_particles, 1))
        
        return mus, dummy_global

    def data_generator_jax(self, key, mus, scale):
        """
        JAX-native version of data generator.
        
        Works directly with JAX arrays for better performance. The scale
        parameter is ignored in this implementation.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        mus : jnp.ndarray
            Component means.
        scale : jnp.ndarray
            Scale parameter (ignored for this model).
            
        Returns
        -------
        jnp.ndarray
            Simulated observations.
        """
        n_particles = mus.shape[0]
        
        # Generate uniform observations around each mean
        return random.uniform(
            key, 
            shape=(n_particles, self.K, self.n_obs), 
            minval=mus - 1, 
            maxval=mus + 1
        )