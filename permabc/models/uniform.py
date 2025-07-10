import jax.numpy as jnp
from jax import random, vmap
from scipy.stats import invgamma, norm
from models import model
from permABC.utils_functions import Theta  # Import Theta from utils
import numpy as np

class Uniform(model):
    def __init__(self, K, n_obs = 1, low_loc = -2., high_loc = 2., high_glob = 1.):
        """
        Uniform model without summary statistics.

        Parameters:
        - K (int): Number of silos.
        - n_obs (int): Number of observations per silo.
        - low (float): Lower bound of the uniform distribution.
        - high (float): Upper bound of the uniform distribution.
        """
        super().__init__(K)
        self.n_obs = n_obs
        self.low_loc = low_loc
        self.high_loc = high_loc
        self.low_glob = 0
        self.high_glob = high_glob

        # Parameter support ranges (for potential truncation)
        self.support_par_loc = jnp.array([[low_loc, high_loc]])
        self.support_par_glob = jnp.array([[0, high_glob]])

        self.dim_loc = 1
        self.dim_glob = 1
        
    
    def prior_generator(self, key, n_particles, n_silos=0):
        mus = random.uniform(key, shape = (n_particles, self.K, 1), minval = self.low_loc, maxval = self.high_loc)
        scale = random.uniform(key, shape = (n_particles, 1), minval = self.low_glob, maxval = self.high_glob)
        return Theta(loc = mus, glob = scale)
    
    def data_generator(self, key, thetas):
        n_particles = thetas.loc.shape[0]
        mus = thetas.loc
        scale = thetas.glob[:,:,None]
        return random.uniform(key, shape = (n_particles, self.K, 1), minval = mus-scale, maxval = mus+scale)
    
    def prior_logpdf(self, thetas):
        return 0.
    
    def prior_generator_jax(self, key, n_particles, n_silos=0):
        mus = random.uniform(key, shape = (n_particles, self.K, 1), minval = self.low_loc, maxval = self.high_loc)
        scale = random.uniform(key, shape = (n_particles, 1), minval = self.low_glob, maxval = self.high_glob)
        return mus, scale

    def data_generator_jax(self, key, mus, scale):
        n_particles = mus.shape[0]
        scale = scale[:,:,None]
        return random.uniform(key, shape = (n_particles, self.K, 1), minval = mus-scale, maxval = mus+scale)