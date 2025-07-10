"""
Proposal kernels for ABC-SMC algorithms.

This module implements various kernel classes for parameter proposals in ABC-SMC,
including Random Walk kernels with adaptive variance and truncated proposals
for bounded parameter spaces.
"""

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import truncnorm, multivariate_normal
from ..utils.functions import Theta  # Fixed relative import
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

MIN_SAMPLES_FOR_VARIANCE = 25

class Kernel:
    """
    Base class for all proposal kernels in ABC-SMC.
    
    This abstract class defines the interface for parameter proposal mechanisms
    in ABC algorithms. It handles variance estimation, permutation-aware
    parameter handling, and provides the framework for different kernel types.
    """
    
    def __init__(
        self,
        model,
        thetas: 'Theta',
        weights: np.ndarray,
        ys_index: Optional[np.ndarray],
        zs_index: Optional[np.ndarray],
        tau_loc_glob: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([])),
        L: int = 0,
        M: int = 0,
        verbose: int = 0
    ) -> None:
        """
        Initialize the Kernel.

        Parameters
        ----------
        model : object
            Statistical model with parameter bounds and dimensionality.
        thetas : Theta
            Current parameter particles (local and global components).
        weights : array_like
            Particle weights for variance computation.
        ys_index : numpy.ndarray, optional
            Row assignment indices for permutation matching.
        zs_index : numpy.ndarray, optional
            Column assignment indices for permutation matching.
        tau_loc_glob : tuple, optional
            Pre-computed variance values (tau_loc, tau_glob).
        L : int, default=0
            Number of matched components. If 0, uses model.K.
        M : int, default=0
            Total number of components. If 0, uses model.K.
        verbose : int, default=0
            Verbosity level for debugging.
        """
        self.model = model
        self.K = model.K
        self.thetas = thetas
        self.weights = jnp.array(weights)
        self.verbose = verbose
        
        self.L = L if L != 0 else self.K
        self.M = M if M != 0 else self.K
    
        self.ys_index = np.repeat([np.arange(model.K)], thetas.loc.shape[0], axis=0) if ys_index is None else np.array(ys_index, dtype=np.int32)
        self.zs_index = np.repeat([np.arange(model.K)], thetas.loc.shape[0], axis=0) if zs_index is None else np.array(zs_index, dtype=np.int32)
        
        if tau_loc_glob[0].size == 0 and tau_loc_glob[1].size == 0:
            self.tau_loc, self.tau_glob = self.get_rw_variance()
        else:
            self.tau_loc, self.tau_glob = tau_loc_glob
        
        self.tau = self.set_rw_variance_by_particle()

    def get_rw_variance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute variance for Random Walk proposal distribution.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of local and global variances.
        """
        if self.L < self.K:
            tau_loc = self.get_tau_loc_under_matching()
        else:
            permuted_thetas_loc = self.thetas.copy().apply_permutation(self.zs_index).loc if self.K > 1 else self.thetas.copy().loc
            tau_loc = jnp.sqrt(2 * jnp.var(permuted_thetas_loc, axis=0))
            
        tau_glob = jnp.sqrt(2 * jnp.var(self.thetas.glob, axis=0))
        return tau_loc, tau_glob

    def get_tau_loc_glob(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return current variance values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of local and global variances.
        """
        return self.tau_loc, self.tau_glob
    
    def get_tau_loc_under_matching(self) -> np.ndarray:
        """
        Compute local variances for under-matching scenarios.

        Returns
        -------
        np.ndarray
            Local variances.
        """
        n_particles = self.thetas.loc.shape[0]
        thetas_match_K: List[List[np.ndarray]] = [[] for _ in range(self.K)]
        
        for i in range(n_particles):
            for k in range(self.L):
                component_idx = self.ys_index[i, k]
                param_idx = self.zs_index[i, k]
                thetas_match_K[component_idx].append(self.thetas.loc[i, param_idx])
        
        tau_loc = np.zeros_like(self.thetas.loc[0])
        for k in range(self.K):
            if len(thetas_match_K[k]) > MIN_SAMPLES_FOR_VARIANCE:
                tau_loc[k] = np.sqrt(2 * np.var(np.array(thetas_match_K[k]), axis=0))
        return tau_loc

    def set_rw_variance_by_particle(self) -> 'Theta':
        """
        Set Random Walk variance for each particle individually.

        Returns
        -------
        Theta
            Per-particle variance structure.
        """
        n_particles = self.thetas.loc.shape[0]
        tau_loc_not_match = np.max(self.tau_loc, axis=0)
        
        if self.K > 1:
            out_loc = np.zeros_like(self.thetas.loc, dtype=np.float64)
            out_loc[np.arange(n_particles)[:, None], self.zs_index[:, :self.L]] = self.tau_loc[self.ys_index]
            out_loc = np.where(out_loc == 0., tau_loc_not_match, out_loc)
        else:
            out_loc = np.repeat([self.tau_loc], n_particles, axis=0)
            
        out_glob = np.repeat([self.tau_glob], n_particles, axis=0)
        return Theta(loc=out_loc, glob=out_glob)
    

    def sample(self, key, thetas: 'Theta') -> 'Theta':
        """
        Propose new parameter values (to be implemented in subclasses).
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas : Theta
            Current parameter values.
            
        Returns
        -------
        Theta
            Proposed parameter values.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Kernel subclasses must implement `sample()` method.")

    def logpdf(self, theta: 'Theta', theta_prop: 'Theta') -> np.ndarray:
        """
        Compute log-density of proposal distribution (to be implemented in subclasses).
        
        Parameters
        ----------
        theta : Theta
            Current parameter values.
        theta_prop : Theta
            Proposed parameter values.
            
        Returns
        -------
        numpy.ndarray
            Log-probability densities for each particle.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Kernel subclasses must implement `logpdf()` method.")


class KernelRW(Kernel):
    """
    Random Walk (RW) Kernel for parameter proposals.
    
    Implements a standard Gaussian random walk proposal mechanism.
    Proposals are generated as: theta_new = theta_old + Normal(0, tau).
    
    This is the most commonly used kernel for ABC-SMC, providing
    good mixing properties when variances are well-tuned.
    """
    
    def sample(self, key) -> 'Theta':
        """
        Propose new parameter values using Random Walk.
        
        Generates proposals by adding Gaussian noise scaled by the
        current variance estimates.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
            
        Returns
        -------
        Theta
            Proposed parameter values with Gaussian perturbations.
        """
        key, key_loc, key_glob = random.split(key, 3)
        
        # Generate Gaussian perturbations
        proposed_loc = self.thetas.loc + random.normal(key_loc, shape=self.thetas.loc.shape) * self.tau.loc
        proposed_glob = self.thetas.glob + random.normal(key_glob, shape=self.thetas.glob.shape) * self.tau.glob
        
        return Theta(loc=proposed_loc, glob=proposed_glob)

    def logpdf(self, theta: 'Theta', theta_prop: 'Theta') -> np.ndarray:
        """
        Compute log-density of Random Walk proposal distribution.
        
        For Gaussian proposals: log p(theta_prop | theta) = -0.5 * ((theta_prop - theta) / tau)^2
        
        Parameters
        ----------
        theta : Theta
            Current parameter values.
        theta_prop : Theta
            Proposed parameter values.
            
        Returns
        -------
        numpy.ndarray
            Log-probability densities, shape (n_particles,).
        """
        # Local parameter log-densities
        logpdf_loc = -0.5 * ((theta_prop.loc - theta.loc) / self.tau_loc) ** 2
        
        # Global parameter log-densities  
        logpdf_glob = -0.5 * ((theta_prop.glob - theta.glob) / self.tau_glob) ** 2
        
        # Sum over all dimensions
        return jnp.sum(logpdf_loc, axis=(1, 2)) + jnp.sum(logpdf_glob, axis=1)


class KernelTruncatedRW(Kernel):
    """
    Truncated Random Walk (TRW) Kernel to enforce parameter constraints.
    
    Uses truncated normal distributions instead of clipping to properly
    handle bounded parameter spaces. This ensures proper probability
    densities and better mixing near boundaries.
    
    This kernel is essential when model parameters have strict bounds
    (e.g., positive rates, probabilities in [0,1]).
    """
    
    def sample(self, key) -> 'Theta':
        """
        Propose new parameter values with truncated normal sampling.
        
        Respects model parameter bounds by using truncated normal
        distributions that are properly normalized within the bounds.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
            
        Returns
        -------
        Theta
            Proposed parameter values within model bounds.
        """
        key, key_loc, key_glob = random.split(key, 3)

        # Extract model bounds
        loc_min, loc_max = self.model.support_par_loc[:, 0], self.model.support_par_loc[:, 1]
        glob_min, glob_max = self.model.support_par_glob[:, 0], self.model.support_par_glob[:, 1]

        # Compute normalized bounds for truncation
        a_loc = (loc_min - self.thetas.loc) / self.tau.loc
        b_loc = (loc_max - self.thetas.loc) / self.tau.loc
        a_glob = (glob_min - self.thetas.glob) / self.tau.glob
        b_glob = (glob_max - self.thetas.glob) / self.tau.glob

        # Sample from truncated normal distributions
        proposed_loc = self.thetas.loc + random.truncated_normal(key_loc, lower=a_loc, upper=b_loc) * self.tau.loc
        proposed_glob = self.thetas.glob + random.truncated_normal(key_glob, lower=a_glob, upper=b_glob) * self.tau.glob
        
        # Convert JAX arrays to NumPy for compatibility
        return Theta(loc=np.array(proposed_loc), glob=np.array(proposed_glob))

    def logpdf(self, thetas_prop: 'Theta') -> np.ndarray:
        """
        Compute log-density of truncated proposal distribution.
        
        Evaluates the properly normalized truncated normal density.
        This accounts for the normalization constant that makes the
        truncated distribution integrate to 1.
        
        Parameters
        ----------
        thetas_prop : Theta
            Proposed parameter values.
            
        Returns
        -------
        numpy.ndarray
            Log-probability densities, shape (n_particles,).
        """
        # Compute normalized bounds (same as in sample())
        a_loc = (self.model.support_par_loc[:, 0] - self.thetas.loc) / self.tau.loc
        b_loc = (self.model.support_par_loc[:, 1] - self.thetas.loc) / self.tau.loc
        a_glob = (self.model.support_par_glob[:, 0] - self.thetas.glob) / self.tau.glob
        b_glob = (self.model.support_par_glob[:, 1] - self.thetas.glob) / self.tau.glob

        # Evaluate truncated normal log-densities
        logpdf_loc = truncnorm.logpdf(thetas_prop.loc, a=a_loc, b=b_loc, 
                                     loc=self.thetas.loc, scale=self.tau.loc)
        logpdf_glob = truncnorm.logpdf(thetas_prop.glob, a=a_glob, b=b_glob, 
                                      loc=self.thetas.glob, scale=self.tau.glob)
        # Sum over all dimensions
        return jnp.sum(logpdf_loc, axis=(1, 2)) + jnp.sum(logpdf_glob, axis=1)

# TODO: Implement correlated kernel for future extensions
# class KernelCorrelatedRW(Kernel):
#     """
#     Correlated Random Walk Kernel using Multivariate Normal distribution.
#     
#     This kernel would account for correlations between parameters by using
#     a full covariance matrix instead of diagonal variances.
#     """
#     def __init__(self, model, thetas, weights, covariance_matrix, tau_k=([], []), verbose=0):
#         super().__init__(model, thetas, weights, tau_k, verbose)
#         self.covariance_matrix = covariance_matrix  # Full covariance matrix
#
#     def sample(self, key, thetas):
#         """Propose using multivariate normal with correlations."""
#         # Implementation would use jax.scipy.stats.multivariate_normal
#         pass
#
#     def logpdf(self, theta, theta_prop):
#         """Compute log-density with correlation structure."""
#         # Implementation would account for full covariance
#         pass