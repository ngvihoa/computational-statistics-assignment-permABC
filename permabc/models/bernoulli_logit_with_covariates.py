"""
Bernoulli logit model with weather covariates.

This module implements a hierarchical logistic regression model where:
- Global parameters (beta) model feature effects across all components
- Local parameters (alpha_k) model component-specific intercepts
- Used for estimating rain probability given weather features
"""

import jax.numpy as jnp
from jax import random, vmap
from scipy.stats import norm
import numpy as np
from scipy.special import expit, logit

# Import from package structure
try:
    from . import ModelBase
    from ..utils.functions import Theta
except ImportError:
    # Fallback for old structure
    try:
        from models import ModelBase
        from utils.functions import Theta
    except ImportError:
        from . import ModelBase
        from ..utils.functions import Theta


class BernoulliLogitWithCovariates(ModelBase):
    """
    Hierarchical Bernoulli-logit model with weather feature covariates.
    
    This model assumes:
    - α_k ~ Normal(μ_α, σ_α²) for each component k (intercepts)
    - β_j ~ Normal(μ_β, σ_β²) for each feature j (global effects)
    - y_k,i | x_k,i ~ Bernoulli(logit^{-1}(α_k + x_k,i^T β))
    
    The model is designed for binary rain probability estimation across multiple
    geographic regions/provinces, using normalized weather features as covariates.
    
    Parameters
    ----------
    K : int
        Number of components (regions/provinces).
    n_obs : int, default=1
        Number of observations per component.
    n_features : int, default=5
        Number of weather features (covariates).
    mu_alpha : float, default=0
        Prior mean for component intercepts α_k.
    sigma_alpha : float, default=2
        Prior standard deviation for component intercepts α_k.
    mu_beta : float, default=0
        Prior mean for feature coefficients β_j.
    sigma_beta : float, default=2
        Prior standard deviation for feature coefficients β_j.
    X_cov : np.ndarray, optional
        Covariate matrix of shape (K, n_obs, n_features).
        If None, will use uniform covariates.
    """
    
    def __init__(
        self,
        K,
        n_obs=1,
        n_features=5,
        mu_alpha=0.0,
        sigma_alpha=2.0,
        mu_beta=0.0,
        sigma_beta=2.0,
        X_cov=None,
    ):
        """Initialize the Bernoulli-logit model with covariates."""
        super().__init__(K)
        
        # Model parameters
        self.n_obs = n_obs
        self.n_features = n_features
        self.mu_alpha = mu_alpha
        self.sigma_alpha = sigma_alpha
        self.mu_beta = mu_beta
        self.sigma_beta = sigma_beta
        
        # Covariate matrix: store as provided (already normalized externally)
        if X_cov is None:
            X_cov = np.ones((K, n_obs, n_features))
        self.X_cov = np.asarray(X_cov, dtype=np.float32)
        
        # Verify shape consistency
        if self.X_cov.shape != (K, n_obs, n_features):
            raise ValueError(
                f"X_cov shape {self.X_cov.shape} does not match expected (K={K}, n_obs={n_obs}, n_features={n_features})"
            )
        
        # Parameter support ranges
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])  # α_k can be any real
        self.support_par_glob = jnp.array([[-jnp.inf, jnp.inf]] * n_features)  # β_j can be any real
        
        # Parameter dimensions and names for plotting/display
        self.dim_loc = 1  # α_k is a scalar per component
        self.dim_glob = n_features  # β_j is a vector per feature
        self.loc_name = ["$\\alpha_{"]  # LaTeX name for local parameters (intercepts)
        self.glob_name = [f"$\\beta_{{{i}}}$" for i in range(n_features)]  # LaTeX names for coefficients
    
    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate samples from the prior distribution.
        
        Uses NumPy random to avoid JAX recompilation when n_particles varies
        across SMC iterations.
        
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
            Parameter samples with loc=(n_particles, K, 1) for α_k
            and glob=(n_particles, n_features) for β_j.
        """
        if n_silos == 0:
            n_silos = self.K
        
        rng = np.random.default_rng(int(key[0]))
        
        # Sample intercepts α_k ~ Normal(μ_α, σ_α²)
        alphas = (
            rng.standard_normal((n_particles, n_silos, 1)) * self.sigma_alpha + self.mu_alpha
        )
        
        # Sample coefficients β_j ~ Normal(μ_β, σ_β²) for each feature
        betas = (
            rng.standard_normal((n_particles, self.n_features)) * self.sigma_beta + self.mu_beta
        )
        
        return Theta(loc=alphas, glob=betas)
    
    def prior_logpdf(self, thetas):
        """
        Compute log probability density of the prior distribution.
        
        Parameters
        ----------
        thetas : Theta
            Parameter values with loc=(n_particles, K, 1) and glob=(n_particles, n_features).
            
        Returns
        -------
        np.ndarray
            Log prior densities of shape (n_particles,).
        """
        # Log pdf for α_k ~ Normal(μ_α, σ_α²)
        log_pdf_alpha = norm.logpdf(
            np.asarray(thetas.loc), loc=self.mu_alpha, scale=self.sigma_alpha
        )
        log_pdf_alpha_sum = np.sum(log_pdf_alpha, axis=(1, 2))
        
        # Log pdf for β_j ~ Normal(μ_β, σ_β²)
        log_pdf_beta = norm.logpdf(
            np.asarray(thetas.glob), loc=self.mu_beta, scale=self.sigma_beta
        )
        log_pdf_beta_sum = np.sum(log_pdf_beta, axis=1)
        
        return log_pdf_alpha_sum + log_pdf_beta_sum
    
    def data_generator(self, key, thetas):
        """
        Generate simulated binary observations from the logit model.
        
        For each component k and observation i, generates binary rain y_k,i from
        Bernoulli(logit^{-1}(α_k + X_cov[k,i,:] · β)).
        
        Uses NumPy for random generation to avoid JAX JIT recompilation.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas : Theta
            Parameter samples with loc=(n_particles, K, 1) and glob=(n_particles, n_features).
            
        Returns
        -------
        np.ndarray
            Simulated binary observations of shape (n_particles, K, n_obs).
        """
        n_particles = thetas.loc.shape[0]
        n_silos = thetas.loc.shape[1]
        
        alphas = np.asarray(thetas.loc)  # (n_particles, K, 1)
        betas = np.asarray(thetas.glob)  # (n_particles, n_features)
        
        # Create RNG
        rng = np.random.default_rng(int(key[0]))

        # Over-sampling can request more simulated components than observed ones.
        # Reuse the observed covariate template cyclically so the model can
        # generate M > K components without changing the external API.
        if n_silos == self.K:
            X_cov = self.X_cov
        else:
            component_ids = np.arange(n_silos) % self.K
            X_cov = self.X_cov[component_ids]
        
        # Compute linear predictor: η_k,i = α_k + X_cov[k,i,:] · β
        # alphas: (n_particles, K, 1)
        # betas: (n_particles, n_features)
        # X_cov: (K, n_obs, n_features)
        
        # Expand dimensions for broadcasting:
        # alphas -> (n_particles, K, 1) 
        # X_cov -> (1, K, n_obs, n_features)
        # betas -> (n_particles, 1, 1, n_features)
        
        X_expanded = X_cov[np.newaxis, :, :, :]  # (1, n_silos, n_obs, n_features)
        betas_expanded = betas[:, np.newaxis, np.newaxis, :]  # (n_particles, 1, 1, n_features)
        alphas_expanded = alphas[:, :, 0]  # (n_particles, n_silos) - extract scalar from last dim
        
        # Compute feature contribution: (n_particles, 1, 1, n_features) * (1, K, n_obs, n_features)
        # Result: (n_particles, K, n_obs, n_features), then sum on last axis -> (n_particles, K, n_obs)
        feature_contribution = np.sum(X_expanded * betas_expanded, axis=-1)  # (n_particles, K, n_obs)
        
        # Add intercepts: (n_particles, n_silos) + (n_particles, n_silos, n_obs)
        eta = alphas_expanded[:, :, np.newaxis] + feature_contribution  # (n_particles, K, n_obs)
        probs = expit(eta)  # (n_particles, K, n_obs)
        
        # Sample binary observations
        zs = rng.binomial(1, probs)  # (n_particles, K, n_obs)
        
        return zs.astype(np.float32)
    
    def distance_matrices_loc(self, zs, y_obs, M=0, L=0):
        """
        Compute local pairwise distance matrices for binary outcomes.
        
        Uses Hamming distance (number of mismatches) as the cost for assigning
        simulated components to observed components.
        
        Parameters
        ----------
        zs : np.ndarray
            Simulated data of shape (n_particles, M, n_obs).
        y_obs : np.ndarray
            Observed data of shape (1, K, n_obs).
        M : int, default=0
            Number of simulated components (defaults to K).
        L : int, default=0
            Number of components to match (defaults to K).
            
        Returns
        -------
        np.ndarray
            Distance matrices of shape (n_particles, K, M) containing
            Hamming distances between observed components (rows) and
            simulated components (columns).
        """
        if M == 0:
            M = self.K
        if L == 0:
            L = self.K
        
        n_particles = zs.shape[0]
        
        # Extract observed data for components 0:K
        y_obs_data = np.asarray(y_obs[0, :self.K, :])  # (K, n_obs)
        
        # Compute pairwise Hamming distances: cost[j,i] = mismatches between
        # observed component j and simulated component i.
        distances = np.zeros((n_particles, self.K, M), dtype=np.float32)
        
        for p in range(n_particles):
            for j in range(self.K):
                for i in range(M):
                    # Hamming distance: number of positions where binary values differ
                    dist = np.sum(np.abs(zs[p, i, :] - y_obs_data[j, :]))
                    distances[p, j, i] = dist
        
        return distances
    
    def distance_global(self, zs, y_obs):
        """
        Compute global distance for parameters.
        
        For this model, we don't have separate global distance components
        beyond the local assignments, so return zeros.
        
        Parameters
        ----------
        zs : np.ndarray
            Simulated data.
        y_obs : np.ndarray
            Observed data.
            
        Returns
        -------
        np.ndarray
            Global distance values (zeros).
        """
        n_particles = zs.shape[0]
        return np.zeros(n_particles, dtype=np.float32)
    
    def distance(self, zs, y_obs):
        """
        Compute total distance between simulated and observed data.
        
        For this model, we use the naive (non-permuted) Hamming distance
        across all K components.
        
        Parameters
        ----------
        zs : np.ndarray
            Simulated data of shape (n_particles, K, n_obs).
        y_obs : np.ndarray
            Observed data of shape (1, K, n_obs).
            
        Returns
        -------
        np.ndarray
            Distance values for each particle.
        """
        # Compute normalized Hamming distance (proportion of mismatches)
        y_obs_data = np.asarray(y_obs[0, :self.K, :])  # (K, n_obs)
        n_particles = zs.shape[0]
        
        distances = np.zeros(n_particles, dtype=np.float32)
        for p in range(n_particles):
            mismatches = np.sum(np.abs(zs[p, :self.K, :] - y_obs_data))
            distances[p] = mismatches / (self.K * self.n_obs)  # Normalize
        
        return distances
    
    def summary(self, z):
        """
        Apply summary transformation to raw data.
        
        For binary data, the summary is the identity (no transformation needed).
        
        Parameters
        ----------
        z : np.ndarray or jnp.ndarray
            Raw simulated observations.
            
        Returns
        -------
        jnp.ndarray
            Same observations (identity transformation).
        """
        return jnp.asarray(z)
    
    def set_X_cov(self, X_cov):
        """
        Update the covariate matrix after initialization.
        
        Useful for testing or when covariates are computed after model creation.
        
        Parameters
        ----------
        X_cov : np.ndarray
            Covariate matrix of shape (K, n_obs, n_features).
        """
        X_cov = np.asarray(X_cov, dtype=np.float32)
        if X_cov.shape != (self.K, self.n_obs, self.n_features):
            raise ValueError(
                f"X_cov shape {X_cov.shape} does not match expected "
                f"(K={self.K}, n_obs={self.n_obs}, n_features={self.n_features})"
            )
        self.X_cov = X_cov
