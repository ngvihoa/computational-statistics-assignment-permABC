"""
SIR epidemic models for permutation-enhanced ABC.

This module implements various SIR (Susceptible-Infected-Recovered) epidemic
models with different parameter structures. These models are particularly
challenging for ABC methods due to their complex dynamics and parameter
identifiability issues.
"""

import jax.numpy as jnp
from jax import random, jit, vmap, lax
import jax
import numpy as np
from functools import partial

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

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Numba backend (~30x faster than JAX on multi-core CPU for SIR loops)
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── Numba SIR simulator ────────────────────────────────────────────────────

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _simulate_sir_numba(S0, I0, R0, beta, gamma, n_pop, n_obs, dt,
                            noise, steps_per_day):
        """Numba-parallel SIR simulation.

        Parameters: all numpy arrays, shapes (N, K) for states/rates,
        noise shape (N, K, n_obs, steps_per_day).
        Returns I_traj of shape (N, K, n_obs).
        """
        N, K = S0.shape
        I_traj = np.empty((N, K, n_obs), dtype=np.float64)

        for p in prange(N):
            for k in range(K):
                S = S0[p, k]
                I = I0[p, k]
                R = R0[p, k]
                for t in range(n_obs):
                    for s in range(steps_per_day):
                        S_safe = max(S, 0.0)
                        I_safe = max(I, 0.0)
                        inf_rate = beta[p, k] * S_safe * I_safe / n_pop * dt
                        new_inf = min(inf_rate * noise[p, k, t, s], S_safe)
                        rec_rate = gamma[p, k] * I_safe * dt
                        new_rec = min(rec_rate, I_safe)
                        I = max(0.0, I_safe + new_inf - new_rec)
                        R = max(0.0, R + new_rec)
                        S = max(0.0, n_pop - I - R)
                    I_traj[p, k, t] = I
        return I_traj


def simulate_sir_numba(S0, I0, R0, beta, gamma, n_pop, n_obs,
                       noise_key, sigma=0.05, steps_per_day=10):
    """Wrapper: generate JAX noise then call Numba kernel. Returns numpy array."""
    dt = 1.0 / steps_per_day
    S0_np = np.asarray(S0, dtype=np.float64)
    I0_np = np.asarray(I0, dtype=np.float64)
    R0_np = np.asarray(R0, dtype=np.float64)
    beta_np = np.asarray(beta, dtype=np.float64)
    gamma_np = np.asarray(gamma, dtype=np.float64)

    N, K = S0_np.shape
    if noise_key is not None and sigma > 0:
        noise = np.asarray(
            jnp.exp(sigma * random.normal(noise_key, (N, K, n_obs, steps_per_day))),
            dtype=np.float64,
        )
    else:
        noise = np.ones((N, K, n_obs, steps_per_day), dtype=np.float64)

    return _simulate_sir_numba(S0_np, I0_np, R0_np, beta_np, gamma_np,
                               float(n_pop), n_obs, dt, noise, steps_per_day)


# ── JAX SIR simulator (fallback) ───────────────────────────────────────────

@partial(jit, static_argnames=['n_obs', 'steps_per_day', 'sigma'])
def simulate_sir_jax(S0, I0, R0, beta, gamma, n_pop, n_obs, dt=0.1,
                     noise_key=None, sigma=0.05, steps_per_day=10):
    """JAX implementation of SIR simulation (used when Numba unavailable)."""
    n_particles, n_regions = S0.shape
    dtype = S0.dtype

    if noise_key is not None and sigma > 0:
        noise_shape = (n_particles, n_regions, n_obs, steps_per_day)
        noise = jnp.exp(sigma * random.normal(noise_key, noise_shape, dtype=dtype))
    else:
        noise = jnp.ones((n_particles, n_regions, n_obs, steps_per_day), dtype=dtype)

    def simulate_particle(particle_idx):
        def simulate_day(state, day_idx):
            S, I, R = state
            def substep(carry, step_idx):
                S_curr, I_curr, R_curr = carry
                S_safe = jnp.maximum(S_curr, 0.0)
                I_safe = jnp.maximum(I_curr, 0.0)
                inf_rate = beta[particle_idx] * S_safe * I_safe / n_pop * dt
                noise_factor = noise[particle_idx, :, day_idx, step_idx]
                new_infections = jnp.minimum(inf_rate * noise_factor, S_safe)
                recovery_rate = gamma[particle_idx] * I_safe * dt
                new_recoveries = jnp.minimum(recovery_rate, I_safe)
                I_new = jnp.maximum(0.0, I_safe + new_infections - new_recoveries)
                R_new = jnp.maximum(0.0, R_curr + new_recoveries)
                S_new = jnp.maximum(0.0, n_pop - I_new - R_new)
                return (S_new, I_new, R_new), I_new
            final_state, _ = lax.scan(substep, (S, I, R), jnp.arange(steps_per_day))
            return final_state, final_state[1]
        initial_state = (S0[particle_idx], I0[particle_idx], R0[particle_idx])
        _, I_trajectory = lax.scan(simulate_day, initial_state, jnp.arange(n_obs))
        return I_trajectory

    I_trajectories = vmap(simulate_particle)(jnp.arange(n_particles))
    return jnp.transpose(I_trajectories, (0, 2, 1))


class SIRWithKnownInit(ModelBase):
    """
    SIR model with known initial conditions.
    
    This model assumes:
    - Initial conditions (S₀, I₀, R₀) are known and fixed
    - β_k ~ Uniform(low_β, high_β) transmission rates per region
    - R₀ ~ Uniform(low_R₀, high_R₀) global basic reproduction number
    - γ_k = β_k / R₀ recovery rates (derived)
    
    The model is useful for studying regional variation in transmission
    when the epidemic's initial state is well-known.
    
    Parameters
    ----------
    K : int
        Number of regions/populations.
    weights_distance : array-like, optional
        Distance weights for regions.
    n_obs : int, default=100
        Number of daily observations.
    n_pop : float, default=1000
        Population size per region.
    low_beta, high_beta : float
        Range for transmission rate β.
    low_r0, high_r0 : float
        Range for basic reproduction number R₀.
    I0, R0 : float
        Fixed initial infected and recovered counts.
    """
    
    def __init__(self, K, weights_distance=None, n_obs=100, n_pop=1000, 
                 low_beta=1e-8, high_beta=5, low_r0=1e-8, high_r0=5, I0=100, R0=100):
        """Initialize SIR model with known initial conditions."""
        super().__init__(K, weights_distance)
        
        # Model parameters
        self.n_obs = n_obs
        self.n_pop = n_pop
        self.I0 = I0
        self.R0 = R0

        # Parameter support ranges
        self.support_par_loc = jnp.array([[low_beta, high_beta]])   # β_k ranges
        self.support_par_glob = jnp.array([[low_r0, high_r0]])     # R₀ range
        
        # Parameter names for plotting
        self.loc_name = ["$\\beta_{"]
        self.glob_name = ["$R_0$"]
        self.dim_loc = 1  # β is scalar per region
        self.dim_glob = 1  # R₀ is single global parameter

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate samples from the prior distribution.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        n_particles : int
            Number of particles to generate.
        n_silos : int, default=0
            Number of regions (if 0, defaults to K).
            
        Returns
        -------
        Theta
            Parameter samples with:
            - loc: Transmission rates β_k of shape (n_particles, n_silos, 1)
            - glob: Basic reproduction number R₀ of shape (n_particles, 1)
        """
        if n_silos == 0:
            n_silos = self.K
            
        key, key_beta, key_r0 = random.split(key, 3)

        # Sample transmission rates: β_k ~ Uniform(low_β, high_β)
        betas = random.uniform(
            key_beta, 
            shape=(n_particles, n_silos, 1), 
            minval=self.support_par_loc[0, 0], 
            maxval=self.support_par_loc[0, 1]
        )
        
        # Sample basic reproduction number: R₀ ~ Uniform(low_R₀, high_R₀)
        r0 = random.uniform(
            key_r0, 
            shape=(n_particles, 1), 
            minval=self.support_par_glob[0, 0], 
            maxval=self.support_par_glob[0, 1]
        )

        return Theta(loc=betas, glob=r0)

    def data_generator(self, key, thetas):
        """
        Generate epidemic simulations.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas : Theta
            Parameter values for simulation.
            
        Returns
        -------
        np.ndarray
            Simulated infected counts of shape (n_particles, n_regions, n_obs).
        """
        # Extract parameters
        betas = thetas.loc[:, :, 0]          # Transmission rates
        gammas = betas / thetas.glob         # Recovery rates: γ = β / R₀
        
        # Set initial conditions (known)
        S0 = jnp.full_like(betas, self.n_pop - self.I0 - self.R0)
        I0_array = jnp.full_like(betas, self.I0)
        R0_array = jnp.full_like(betas, self.R0)

        # Simulate epidemic
        if _HAS_NUMBA:
            result = simulate_sir_numba(
                S0, I0_array, R0_array, betas, gammas,
                n_pop=self.n_pop, n_obs=self.n_obs, noise_key=key,
            )
        else:
            result = simulate_sir_jax(
                S0, I0_array, R0_array, betas, gammas,
                n_pop=self.n_pop, n_obs=self.n_obs, noise_key=key,
            )

        return np.array(result)

    def prior_logpdf(self, thetas):
        """
        Compute log probability density of the prior.

        For uniform priors, this is constant within support.
        
        Parameters
        ----------
        thetas : Theta
            Parameter values to evaluate.
            
        Returns
        -------
        np.ndarray
            Log probability densities (constant for uniform priors).
        """
        n_particles = thetas.loc.shape[0]
        n_regions = thetas.loc.shape[1]
        
        # Log density for uniform distributions
        log_beta_density = -jnp.log(self.support_par_loc[0, 1] - self.support_par_loc[0, 0])
        log_r0_density = -jnp.log(self.support_par_glob[0, 1] - self.support_par_glob[0, 0])
        
        # Total log density: sum over all β_k plus R₀
        total_log_density = n_regions * log_beta_density + log_r0_density
        
        return jnp.full(n_particles, total_log_density)


class SIRWithUnknownInit(ModelBase):
    """
    SIR model with unknown initial conditions.
    
    This more complex model treats initial conditions as unknown parameters:
    - I₀_k ~ Uniform(low_I, high_I) initial infected per region
    - R₀_k ~ Uniform(low_R, high_R) initial recovered per region  
    - β_k ~ Uniform(low_β, high_β) transmission rates per region
    - R₀ ~ Uniform(low_R₀, high_R₀) global basic reproduction number
    - γ_k = β_k / R₀ recovery rates (derived)
    
    This model is more realistic but also more challenging for inference
    due to the increased parameter dimensionality and identifiability issues.
    
    Parameters
    ----------
    K : int
        Number of regions/populations.
    weights_distance : array-like, optional
        Distance weights for regions.
    n_obs : int, default=100
        Number of daily observations.
    n_pop : float, default=1000
        Population size per region.
    low_I, high_I : float
        Range for initial infected counts.
    low_R, high_R : float
        Range for initial recovered counts.
    low_beta, high_beta : float
        Range for transmission rates.
    low_r0, high_r0 : float
        Range for basic reproduction number.
    """
    
    def __init__(self, K, weights_distance=None, n_obs=100, n_pop=1000, 
                 low_I=1e-8, high_I=1000, low_R=1e-8, high_R=1000, 
                 low_beta=1e-8, high_beta=5, low_r0=1e-8, high_r0=5):
        """Initialize SIR model with unknown initial conditions."""
        super().__init__(K, weights_distance)
        
        # Model parameters
        self.n_obs = n_obs
        self.n_pop = n_pop

        # Parameter support ranges
        self.support_par_loc = jnp.array([
            [low_I, high_I],        # I₀_k ranges
            [low_R, high_R],        # R₀_k ranges  
            [low_beta, high_beta]   # β_k ranges
        ])
        self.support_par_glob = jnp.array([[low_r0, high_r0]])  # R₀ range
        
        # Parameter names and dimensions
        self.loc_name = ["$I^0_{", "$R^0_{", "$\\beta_{"]
        self.glob_name = ["$R_0$"]
        self.dim_loc = 3  # Three local parameters per region
        self.dim_glob = 1  # One global parameter

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate samples from the prior distribution.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        n_particles : int
            Number of particles to generate.
        n_silos : int, default=0
            Number of regions (if 0, defaults to K).
            
        Returns
        -------
        Theta
            Parameter samples with:
            - loc: [I₀_k, R₀_k, β_k] of shape (n_particles, n_silos, 3)
            - glob: R₀ of shape (n_particles, 1)
        """
        if n_silos == 0: 
            n_silos = self.K
            
        key, key_loc, key_glob = random.split(key, 3)
        
        # Sample local parameters uniformly within support
        loc = random.uniform(
            key_loc, 
            shape=(n_particles, n_silos, self.support_par_loc.shape[0]), 
            minval=self.support_par_loc[:, 0], 
            maxval=self.support_par_loc[:, 1]
        )
        
        # Sample global parameter uniformly within support
        glob = random.uniform(
            key_glob, 
            shape=(n_particles, self.support_par_glob.shape[0]), 
            minval=self.support_par_glob[:, 0], 
            maxval=self.support_par_glob[:, 1]
        )
        
        return Theta(loc=loc, glob=glob)

    def data_generator(self, key, thetas):
        """
        Generate epidemic simulations with unknown initial conditions.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas : Theta
            Parameter values for simulation.
            
        Returns
        -------
        np.ndarray
            Simulated infected counts of shape (n_particles, n_regions, n_obs).
        """
        # Extract parameters
        I0 = thetas.loc[:, :, 0]        # Initial infected
        R0_init = thetas.loc[:, :, 1]   # Initial recovered  
        betas = thetas.loc[:, :, 2]     # Transmission rates
        gammas = betas / thetas.glob    # Recovery rates: γ = β / R₀
        
        # Compute initial susceptible (ensuring non-negative)
        S0 = jnp.maximum(0.0, self.n_pop - I0 - R0_init)
        
        # Simulate epidemic
        if _HAS_NUMBA:
            result = simulate_sir_numba(
                S0, I0, R0_init, betas, gammas,
                n_pop=self.n_pop, n_obs=self.n_obs, noise_key=key,
            )
        else:
            result = simulate_sir_jax(
                S0, I0, R0_init, betas, gammas,
                n_pop=self.n_pop, n_obs=self.n_obs, noise_key=key,
            )

        return np.array(result)

    def prior_logpdf(self, thetas):
        """
        Compute log probability density of the prior.
        
        For uniform priors, this is constant within support.
        
        Parameters
        ----------
        thetas : Theta
            Parameter values to evaluate.
            
        Returns
        -------
        np.ndarray
            Log probability densities.
        """
        # For uniform priors, return constant log density
        # In practice, you might want to check parameter bounds here
        n_particles = thetas.loc.shape[0]
        return jnp.zeros(n_particles)

class SIR_real_world(ModelBase):
    """
    SIR model for real-world data analysis.
    
    This model uses a specific parameterization suitable for epidemiological inference:
    - R0 is a global parameter, representing the basic reproduction number.
    - gamma_k are local parameters, representing the recovery rate for each region k.
    - beta_k are derived parameters: beta_k = R0 * gamma_k.
    - Initial conditions (I0_k, R0_k) are also treated as local parameters.
    """
    def __init__(self, K, weights_distance=None, n_obs=100, n_pop=100000,
                 low_I=.1, high_I=100,
                 low_R=0.0001, high_R=50,
                 low_gamma=.0001, high_gamma=0.2,
                 low_r0=0.0001, high_r0=1.5,
                 sigma=0.05):

        # Correctly call the parent class's __init__ method
        super().__init__(K, weights_distance)
        self.n_obs = n_obs
        self.n_pop = n_pop
        self.sigma = sigma

        # Define the parameter support ranges
        self.support_par_loc = jnp.array([
            [low_I, high_I],        # I0_k (Initial Infected)
            [low_R, high_R],        # R0_k (Initial Recovered)
            [low_gamma, high_gamma] # gamma_k (Recovery Rate)
        ])
        self.support_par_glob = jnp.array([
            [low_r0, high_r0]       # R0_global (Basic Reproduction Number)
        ])
        
        # Define parameter names and dimensions for clarity
        self.loc_name = ["$I^0_{", "$R^0_{", "$\\gamma_{"]
        self.glob_name = ["$R_0$"]
        self.dim_loc = 3
        self.dim_glob = 1

    def prior_generator(self, key, n_particles, n_silos=0):
        """Generates samples from the uniform prior distributions."""
        if n_silos == 0:
            n_silos = self.K
            
        key, key_loc, key_glob = random.split(key, 3)
        
        loc = random.uniform(
            key_loc, shape=(n_particles, n_silos, self.dim_loc),
            minval=self.support_par_loc[:, 0], 
            maxval=self.support_par_loc[:, 1]
        )
        glob = random.uniform(
            key_glob, shape=(n_particles, self.dim_glob),
            minval=self.support_par_glob[:, 0], 
            maxval=self.support_par_glob[:, 1]
        )
        return Theta(loc=loc, glob=glob)

    def data_generator(self, key, thetas: Theta):
        """Generates epidemic data from the given parameters."""
        I0 = thetas.loc[:, :, 0]
        R0_init = thetas.loc[:, :, 1]
        gamma = thetas.loc[:, :, 2]
        R0_global = thetas.glob

        beta = R0_global * gamma
        S0 = jnp.maximum(0.0, self.n_pop - I0 - R0_init)

        if _HAS_NUMBA:
            result = simulate_sir_numba(
                S0, I0, R0_init, beta, gamma,
                n_pop=self.n_pop, n_obs=self.n_obs,
                noise_key=key, sigma=self.sigma,
            )
        else:
            result = simulate_sir_jax(
                S0, I0, R0_init, beta, gamma,
                n_pop=self.n_pop, n_obs=self.n_obs,
                noise_key=key, sigma=self.sigma,
            )
        return np.array(result)

    def prior_logpdf(self, thetas: Theta):
        """Computes the log probability of the priors."""
        n_particles = thetas.loc.shape[0]
        n_regions = thetas.loc.shape[1]

        # Calculate log density for uniform distributions
        # log(1 / (max - min)) = -log(max - min)
        log_density_loc = -jnp.sum(jnp.log(self.support_par_loc[:, 1] - self.support_par_loc[:, 0]))
        log_density_glob = -jnp.sum(jnp.log(self.support_par_glob[:, 1] - self.support_par_glob[:, 0]))
        
        # Total log density is sum over all local parameters + global parameters
        total_log_density = n_regions * log_density_loc + log_density_glob
        
        return jnp.full(n_particles, total_log_density)


# Export key classes
__all__ = [
    'SIRWithKnownInit',
    'SIRWithUnknownInit', 
    'simulate_sir_jax',
    'SIR_real_world'
]