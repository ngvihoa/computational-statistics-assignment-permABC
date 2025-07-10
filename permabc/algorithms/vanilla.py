"""
Vanilla ABC algorithms with and without permutation optimization.

This module implements the basic ABC rejection sampling algorithm in both
standard and permutation-enhanced versions. These are the simplest ABC methods
that generate samples directly from the prior until acceptance.
"""
from typing import Callable, Tuple, Any
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.lax import while_loop
from ..utils.functions import Theta  # Fixed relative import
from ..core.distances import optimal_index_distance  # Fixed relative import
import numpy as np


def vanilla_single(key: random.PRNGKey, prior_simulator: Callable, data_simulator: Callable, 
                   discrepancy: Callable, epsilon: float, true_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:    
    """
    Generate a single accepted particle using ABC rejection sampling.
    
    This is the core function for vanilla ABC that repeatedly samples from
    the prior and simulates data until the distance criterion is met.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    prior_simulator : callable
        Function to generate parameters from prior: (key, n_samples) -> (theta_loc, theta_glob).
    data_simulator : callable
        Function to simulate data: (key, theta_loc, theta_glob) -> simulated_data.
    discrepancy : callable
        Distance function: (sim_data, obs_data) -> distance.
    epsilon : float
        ABC acceptance threshold.
    true_data : numpy.ndarray
        Observed data for comparison.
        
    Returns
    -------
    data : numpy.ndarray
        Accepted simulated data.
    theta_loc : numpy.ndarray
        Accepted local parameters.
    theta_glob : numpy.ndarray
        Accepted global parameters.
    dist : float
        Distance value for accepted particle.
    n_sim : int
        Number of simulations performed to get acceptance.
        
    Notes
    -----
    Uses JAX's while_loop for efficient compilation and execution.
    The loop continues until distance < epsilon.
    
    This function is JIT-compiled with static arguments for performance.
    """
    def cond_fun(val):
        """Continue while distance >= epsilon."""
        _, _, _, _, dist, _ = val
        return dist >= epsilon
    
    def body_fun(val):
        """One iteration of proposal-simulate-evaluate."""
        key, z, theta_loc, theta_glob, dist, n_sim = val
        key, key_theta, key_data = random.split(key, 3)
        
        # Sample from prior
        theta_prop_loc, thetas_prop_glob = prior_simulator(key_theta, 1)
        
        # Simulate data
        data_prop = data_simulator(key_data, theta_prop_loc, thetas_prop_glob)
        
        # Compute distance
        dist = jnp.reshape(discrepancy(data_prop, true_data), ())
        
        return key, data_prop, theta_prop_loc, thetas_prop_glob, dist, n_sim + 1
    
    # Initialize with values that ensure at least one iteration
    key, key_theta = random.split(key)
    fake_theta_loc, fake_theta_glob = prior_simulator(key_theta, 1)
    fake_data = jnp.zeros_like(true_data).astype(float).reshape(true_data.shape)
    n_sim = 0
    
    # Run rejection loop
    key, data, theta_loc, theta_glob, dist, n_sim = while_loop(
        cond_fun, body_fun, 
        (key, fake_data, fake_theta_loc, fake_theta_glob, epsilon + 1, n_sim)
    )
    
    return data, theta_loc, theta_glob, dist, n_sim


# JIT compile with static arguments for performance
vanilla_single = jit(vanilla_single, static_argnums=(1, 2, 3))


def abc_vanilla(key: random.PRNGKey, model: Any, n_points: int, epsilon: float, y_obs: np.ndarray) -> Tuple[np.ndarray, Theta, np.ndarray, int]:
    """
    Standard ABC rejection sampling algorithm.
    
    Generates n_points accepted particles by repeatedly sampling from the prior
    and accepting only those with distance < epsilon. This is the most basic
    ABC algorithm, suitable for simple problems or as a baseline.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with prior_generator_jax, data_generator_jax, and distance methods.
    n_points : int
        Number of particles to generate.
    epsilon : float
        ABC acceptance threshold. Use np.inf for prior predictive sampling.
    y_obs : numpy.ndarray
        Observed data for distance computation.
        
    Returns
    -------
    datas : numpy.ndarray
        Accepted simulated datasets, shape (n_points, ...).
    thetas : Theta
        Accepted parameter particles.
    dists : numpy.ndarray
        Distance values for accepted particles.
    total_simulations : int
        Total number of simulations performed.
        
    Notes
    -----
    Algorithm steps:
    1. For each required particle:
       - Sample parameters from prior
       - Simulate data
       - Compute distance
       - Accept if distance < epsilon, otherwise repeat
    2. Return all accepted particles
    
    Efficiency considerations:
    - Very inefficient for small epsilon (high rejection rate)
    - Use ABC-SMC for better efficiency in challenging problems
    - Set epsilon=np.inf for prior predictive sampling (no rejection)
    
    Examples
    --------
    >>> key = random.PRNGKey(42)
    >>> datas, thetas, dists, n_sim = abc_vanilla(key, model, 1000, 0.1, y_obs)
    >>> print(f"Generated {len(datas)} particles with {n_sim} simulations")
    """
    def prior_predictive(key, prior_simulator, data_simulator, discrepancy, true_data):
        """
        Single sample from prior predictive (no rejection).
        
        Used when epsilon=np.inf to sample directly from prior predictive
        distribution without any ABC rejection.
        """
        key, key_theta, key_data = random.split(key, 3)
        theta_loc, theta_glob = prior_simulator(key_theta, 1)
        data = data_simulator(key_data, theta_loc, theta_glob)
        dist = discrepancy(data, true_data)
        return data[0], theta_loc[0], theta_glob, dist, 1
    
    # Generate independent random keys for each particle
    keys = random.split(key, n_points)   
    
    if epsilon != np.inf:
        # ABC rejection sampling
        datas, thetas_loc, thetas_glob, dists, n_sim = vmap(
            vanilla_single, 
            in_axes=(0, None, None, None, None, None)
        )(keys, model.prior_generator_jax, model.data_generator_jax, 
          model.distance, epsilon, y_obs)
    else: 
        # Prior predictive sampling (no rejection)
        datas, thetas_loc, thetas_glob, dists, n_sim = vmap(
            prior_predictive, 
            in_axes=(0, None, None, None, None)
        )(keys, model.prior_generator_jax, model.data_generator_jax, 
          model.distance, y_obs)
    
    # Construct Theta object
    thetas = Theta(loc=thetas_loc, glob=thetas_glob)
    
    return datas, thetas, dists, np.sum(n_sim)


def perm_abc_vanilla(key: random.PRNGKey, model: Any, n_points: int, epsilon: float, y_obs: np.ndarray) -> Tuple[np.ndarray, Theta, np.ndarray, int]:
    """
    Permutation-enhanced ABC rejection sampling algorithm.
    
    Combines standard ABC rejection sampling with optimal permutation matching
    to improve parameter inference in problems with label switching or
    component matching issues.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with permutation support.
    n_points : int
        Number of particles to generate.
    epsilon : float
        ABC acceptance threshold for initial sampling.
    y_obs : numpy.ndarray
        Observed data for distance computation.
        
    Returns
    -------
    datas_perm : numpy.ndarray
        Permuted simulated datasets for optimal matching.
    thetas_perm : Theta
        Permuted parameter particles.
    dists_perm : numpy.ndarray
        Optimal distances after permutation matching.
    total_simulations : int
        Total number of simulations performed.
        
    Notes
    -----
    Algorithm steps:
    1. Standard ABC rejection sampling to get n_points particles
    2. Optimal permutation matching using Linear Sum Assignment
    3. Apply permutations to both data and parameters
    4. Return optimally matched particles

    Examples
    --------
    >>> key = random.PRNGKey(42)
    >>> # For a 3-component Gaussian mixture
    >>> datas, thetas, dists, n_sim = perm_abc_vanilla(key, model, 1000, 0.1, y_obs)
    >>> print(f"Optimal matching reduced distances by {np.mean(original_dists - dists):.3f}")
    """
    # Step 1: Standard ABC rejection sampling
    key, subkey = random.split(key)
    datas, thetas, _, n_sim = abc_vanilla(subkey, model, n_points, epsilon, y_obs)
    
    # Step 2: Optimal permutation matching
    # Note: Using epsilon=0 forces full optimization for all particles
    dists_perm, _, zs_index, _ = optimal_index_distance(
        model=model, 
        zs=datas, 
        y_obs=y_obs, 
        epsilon=0,  # Force optimization for all particles
        verbose=2
    )
    
    # Step 3: Apply permutations
    # Permute data according to optimal assignments
    datas_perm = datas[np.arange(n_points)[:, None], zs_index]
    
    # Permute parameters according to optimal assignments
    thetas_perm = thetas.apply_permutation(zs_index)
    
    return datas_perm, thetas_perm, dists_perm, np.sum(n_sim)

