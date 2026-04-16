"""
Population Monte Carlo (PMC) algorithm for permutation-based ABC inference.

This module implements ABC-PMC which uses importance sampling with adaptive
proposal distributions rather than sequential Monte Carlo.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, multivariate_normal
from scipy.optimize import linear_sum_assignment
from jax import random, jit, vmap
import jax.numpy as jnp
from ..utils.functions import Theta
from jax.scipy.stats import truncnorm
from jax.scipy.special import logsumexp
from tqdm import tqdm
import time
from typing import Tuple, Any


def init_pmc(key: random.PRNGKey, model: Any, n_particles: int, y_obs: np.ndarray, 
             update_weights_distance: bool = True, verbose: int = 1) -> Tuple[Theta, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Initialize Population Monte Carlo algorithm.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with permutation support.
    n_particles : int
        Number of particles to initialize.
    y_obs : numpy.ndarray
        Observed data.
    update_weights_distance : bool, default=True
        Whether to update distance weights.
    verbose : int, default=1
        Verbosity level.
        
    Returns
    -------
    tuple
        (thetas, zs, distance_values, weights, ess_val)
    """
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    
    # Generate initial particles from prior
    thetas = model.prior_generator(key_thetas, n_particles, K)
    zs = model.data_generator(key_zs, thetas)
    
    # Update distance weights if requested
    if update_weights_distance: 
        model.update_weights_distance(zs, verbose)
    
    # Compute initial distances
    distance_values = model.distance(zs, y_obs)
    
    # Initialize uniform weights
    weights = np.ones(n_particles) / n_particles
    ess_val = n_particles
    
    return thetas, zs, distance_values, weights, ess_val


def update_epsilon(alive_distances: np.ndarray, epsilon_target: float, alpha: float) -> float:
    """
    Update epsilon tolerance using quantile-based adaptation.
    
    Parameters
    ----------
    alive_distances : numpy.ndarray
        Current distance values for alive particles.
    epsilon_target : float
        Target epsilon value.
    alpha : float
        Quantile level for epsilon update.
        
    Returns
    -------
    float
        Updated epsilon value.
    """
    return max(np.quantile(alive_distances, alpha), epsilon_target)


def ess(weights: np.ndarray) -> float:
    """
    Compute Effective Sample Size from particle weights.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Particle weights.
        
    Returns
    -------
    float
        Effective sample size.
    """
    # Handle edge cases
    if len(weights) == 0:
        return 0.0
    
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        return 0.0
    
    # Normalize weights to avoid numerical issues
    normalized_weights = weights / weights_sum
    weights_squared_sum = np.sum(normalized_weights**2)
    
    if weights_squared_sum == 0:
        return 0.0
    
    return 1.0 / weights_squared_sum


def move_pmc(key: random.PRNGKey, model: Any, thetas: Theta, weights: np.ndarray, 
             y_obs: np.ndarray, size: int, std_loc: float, std_glob: float, verbose: int = 1) -> Tuple[Theta, np.ndarray, np.ndarray]:
    """
    Generate new particles using importance sampling.
    
    Proposes new particles by sampling from previous generation with 
    truncated normal perturbations for both local and global parameters.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model.
    thetas : Theta
        Current parameter particles.
    weights : numpy.ndarray
        Current particle weights.
    y_obs : numpy.ndarray
        Observed data.
    size : int
        Number of proposals to generate.
    std_loc : float
        Standard deviation for local parameter proposals.
    std_glob : float
        Standard deviation for global parameter proposals.
    verbose : int, default=1
        Verbosity level.
        
    Returns
    -------
    tuple
        (proposed_thetas, proposed_distances, proposed_zs)
    """
    key, key_index, key_loc, key_glob, key_data = random.split(key, 5)
    
    # Ensure weights are normalized and non-zero
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        if verbose > 0:
            print("Warning: All weights are zero, using uniform sampling")
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights_sum
    
    # Sample parent particles based on weights
    indexes = np.array(random.choice(
        key_index, a=thetas.loc.shape[0], shape=(size,), p=weights, replace=True
    ))
    proposed_thetas = thetas[indexes]
    
    # Propose local parameters with truncated normal
    proposed_thetas_loc = random.truncated_normal(
        key_loc, 
        lower=(model.support_par_loc[0, 0] - proposed_thetas.loc) / std_loc, 
        upper=(model.support_par_loc[0, 1] - proposed_thetas.loc) / std_loc, 
        shape=proposed_thetas.loc.shape
    ) * std_loc + proposed_thetas.loc
    
    # Handle infinite values in local parameters
    max_attempts = 1000

    attempt = 0
    while np.isinf(proposed_thetas_loc).any() and attempt < max_attempts:
        if verbose > 1:
            print(f"Warning: Infinite values in proposed local parameters, regenerating... (attempt {attempt+1}/{max_attempts})")
        key, key_loc = random.split(key)
        proposed_thetas_loc = random.truncated_normal(
            key_loc, 
            lower=(model.support_par_loc[0, 0] - proposed_thetas.loc) / std_loc, 
            upper=(model.support_par_loc[0, 1] - proposed_thetas.loc) / std_loc, 
            shape=proposed_thetas.loc.shape
        ) * std_loc + proposed_thetas.loc
        attempt += 1
    
    if np.isinf(proposed_thetas_loc).any():
        if verbose > 0:
            print("Warning: Failed to generate finite local parameters, using original values")
        proposed_thetas_loc = proposed_thetas.loc

    # Propose global parameters with truncated normal
    # Fix: Use correct upper bound for global parameters
    proposed_thetas_glob = random.truncated_normal(
        key_glob, 
        lower=(model.support_par_glob[0, 0] - proposed_thetas.glob) / std_glob, 
        upper=(model.support_par_glob[0, 1] - proposed_thetas.glob) / std_glob,  # Fixed: was using par_loc bounds
        shape=proposed_thetas.glob.shape
    ) * std_glob + proposed_thetas.glob
    
    # Handle infinite values in global parameters
    attempt = 0
    while np.isinf(proposed_thetas_glob).any() and attempt < max_attempts:
        if verbose > 1:
            print(f"Warning: Infinite values in proposed global parameters, regenerating... (attempt {attempt+1}/{max_attempts})")
        key, key_glob = random.split(key)
        proposed_thetas_glob = random.truncated_normal(
            key_glob, 
            lower=(model.support_par_glob[0, 0] - proposed_thetas.glob) / std_glob, 
            upper=(model.support_par_glob[0, 1] - proposed_thetas.glob) / std_glob,  # Fixed: was using par_loc bounds
            shape=proposed_thetas.glob.shape
        ) * std_glob + proposed_thetas.glob
        attempt += 1
    
    if np.isinf(proposed_thetas_glob).any():
        if verbose > 0:
            print("Warning: Failed to generate finite global parameters, using original values")
        proposed_thetas_glob = proposed_thetas.glob
    
    # Create proposed theta object
    proposed_thetas = Theta(loc=proposed_thetas_loc, glob=proposed_thetas_glob)

    # Generate data and compute distances
    proposed_zs = model.data_generator(key_data, proposed_thetas)
    proposed_distances = model.distance(proposed_zs, y_obs)
    
    return proposed_thetas, proposed_distances, proposed_zs


@jit 
def K_t_ij(thetas_t_loc: np.ndarray, thetas_t_glob: np.ndarray, thetas_t_1_loc: np.ndarray, 
           thetas_t_1_glob: np.ndarray, weights_t_1: np.ndarray, std_loc: float, std_glob: float, 
           a_loc: float, b_loc: float, a_glob: float, b_glob: float) -> float:

    """
    Compute log kernel density for single particle pair (i,j).
    
    Calculates the log probability density of transitioning from particle j
    at time t-1 to particle i at time t using truncated normal kernels.
    """
    logpdf_loc = jnp.sum(truncnorm.logpdf(
        x=thetas_t_loc, 
        a=(a_loc - thetas_t_1_loc) / std_loc, 
        b=(b_loc - thetas_t_1_loc) / std_loc, 
        loc=thetas_t_1_loc, 
        scale=std_loc
    ))
    
    logpdf_glob = truncnorm.logpdf(
        x=thetas_t_glob, 
        a=(a_glob - thetas_t_1_glob) / std_glob, 
        b=(b_glob - thetas_t_1_glob) / std_glob, 
        loc=thetas_t_1_glob, 
        scale=std_glob
    )
    
    return logpdf_loc + logpdf_glob + jnp.log(weights_t_1)


@jit
def K_t_i(thetas_t_loc: np.ndarray, thetas_t_glob: np.ndarray, thetas_t_1_loc: np.ndarray, 
          thetas_t_1_glob: np.ndarray, weights_t_1: np.ndarray, std_loc: float, std_glob: float, 
          a_loc: float, b_loc: float, a_glob: float, b_glob: float) -> np.ndarray:
    """
    Compute log kernel density for particle i summed over all j.
    
    Calculates the total log probability density for generating particle i
    from the previous generation using importance sampling.
    """
    res = vmap(K_t_ij, (None, None, 0, 0, 0, None, None, None, None, None, None))(
        thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, 
        std_loc, std_glob, a_loc, b_loc, a_glob, b_glob
    )
    return logsumexp(res, axis=0)


@jit
def K_t(thetas_t_loc: np.ndarray, thetas_t_glob: np.ndarray, thetas_t_1_loc: np.ndarray, 
        thetas_t_1_glob: np.ndarray, weights_t_1: np.ndarray, std_loc: float, std_glob: float, 
        a_loc: float, b_loc: float, a_glob: float, b_glob: float) -> np.ndarray:
    """
    Compute log kernel densities for all particles at time t.
    
    Vectorized computation of kernel densities for importance weight updates.
    """
    return vmap(K_t_i, (0, 0, None, None, None, None, None, None, None, None, None))(
        thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, 
        std_loc, std_glob, a_loc, b_loc, a_glob, b_glob
    )


def update_weights(model: Any, thetas_t: Theta, thetas_t_1: Theta, weights_t_1: np.ndarray, 
                   std_loc: float, std_glob: float, verbose: int = 1) -> np.ndarray:
    """
    Update importance weights for PMC algorithm.
    
    Computes new particle weights using the ratio of prior density to 
    proposal density (kernel density from previous generation).
    
    Parameters
    ----------
    model : object
        Statistical model.
    thetas_t : Theta
        Current generation parameters.
    thetas_t_1 : Theta
        Previous generation parameters.
    weights_t_1 : numpy.ndarray
        Previous generation weights.
    std_loc : float
        Local parameter proposal standard deviation.
    std_glob : float
        Global parameter proposal standard deviation.
    verbose : int, default=1
        Verbosity level.
        
    Returns
    -------
    numpy.ndarray
        Updated particle weights.
    """
    # Debug: Print shapes
    if verbose > 1:
        print(f"Update weights - thetas_t shape: loc={thetas_t.loc.shape}, glob={thetas_t.glob.shape}")
        print(f"Update weights - thetas_t_1 shape: loc={thetas_t_1.loc.shape}, glob={thetas_t_1.glob.shape}")
        print(f"Update weights - weights_t_1 shape: {weights_t_1.shape}")
    
    # Check for empty inputs
    if len(thetas_t.loc) == 0 or len(thetas_t_1.loc) == 0:
        if verbose > 0:
            print("Warning: Empty theta arrays in weight update")
        return np.array([])
    
    # Extract parameter bounds
    a_loc = model.support_par_loc[0, 0]
    b_loc = model.support_par_loc[0, 1]
    a_glob = model.support_par_glob[0, 0]
    b_glob = model.support_par_glob[0, 1]
    
    # Prepare arrays for computation
    std_loc = std_loc.squeeze()
    std_glob = std_glob.squeeze()
    thetas_t_loc, thetas_t_glob = np.array(thetas_t.loc).squeeze(), np.array(thetas_t.glob).squeeze()
    thetas_t_1_loc, thetas_t_1_glob = np.array(thetas_t_1.loc).squeeze(), np.array(thetas_t_1.glob).squeeze()
    weights_t_1 = np.array(weights_t_1).squeeze()
    
    # Ensure weights are normalized
    weights_t_1_sum = np.sum(weights_t_1)
    if weights_t_1_sum > 0:
        weights_t_1 = weights_t_1 / weights_t_1_sum
    else:
        weights_t_1 = np.ones_like(weights_t_1) / len(weights_t_1)
    
    # Compute kernel densities (proposal densities)
    logdenominateur = K_t(
        thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, 
        std_loc, std_glob, a_loc, b_loc, a_glob, b_glob
    )
    
    # Compute prior densities
    logprior = model.prior_logpdf(thetas_t).reshape(-1)
    
    if verbose > 1:
        print(f"Update weights - logdenominateur shape: {logdenominateur.shape}")
        print(f"Update weights - logprior shape: {logprior.shape}")
    
    # Update weights: w ∝ prior / proposal
    # Use log-sum-exp trick for numerical stability
    log_weights = logprior - logdenominateur
    
    # Handle edge cases
    if len(log_weights) == 0:
        if verbose > 0:
            print("Warning: No particles to update weights for")
        return np.array([])
    
    # Handle numerical issues
    if np.any(np.isinf(log_weights)) or np.any(np.isnan(log_weights)):
        if verbose > 0:
            print("Warning: Infinite or NaN weights detected, using uniform weights")
        weights = np.ones(len(log_weights)) / len(log_weights)
    else:
        # Subtract maximum for numerical stability
        log_weights_max = np.max(log_weights)
        weights = np.exp(log_weights - log_weights_max)
        weights_sum = np.sum(weights)
        
        if weights_sum == 0 or np.isnan(weights_sum):
            if verbose > 0:
                print("Warning: Weights sum to zero or NaN, using uniform weights")
            weights = np.ones(len(log_weights)) / len(log_weights)
        else:
            weights = weights / weights_sum
    
    if verbose > 1:
        print(f"Weight update - Log denominator: min={np.min(logdenominateur):.3f}, max={np.max(logdenominateur):.3f}")
        print(f"Weight update - Log prior: min={np.min(logprior):.3f}, max={np.max(logprior):.3f}")
        print(f"Weight update - Weights: min={np.min(weights):.3f}, max={np.max(weights):.3f}")
    
    return weights


def abc_pmc(
    key: random.PRNGKey,
    model: Any,
    n_particles: int,
    epsilon_target: float,
    alpha: float,
    y_obs: np.ndarray,
    epsilon_1: float = np.inf,
    N_sim_max: float = np.inf,
    update_weights_distance: bool = True,
    verbose: int = 1,
    stopping_accept_rate: float = 0.015
) -> dict:
    """
    Population Monte Carlo for ABC inference.
    
    Implements ABC-PMC algorithm that iteratively reduces epsilon tolerance
    while maintaining particle diversity through importance sampling.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model.
    n_particles : int
        Number of particles to maintain.
    epsilon_target : float
        Target tolerance value.
    alpha : float
        Quantile for epsilon reduction.
    y_obs : numpy.ndarray
        Observed data.
    epsilon_1 : float, default=np.inf
        Initial epsilon (estimated if np.inf).
    N_sim_max : int, default=np.inf
        Maximum number of simulations.
    update_weights_distance : bool, default=True
        Whether to update distance weights.
    verbose : int, default=1
        Verbosity level.
    stopping_accept_rate : float, default=0.015
        Minimum acceptance rate before stopping.
        
    Returns
    -------
    dict
        Results dictionary with algorithm evolution.
        
    Notes
    -----
    PMC algorithm:
    1. Start with prior samples
    2. Each iteration: reduce epsilon, generate new particles via importance sampling
    3. Update weights using prior/proposal ratio
    4. Resample if ESS becomes too low
    5. Continue until epsilon_target reached
    """
    K = model.K
    time_0 = time.time()
    
    # Initialize from prior
    key, key_thetas, key_zs = random.split(key, 3)
    new_thetas = model.prior_generator(key_thetas, n_particles)
    new_zs = model.data_generator(key_zs, new_thetas)
    
    if update_weights_distance:
        model.update_weights_distance(new_zs)
    
    # Compute initial distances and weights
    distance_values = model.distance(new_zs, y_obs)
    weights = np.ones(n_particles) / n_particles
    epsilon = np.inf
    ess_val = ess(weights)
    
    # Initialize tracking arrays
    time_init = time.time() - time_0
    Thetas, Weights, Ess, Epsilon = [new_thetas], [weights], [ess_val], [epsilon]
    Dist, Nsim, Zs, Acc_rate, Time = [distance_values], [n_particles * K], [new_zs], [1], [time_init]
    Unique_p, Unique_c = [1.], [1.]
    
    t = 1
    n_sim_total = n_particles * K
    if verbose >= 1:
        print(f"Iteration 0: Epsilon = {epsilon:.3f}, ESS = {ess_val:0.0f} ({ess_val/n_particles:.1%})")
    accept_rate = 1.0  # Initial acceptance rate
    # Main PMC loop
    while epsilon > epsilon_target:
        # Update epsilon with more conservative strategy
        if t == 1 and epsilon_1 != np.inf:
            epsilon = epsilon_1
        else:
            # Use a more conservative epsilon update to avoid too aggressive reduction
            proposed_epsilon = max(np.quantile(distance_values, alpha), epsilon_target)
            
            # Don't reduce epsilon too aggressively - ensure some minimum reduction
            if proposed_epsilon >= epsilon * 0.99:  # Less than 1% reduction
                proposed_epsilon = epsilon * 0.95  # Force 5% reduction
            
            # But don't go below target
            epsilon = max(proposed_epsilon, epsilon_target)
        
        if verbose >= 1:
            print(f"\nIteration {t}: Epsilon = {epsilon:.3f}")
        
        # Store current generation
        zs = new_zs.copy()
        thetas = new_thetas.copy()
        
        # Compute proposal standard deviations with numerical safeguards
        try:
            # Ensure we have enough particles for covariance computation
            if len(weights) != len(thetas.loc):
                if verbose > 0:
                    print(f"Warning: Weight length ({len(weights)}) != theta length ({len(thetas.loc)})")
                # Adjust weights to match
                if len(weights) > len(thetas.loc):
                    weights = weights[:len(thetas.loc)]
                else:
                    # Pad with uniform weights
                    weights = np.append(weights, np.ones(len(thetas.loc) - len(weights)) / len(thetas.loc))
                weights = weights / np.sum(weights)  # Renormalize
            
            if K > 1: 
                # Reshape for covariance computation
                theta_loc_for_cov = thetas.loc.squeeze()
                if theta_loc_for_cov.ndim == 1:
                    theta_loc_for_cov = theta_loc_for_cov.reshape(1, -1)
                
                cov_matrix = np.cov(theta_loc_for_cov.T, aweights=weights)
                
                # Handle scalar case
                if cov_matrix.ndim == 0:
                    cov_matrix = np.array([[cov_matrix]])
                elif cov_matrix.ndim == 1:
                    cov_matrix = np.diag(cov_matrix)
                
                if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                    raise ValueError("Invalid covariance matrix")
                    
                diag_cov = np.diag(cov_matrix)
                if np.any(diag_cov <= 0):
                    diag_cov = np.maximum(diag_cov, 1e-8)  # Ensure positive variance
                std_loc = np.sqrt(2 * diag_cov)[None, :, None]
            else: 
                theta_loc_squeezed = thetas.loc.squeeze()
                if theta_loc_squeezed.ndim == 0:
                    theta_loc_squeezed = np.array([theta_loc_squeezed])
                
                var_loc = np.cov(theta_loc_squeezed, aweights=weights)
                if np.isnan(var_loc) or np.isinf(var_loc) or var_loc <= 0:
                    var_loc = 1e-8
                std_loc = np.sqrt(2 * var_loc)
            
            theta_glob_squeezed = thetas.glob.squeeze()
            if theta_glob_squeezed.ndim == 0:
                theta_glob_squeezed = np.array([theta_glob_squeezed])
                
            var_glob = np.cov(theta_glob_squeezed, aweights=weights)
            if np.isnan(var_glob) or np.isinf(var_glob) or var_glob <= 0:
                var_glob = 1e-8
            std_glob = np.sqrt(2 * var_glob)
            
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as e:
            if verbose > 0:
                print(f"Warning: Error computing covariance, using fallback std: {e}")
            # Use fallback standard deviations
            std_loc = 0.1 * np.ones_like(thetas.loc[0:1])
            std_glob = 0.1

        if verbose > 1:
            print(f"Proposal std - Local: min={np.min(std_loc):.3f}, max={np.max(std_loc):.3f}")
            print(f"Proposal std - Global: {std_glob:.3f}")
        
        # Generate new particle generation
        new_thetas = Theta()
        new_zs = np.empty((0, zs.shape[1], zs.shape[2]))
        distance_values = np.empty(0)
        n_accept = 0
        n_sim = 0
        
        # Importance sampling until we have n_particles accepted
        batch_size = max(1, int(n_particles * accept_rate))  # Smaller initial batch size
        accept_rate = 1.0  # Reset acceptance rate for this iteration
        while n_accept < n_particles:
            key, key_move = random.split(key)
            
            # Adaptive batch size - start small and increase if acceptance rate is good
            current_batch_size = min(batch_size, n_particles - n_accept)
            
            # Propose new particles
            proposed_thetas, proposed_distances, proposed_zs = move_pmc(
                key=key_move, model=model, thetas=thetas, weights=weights, y_obs=y_obs, 
                size=int(current_batch_size), std_loc=std_loc, std_glob=std_glob, verbose=verbose
            )
            
            # Accept particles within tolerance
            accept = np.where(proposed_distances < epsilon)[0]
            
            if len(accept) > 0:
                new_thetas = new_thetas.append(proposed_thetas[accept])
                distance_values = np.append(distance_values, proposed_distances[accept])
                new_zs = np.append(new_zs, proposed_zs[accept], axis=0)

            n_sim += current_batch_size
            n_accept += len(accept)
            
            
            # If we're not getting any accepts, relax epsilon slightly
            # if attempt > 20 and n_accept == 0:
            #     epsilon_relaxed = epsilon * 1.1  # Relax by 10%
            #     if verbose >= 1:
            #         print(f"No particles accepted after {attempt} attempts, relaxing epsilon to {epsilon_relaxed:.3f}")
            #     epsilon = epsilon_relaxed
            
            # if verbose > 1 and attempt % 10 == 0:
            #     print(f"Attempt {attempt}: {n_accept}/{n_particles} particles accepted, current acceptance rate: {n_accept/n_sim:.3f}")
        
        # if attempt >= max_attempts:
        #     if verbose >= 1:
        #         print(f"Maximum attempts reached ({max_attempts}), stopping with {n_accept} particles")
        #     if n_accept == 0:
        #         if verbose >= 1:
        #             print("No particles accepted, terminating algorithm")
        #         break
        #     elif n_accept < n_particles // 2:
        #         if verbose >= 1:
        #             print(f"Very few particles accepted ({n_accept}/{n_particles}), consider relaxing epsilon_target")
        #         # Use what we have
        #         n_particles = n_accept
        
        # Compute acceptance rate
        accept_rate = n_accept/ n_sim
        
        if verbose >= 1:
            print(f"Simulations: {n_sim} (Accept rate: {accept_rate:.1%})")
        
        # Check stopping criteria
        if accept_rate < stopping_accept_rate or np.sum(Nsim) > N_sim_max: 
            if verbose >= 1:
                print("Stopping: Accept rate too low or simulation limit reached")
            epsilon_target = epsilon
        
        # Check if we have enough particles
        if len(new_thetas.loc) == 0:
            if verbose >= 1:
                print("No particles accepted, using previous generation")
            # Use previous generation as final result
            break
        
        # Ensure we have at least some particles to work with
        actual_n_particles = min(len(new_thetas.loc), n_particles)
      
        
        # Truncate to exact number of particles - be careful with Theta slicing
        if len(new_thetas.loc) > n_particles:
            new_thetas = Theta(
                loc=new_thetas.loc[:n_particles],
                glob=new_thetas.glob[:n_particles]
            )
        if len(new_zs) > n_particles:
            new_zs = new_zs[:n_particles]
        if len(distance_values) > n_particles:
            distance_values = distance_values[:n_particles]
        
        # Verify we have valid shapes
        if len(new_thetas.loc) == 0 or len(new_thetas.glob) == 0:
            if verbose >= 1:
                print("Invalid theta shapes after truncation, stopping")
            break
        
        # Update importance weights
        if verbose > 1:
            print("Updating weights...")
        weights = update_weights(model, new_thetas, thetas, weights, std_loc, std_glob, verbose)
        weights = np.array(weights)
        
        # Handle empty weights
        if len(weights) == 0:
            if verbose >= 1:
                print("No valid weights computed, using uniform weights")
            weights = np.ones(n_particles) / n_particles
            
        ess_val = ess(weights)
        
        if verbose >= 1:
            print(f"ESS = {ess_val:0.0f} ({ess_val/n_particles:.1%})")
        
        # Resample if ESS too low
        if ess_val < n_particles / 2:
            if verbose >= 1:
                print("Resampling...")
            
            key, key_index = random.split(key)
            index = random.choice(
                key_index, a=np.arange(n_particles), shape=(n_particles,), p=weights, replace=True
            )
            index = np.array(index, dtype=np.int64)
            
            # Apply resampling
            new_thetas = new_thetas[index]
            new_zs = new_zs[index]
            distance_values = distance_values[index]
            weights = np.ones(n_particles) / n_particles
            ess_val = ess(weights)
            
            if verbose > 1:
                unique_count = len(np.unique(index))
                print(f"Unique particles after resampling: {unique_count}")
        
        # Compute diagnostics
        unique_p = len(np.unique(new_thetas.loc, axis=0)) / n_particles
        unique_c = len(np.unique(new_thetas.glob, axis=0)) / n_particles
        
        # Store results
        Weights.append(weights)
        Ess.append(ess_val)
        Epsilon = np.append(Epsilon, epsilon)
        Thetas.append(new_thetas)
        Dist.append(distance_values)
        Zs.append(new_zs)
        Acc_rate.append(accept_rate)
        Nsim.append(n_sim * K)
        Time.append(time.time() - time_0)
        Unique_p.append(unique_p)
        Unique_c.append(unique_c)
        
        t += 1
    
    # Compile results
    results = {
        "Thetas": Thetas, "Zs": Zs, "Weights": Weights, "Ess": Ess, 
        "Eps_values": Epsilon, "Dist": Dist, "N_sim": Nsim, "Time": Time, 
        "unique_part": Unique_p, "unique_comp": Unique_c, "Acc_rate": Acc_rate, 
        "time_final": time.time() - time_0
    }
    
    return results