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
from ..utils.functions import Theta
from ..assignment import optimal_index_distance, compute_total_distance, do_swap, do_hilbert
from ..assignment.solvers.lsa import solve_lsa
from ..assignment.solvers.sinkhorn import sinkhorn_assignment
import numpy as np
from typing import Optional


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


def _sample_prior_batch(key, model, n, y_obs):
    """Sample n particles from the prior and simulate data (no rejection).

    Uses the model's native batch APIs directly instead of vmap over
    individual samples, which avoids JAX tracing overhead and is
    significantly faster for large n.

    Returns NumPy arrays to avoid repeated JAX->NumPy conversions downstream.
    """
    key1, key2 = random.split(key)
    theta_loc, theta_glob = model.prior_generator_jax(key1, n)
    datas = model.data_generator_jax(key2, theta_loc, theta_glob)
    return (
        np.asarray(datas),
        np.asarray(theta_loc),
        np.asarray(theta_glob),
        n,
    )


def perm_abc_vanilla(
    key: random.PRNGKey,
    model: Any,
    n_points: int,
    epsilon: float,
    y_obs: np.ndarray,
    try_hilbert: bool = False,
    try_sinkhorn: bool = False,
    try_swaps: bool = False,
    try_lsa: bool = True,
    alpha_hint: float = 0.7,
) -> Tuple[np.ndarray, Theta, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Permutation-enhanced ABC rejection sampling algorithm.

    Parameters
    ----------
    try_hilbert : bool, default=False
        Use Hilbert curve assignment instead of LSA.
    try_sinkhorn : bool, default=False
        Use Sinkhorn assignment instead of LSA.
    try_swaps : bool, default=False
        Apply pairwise swap refinement after assignment.
    try_lsa : bool, default=True
        Use exact LSA (default).
    alpha_hint : float
        Expected acceptance rate used to size the initial pool.

    Returns
    -------
    datas_perm, thetas_perm, dists_perm, ys_index, zs_index, total_simulations
    """
    if try_hilbert and try_sinkhorn:
        raise ValueError("Cannot use both Hilbert and Sinkhorn simultaneously")

    # Determine primary solver (vanilla = no previous sigma, always full)
    if try_hilbert:
        _method = "hilbert"
    elif try_sinkhorn:
        _method = "sinkhorn"
    else:
        _method = "lsa"
    _swap = try_swaps

    y_obs_np = np.asarray(y_obs)
    if y_obs_np.ndim == 2:
        y_obs_np = y_obs_np[None, ...]

    K = model.K

    def _solve_assignment(datas_np):
        """Compute distances + assignment using the chosen method."""
        if _method == "lsa":
            local_mats = np.asarray(model.distance_matrices_loc(datas_np, y_obs_np))
            global_d = np.asarray(model.distance_global(datas_np, y_obs_np))
            local_mats = np.where(np.isinf(local_mats), 1e12, local_mats)
            ys_idx, zs_idx = solve_lsa(local_mats, parallel=True)
            if _swap:
                ys_idx, zs_idx = do_swap(local_mats, ys_idx, zs_idx)
            dists = np.asarray(compute_total_distance(zs_idx, ys_idx, local_mats, global_d))
            return dists, ys_idx, zs_idx

        if _method == "hilbert":
            zs_slice = datas_np[:, :K]
            y_ref = y_obs_np[0, :K]
            weights = np.asarray(model.weights_distance[:K])
            global_d = np.asarray(model.distance_global(datas_np, y_obs_np))
            h_dist, ys_idx, zs_idx = do_hilbert(zs_slice, y_ref, weights)
            if _swap:
                local_mats = np.asarray(model.distance_matrices_loc(datas_np, y_obs_np))
                local_mats = np.where(np.isinf(local_mats), 1e12, local_mats)
                ys_idx, zs_idx = do_swap(local_mats, ys_idx, zs_idx)
                dists = np.asarray(compute_total_distance(zs_idx, ys_idx, local_mats, global_d))
            else:
                dists = np.sqrt(h_dist ** 2 + global_d)
            return dists, ys_idx, zs_idx

        if _method == "sinkhorn":
            local_mats = np.asarray(model.distance_matrices_loc(datas_np, y_obs_np))
            global_d = np.asarray(model.distance_global(datas_np, y_obs_np))
            local_mats = np.where(np.isinf(local_mats), 1e12, local_mats)
            ys_idx, zs_idx = sinkhorn_assignment(local_mats)
            if _swap:
                ys_idx, zs_idx = do_swap(local_mats, ys_idx, zs_idx)
            dists = np.asarray(compute_total_distance(zs_idx, ys_idx, local_mats, global_d))
            return dists, ys_idx, zs_idx

        raise ValueError(f"Unknown method '{_method}'")

    def _permute_batch(datas_np, locs_np, zs_idx, sel=None):
        """Apply column-permutation to data and loc arrays.

        If *sel* is given, only those rows are permuted and returned.
        """
        if sel is not None:
            perm = zs_idx[sel]
            arange = np.arange(len(sel))[:, None]
            return datas_np[sel][arange, perm], locs_np[sel][arange, perm]
        arange = np.arange(len(datas_np))[:, None]
        return datas_np[arange, zs_idx], locs_np[arange, zs_idx]

    # ------------------------------------------------------------------
    # eps = inf : prior predictive, no filtering
    # ------------------------------------------------------------------
    if epsilon == np.inf:
        key, subkey = random.split(key)
        datas_np, locs_np, globs_np, n_sim = _sample_prior_batch(
            subkey, model, n_points, y_obs_np,
        )
        dists, ys_idx, zs_idx = _solve_assignment(datas_np)
        d_perm, l_perm = _permute_batch(datas_np, locs_np, zs_idx)
        return (
            d_perm,
            Theta(loc=l_perm, glob=globs_np),
            dists,
            ys_idx.astype(np.int32),
            zs_idx.astype(np.int32),
            int(n_sim),
        )

    # ------------------------------------------------------------------
    # eps < inf : pool-and-filter on perm distances
    # ------------------------------------------------------------------
    all_data, all_loc, all_glob = [], [], []
    all_dists, all_ys, all_zs = [], [], []
    n_sim_total = 0
    collected = 0
    alpha_est = max(min(alpha_hint, 0.95), 0.05)

    while collected < n_points:
        n_needed = n_points - collected
        n_pool = int(np.ceil(n_needed / alpha_est)) + 50

        key, subkey = random.split(key)
        datas_np, locs_np, globs_np, n_sim = _sample_prior_batch(
            subkey, model, n_pool, y_obs_np,
        )
        n_sim_total += int(n_sim)

        dists, ys_idx, zs_idx = _solve_assignment(datas_np)

        idx = np.where(dists < epsilon)[0]
        n_acc = len(idx)

        if n_acc > 0:
            take = min(n_acc, n_needed)
            sel = idx[:take]
            d_perm, l_perm = _permute_batch(datas_np, locs_np, zs_idx, sel)
            all_data.append(d_perm)
            all_loc.append(l_perm)
            all_glob.append(globs_np[sel])
            all_dists.append(dists[sel])
            all_ys.append(ys_idx[sel])
            all_zs.append(zs_idx[sel])
            collected += take

        alpha_est = max(n_acc / n_pool, 0.01)

    return (
        np.concatenate(all_data)[:n_points],
        Theta(
            loc=np.concatenate(all_loc)[:n_points],
            glob=np.concatenate(all_glob)[:n_points],
        ),
        np.concatenate(all_dists)[:n_points],
        np.concatenate(all_ys)[:n_points].astype(np.int32),
        np.concatenate(all_zs)[:n_points].astype(np.int32),
        n_sim_total,
    )

