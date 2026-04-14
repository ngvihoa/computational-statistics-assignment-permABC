"""
MCMC moves for ABC-SMC algorithms.

This module implements various MCMC moves including standard Metropolis-Hastings
and block-wise Gibbs sampling specifically designed for permutation-based ABC.
These moves are the core of the ABC-SMC algorithm progression.
"""

import numpy as np
from jax import random
from ..assignment.dispatch import optimal_index_distance
from ..utils.functions import Theta
from collections import namedtuple


# === RESULT STRUCTURES FOR CLEAN INTERFACES ===

# Standard move result structure
MoveResult = namedtuple('MoveResult', [
    'thetas',           # Updated parameter particles
    'zs',               # Updated simulated data
    'distance_values',  # Updated distance values
    'ys_index',         # Updated row assignments (None for non-perm)
    'zs_index',         # Updated column assignments (None for non-perm)
    'n_lsa',            # Number of LSA problems solved
    'accept_rate',      # Overall acceptance rate
    'n_simulations'     # Total number of simulations
])

# Block Gibbs move result structure
BlockGibbsResult = namedtuple('BlockGibbsResult', [
    'thetas',           # Updated parameter particles
    'zs',               # Updated simulated data
    'distance_values',  # Updated distance values
    'ys_index',         # Updated row assignments (None for non-perm)
    'zs_index',         # Updated column assignments (None for non-perm)
    'n_lsa',            # Number of LSA problems solved
    'accept_rate_global',  # Global parameter acceptance rate
    'accept_rate_local',   # Local parameter acceptance rate
    'n_simulations'     # Total number of simulations
])


def create_block(key, matched_index, H, K, M=0, L=0):
    """
    Create random blocks of indices for block-wise Gibbs sampling.

    Uses pure NumPy to avoid JAX JIT recompilation when particle counts vary.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key for block creation.
    matched_index : numpy.ndarray
        Current component assignments, shape (n_particles, K).
    H : int
        Number of blocks to create.
    K : int
        Number of observed components.
    M : int, default=0
        Total number of available components. If 0, uses K.
    L : int, default=0
        Number of matched components. If 0, uses K.

    Returns
    -------
    blocks : list of numpy.ndarray
        List of index arrays, each of shape (n_particles, block_size).
    """
    if L == 0: L = K
    if M == 0: M = K
    n_particles = matched_index.shape[0]
    matched_np = np.asarray(matched_index)

    rng = np.random.default_rng(int(key[0]))
    matched_perms = rng.permuted(np.broadcast_to(matched_np, (n_particles, L)).copy(), axis=1)

    bins = np.linspace(0, L, H + 1, dtype=np.intp)
    matched_blocks = [matched_perms[:, bins[i]:bins[i + 1]] for i in range(H)]

    if L < M:
        # Vectorized: build boolean mask of matched indices, then extract unmatched
        matched_set = np.zeros((n_particles, M), dtype=bool)
        np.put_along_axis(matched_set, matched_np[:, :L].astype(np.intp), True, axis=1)
        all_idx = np.broadcast_to(np.arange(M), (n_particles, M))
        unmatched = all_idx[~matched_set].reshape(n_particles, M - L)
        matched_blocks.append(unmatched.astype(matched_np.dtype))

    return matched_blocks


def move_smc(key, model, thetas, zs, weights, epsilon, y_obs, distance_values, kernel,
             ys_index=None, zs_index=None, verbose=1, perm=False, M=0, L=0, parallel=False,
             cascade=None):
    """
    Standard Random Walk Metropolis-Hastings move for ABC-SMC.
    
    This is the core MCMC move for ABC-SMC algorithms. It proposes new parameters,
    simulates data, computes distances (with optional permutation optimization),
    and accepts/rejects based on ABC criterion and Metropolis-Hastings ratio.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with data generation and distance methods.
    thetas : Theta
        Current parameter particles.
    zs : numpy.ndarray
        Current simulated observations.
    weights : numpy.ndarray
        Particle weights (used in kernel variance computation).
    epsilon : float
        ABC acceptance threshold.
    y_obs : numpy.ndarray
        Observed data for distance computation.
    distance_values : numpy.ndarray
        Current distance values for all particles.
    kernel : class
        Kernel class for parameter proposals (e.g., KernelRW).
    ys_index : numpy.ndarray, optional
        Current row assignments for permutation matching.
    zs_index : numpy.ndarray, optional
        Current column assignments for permutation matching.
    verbose : int, default=1
        Verbosity level for diagnostic output.
    perm : bool, default=False
        Whether to use permutation-based distance computation.
    M : int, default=0
        Total number of simulated components.
    L : int, default=0
        Number of components to match.
    parallel : bool, default=False
        Whether to use parallel LSA computation.
        
    Returns
    -------
    MoveResult
        Named tuple containing:
        - thetas: Updated parameter particles
        - zs: Updated simulated observations
        - distance_values: Updated distance values
        - ys_index: Updated row assignments (None if perm=False)
        - zs_index: Updated column assignments (None if perm=False)
        - n_lsa: Number of LSA problems solved
        - accept_rate: Overall acceptance rate
        - n_simulations: Total number of simulations performed
        
    Notes
    -----
    Algorithm steps:
    1. Propose new parameters using kernel
    2. Simulate new data from proposed parameters  
    3. Compute distances (with permutation optimization if perm=True)
    4. Calculate Metropolis-Hastings acceptance probability
    5. Accept/reject proposals and update particles
    
    The acceptance probability combines:
    - ABC criterion: distance < epsilon
    - Prior ratio: p(theta_new) / p(theta_old)
    - Proposal ratio: q(theta_old | theta_new) / q(theta_new | theta_old)
    
    Examples
    --------
    >>> result = move_smc(key, model, thetas, zs, weights, epsilon, y_obs, 
    ...                   distances, kernel, perm=True)
    >>> print(f"Acceptance rate: {result.accept_rate:.2%}")
    >>> updated_thetas = result.thetas
    >>> n_lsa_solved = result.n_lsa
    """
    n_particles = thetas.loc.shape[0]
    if M == 0: M = model.K
    if L == 0: L = model.K
    K = model.K
    
    # Split random key for different operations
    key, key_kernel, key_data, key_uniform = random.split(key, 4)
    
    # Step 1: Propose new parameters
    if verbose > 1: 
        print("1. Forward kernel proposal...")
        
    forward_kernel = kernel(model=model, thetas=thetas, weights=weights, 
                           ys_index=ys_index, zs_index=zs_index, 
                           verbose=verbose, M=M, L=L)
    proposed_thetas = forward_kernel.sample(key_kernel)
    
    # Step 2: Simulate new data
    proposed_zs = model.data_generator(key_data, proposed_thetas)
    
    # Step 3: Compute distances with optional permutation optimization
    if perm and K > 1:
        # Use permutation-based distance computation
        proposed_distances, proposed_ys_index, proposed_zs_index, n_lsa = optimal_index_distance(
            model=model, zs=proposed_zs, y_obs=y_obs, epsilon=epsilon,
            ys_index=ys_index, zs_index=zs_index, verbose=verbose,
            M=M, L=L, parallel=parallel,
            cascade=cascade,
        )
    else: 
        # Standard distance computation
        proposed_distances = np.array(model.distance(proposed_zs, y_obs))
        proposed_ys_index = None
        proposed_zs_index = None
        n_lsa = 0
    
    # Step 4: Compute Metropolis-Hastings acceptance probability
    
    # Backward kernel for proposal ratio
    backward_kernel = kernel(model=model, thetas=proposed_thetas, weights=weights, 
                            ys_index=proposed_ys_index, zs_index=proposed_zs_index, 
                            verbose=verbose, tau_loc_glob=forward_kernel.get_tau_loc_glob(), 
                            M=M, L=L)
    
    # Prior probability ratios (NumPy to avoid JAX recompilation)
    prior_forward = np.asarray(model.prior_logpdf(proposed_thetas))
    prior_backward = np.asarray(model.prior_logpdf(thetas))
    prior_logratio = np.minimum(prior_forward - prior_backward, 703)
    
    # Proposal probability ratio
    kernel_logratio = np.asarray(backward_kernel.logpdf(thetas)) - np.asarray(forward_kernel.logpdf(proposed_thetas))
    
    # Combined acceptance probability
    accept_prob = (proposed_distances < epsilon).astype(np.float64) * np.exp(prior_logratio + kernel_logratio)
    accept_prob = np.nan_to_num(np.minimum(accept_prob, 1))
    
    # Step 5: Accept/reject proposals
    uniform_samples = np.random.default_rng(int(key_uniform[0])).uniform(size=n_particles)
    accept = uniform_samples <= accept_prob
    
    # Update accepted particles
    if verbose > 2:
        print("Updating particles - type check:")
        print("thetas.loc type:", type(thetas.loc), "proposed_thetas.loc type:", type(proposed_thetas.loc))
        
    thetas[accept] = proposed_thetas[accept]
    zs[accept] = proposed_zs[accept]
    distance_values = np.array(distance_values)
    distance_values[accept] = proposed_distances[accept]
    
    # Update permutation indices if applicable
    if perm and zs_index is not None and ys_index is not None and K > 1:
        zs_index[accept] = proposed_zs_index[accept]
        ys_index[accept] = proposed_ys_index[accept]
        
    n_accept = np.sum(accept)
    accept_rate = n_accept / n_particles
    
    # Diagnostic output
    if verbose > 1:
        ploc = np.asarray(proposed_thetas.loc)
        pglob = np.asarray(proposed_thetas.glob)
        print(f"2. Proposal statistics:")
        print(f"   Local params: min={np.min(ploc):.2f}, "
              f"max={np.max(ploc):.2f}, mean={np.mean(ploc):.2f}")
        print(f"   Global params: min={np.min(pglob):.2f}, "
              f"max={np.max(pglob):.2f}, mean={np.mean(pglob):.2f}")
        
        print(f"3. MH acceptance:")
        print(f"   Accept prob: min={np.min(accept_prob):.3f}, "
              f"max={np.max(accept_prob):.3f}, mean={np.mean(accept_prob):.3f}")
        print(f"   Prior ratio: min={np.min(prior_logratio):.3f}, "
              f"max={np.max(prior_logratio):.3f}, mean={np.mean(prior_logratio):.3f}")
        print(f"   Kernel ratio: min={np.min(kernel_logratio):.3f}, "
              f"max={np.max(kernel_logratio):.3f}, mean={np.mean(kernel_logratio):.3f}")
        
        print(f"4. Results:")
        print(f"   Acceptance: {np.mean(accept):.2%}")
        print(f"   ABC rejection: {np.mean(proposed_distances >= epsilon):.2%}")
        print(f"   MH rejection: {np.mean(np.logical_and(proposed_distances < epsilon, accept_prob < uniform_samples)):.2%}")
        
        if verbose > 2: 
            print(f"   Prior forward: min={np.min(prior_forward):.3f}, "
                  f"max={np.max(prior_forward):.3f}, mean={np.mean(prior_forward):.3f}")
            print(f"   Prior backward: min={np.min(prior_backward):.3f}, "
                  f"max={np.max(prior_backward):.3f}, mean={np.mean(prior_backward):.3f}")
    
    # Return structured result
    return MoveResult(
        thetas=thetas,
        zs=zs,
        distance_values=distance_values,
        ys_index=ys_index,
        zs_index=zs_index,
        n_lsa=n_lsa,
        accept_rate=accept_rate,
        n_simulations=n_particles * M
    )


def move_smc_gibbs_blocks(key, model, thetas, zs, weights, epsilon, y_obs, distance_values, kernel,
                         ys_index=None, zs_index=None, verbose=1, perm=True, M=0, L=0, H=0,
                         both_loc_glob=True, parallel=False,
                         cascade=None):
    """
    Block-wise Gibbs Metropolis-Hastings move for ABC-SMC.
    
    This advanced move implements a Gibbs-style strategy where parameters are
    updated in blocks rather than all at once. This can improve mixing and
    efficiency, especially for high-dimensional parameter spaces.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model.
    thetas : Theta
        Current parameter particles.
    zs : numpy.ndarray
        Current simulated observations.
    weights : numpy.ndarray
        Particle weights.
    epsilon : float
        ABC acceptance threshold.
    y_obs : numpy.ndarray
        Observed data.
    distance_values : numpy.ndarray
        Current distance values.
    kernel : class
        Kernel class for parameter proposals.
    ys_index : numpy.ndarray, optional
        Current row assignments.
    zs_index : numpy.ndarray, optional
        Current column assignments.
    verbose : int, default=1
        Verbosity level.
    perm : bool, default=True
        Whether to use permutation-based distances.
    M : int, default=0
        Total number of simulated components.
    L : int, default=0
        Number of components to match.
    H : int, default=0
        Number of blocks for local updates. If 0, uses 1.
    both_loc_glob : bool, default=True
        If True, updates both local and global parameters.
        If False, randomly chooses between local and global updates.
    parallel : bool, default=False
        Whether to use parallel LSA computation.
        
    Returns
    -------
    BlockGibbsResult
        Named tuple containing:
        - thetas: Updated parameter particles
        - zs: Updated simulated observations
        - distance_values: Updated distance values
        - ys_index: Updated row assignments (None if perm=False)
        - zs_index: Updated column assignments (None if perm=False)
        - n_lsa: Total number of LSA problems solved
        - accept_rate_global: Acceptance rate for global parameter updates
        - accept_rate_local: Acceptance rate for local parameter updates
        - n_simulations: Total number of simulations performed
        
    Notes
    -----
    Algorithm structure:
    1. Global parameter updates: Update all global parameters at once
    2. Local parameter updates: Update parameters in H random blocks
    
    Block strategy benefits:
    - Better mixing for correlated parameters
    - More efficient for high-dimensional spaces
    - Allows different acceptance rates for different parameter types
    
    This is particularly effective for permABC where local parameters
    have complex correlation structures due to permutation matching.
    
    Examples
    --------
    >>> result = move_smc_gibbs_blocks(key, model, thetas, zs, weights, 
    ...                                epsilon, y_obs, distances, kernel, H=3)
    >>> print(f"Global acceptance: {result.accept_rate_global:.2%}")
    >>> print(f"Local acceptance: {result.accept_rate_local:.2%}")
    >>> print(f"LSA problems solved: {result.n_lsa}")
    """
    n_particles = thetas.loc.shape[0]
    if M == 0: M = model.K
    if L == 0: L = model.K
    K = model.K
    if H == 0: H = 1
    
    # Split random keys — separate keys for global and local data generation
    key, key_kernel, key_data_glob, key_data_loc, key_blocks, key_uniform = random.split(key, 6)
    
    # Initialize counters
    n_lsa = 0
    n_accept_global, n_accept_local = 0, 0
    n_sim_global, n_sim_local = 0, 0
    
    # Create base proposal
    forward_kernel = kernel(model=model, thetas=thetas, weights=weights, 
                           ys_index=ys_index, zs_index=zs_index, 
                           verbose=verbose, M=M, L=L)
    proposed_thetas = forward_kernel.sample(key_kernel)

    # Determine update strategy (NumPy to avoid JAX recompilation)
    if both_loc_glob:
        glob_update = loc_update = np.arange(n_particles)
        block_choice_glob = block_choice_loc = np.full((n_particles,), True)
    else:
        rng_block = np.random.default_rng(int(key_blocks[0]))
        block_choice_glob = rng_block.uniform(size=n_particles) < 0.5
        block_choice_loc = ~block_choice_glob
        
        glob_update = np.where(block_choice_glob)[0]
        loc_update = np.where(block_choice_loc)[0]

    # === GLOBAL PARAMETER UPDATE ===
    if len(glob_update) > 0:
        if verbose > 1: 
            print("Global parameter update...")
            
        # Create global-only proposal (keep local parameters fixed)
        forward_kernel_glob = kernel(
            model=model, thetas=thetas[glob_update], weights=weights[glob_update], 
            ys_index=None if ys_index is None else ys_index[glob_update], 
            zs_index=None if zs_index is None else zs_index[glob_update],
            verbose=verbose, M=M, L=L, tau_loc_glob=forward_kernel.get_tau_loc_glob()
        )
        
        # Construct proposal with only global parameters changed
        proposed_thetas_glob = thetas.copy()
        proposed_thetas_full = Theta(
            loc=thetas[glob_update].loc.copy(),  # Keep local params fixed
            glob=proposed_thetas[glob_update].glob.copy()  # Use proposed global params
        )
        proposed_thetas_glob[glob_update] = proposed_thetas_full
        
        # Simulate data with new global parameters (only glob changed)
        proposed_zs_glob = zs.copy()
        proposed_zs_glob[glob_update] = model.data_generator(key_data_glob, proposed_thetas_glob[glob_update])
        
        # Compute distances
        if perm and K > 1:
            proposed_distances_glob, proposed_ys_index_glob, proposed_zs_index_glob, n_lsa_glob = optimal_index_distance(
                model=model,
                zs=proposed_zs_glob[glob_update],
                y_obs=y_obs,
                epsilon=epsilon,
                ys_index=None if ys_index is None else ys_index[glob_update],
                zs_index=None if zs_index is None else zs_index[glob_update],
                verbose=verbose,
                M=M,
                L=L,
                parallel=parallel,
                cascade=cascade,
            )
            n_lsa += n_lsa_glob
        else:
            proposed_distances_glob = np.array(model.distance(proposed_zs_glob[glob_update], y_obs))
            proposed_ys_index_glob = ys_index[glob_update] if ys_index is not None else None
            proposed_zs_index_glob = zs_index[glob_update] if zs_index is not None else None
        
        # Compute acceptance probability for global update
        backward_kernel_glob = kernel(
            model=model, thetas=proposed_thetas_glob[glob_update], weights=weights[glob_update], 
            ys_index=proposed_ys_index_glob, zs_index=proposed_zs_index_glob, 
            verbose=verbose, M=M, L=L, tau_loc_glob=forward_kernel.get_tau_loc_glob()
        )
        
        prior_forward = np.asarray(model.prior_logpdf(proposed_thetas_glob[glob_update]))
        prior_backward = np.asarray(model.prior_logpdf(thetas[glob_update]))
        prior_logratio = np.minimum(prior_forward - prior_backward, 703)
        kernel_logratio = np.asarray(backward_kernel_glob.logpdf(thetas[glob_update])) - np.asarray(forward_kernel_glob.logpdf(proposed_thetas_glob[glob_update]))
        
        accept_prob = (proposed_distances_glob < epsilon).astype(np.float64) * np.exp(prior_logratio + kernel_logratio)
        accept_prob = np.nan_to_num(np.minimum(accept_prob, 1))

        # Accept/reject global proposals
        rng_glob = np.random.default_rng(int(key_uniform[0]))
        uniform_samples = rng_glob.uniform(size=np.sum(block_choice_glob))
        accept = uniform_samples <= accept_prob
        
        # Update accepted particles (indexing already creates copies)
        thetas[glob_update[accept]] = proposed_thetas_glob[glob_update[accept]]
        zs[glob_update[accept]] = proposed_zs_glob[glob_update[accept]]
        distance_values = np.asarray(distance_values)
        distance_values[glob_update[accept]] = proposed_distances_glob[accept]
        
        if perm and zs_index is not None and ys_index is not None and K > 1:
            zs_index[glob_update[accept]] = proposed_zs_index_glob[accept]
            ys_index[glob_update[accept]] = proposed_ys_index_glob[accept]
        
        # Update global counters
        n_accept_global = np.sum(accept) * M
        n_sim_global = np.sum(block_choice_glob) * M

        if verbose > 1: 
            abc_reject = np.mean(proposed_distances_glob >= epsilon)
            mh_reject = np.mean(np.logical_and(proposed_distances_glob < epsilon, accept_prob < uniform_samples))
            print(f"Global move: acceptance rate = {np.sum(accept)/np.sum(block_choice_glob):.2%} "
                  f"(ABC rejection: {abc_reject:.2%}, MH rejection: {mh_reject:.2%})")

    # === LOCAL PARAMETER BLOCK UPDATES ===
    if len(loc_update) > 0:
        if verbose > 1:
            print(f"Local parameter block updates (H={H} blocks)...")
            
        # Create blocks for local parameter updates
        current_matched_index = zs_index[loc_update] if zs_index is not None else np.repeat([np.arange(K)], len(loc_update), axis=0)
        blocks = create_block(key_blocks, matched_index=current_matched_index, K=K, H=H, M=M, L=L)

        # Initialize local proposals (global params fixed, local params from proposal)
        # Pre-compute the full proposed loc/zs ONCE (not per block)
        loc_src = np.asarray(proposed_thetas[loc_update].loc)
        glob_fixed = np.asarray(thetas[loc_update].glob)
        proposed_thetas_loc = thetas.copy()
        proposed_thetas_loc[loc_update] = Theta(loc=loc_src.copy(), glob=glob_fixed.copy())

        # Simulate data for local updates — use proposed_thetas_loc (glob fixed!)
        proposed_zs_loc = zs.copy()
        proposed_zs_loc[loc_update] = model.data_generator(key_data_loc, proposed_thetas_loc[loc_update])

        # Pre-allocate reusable buffers for the block loop (avoid H full copies)
        loc_buf = Theta._ensure_numpy(thetas.loc).copy()   # mutable copy of current loc
        zs_buf = zs.copy()                                  # mutable copy of current zs

        # Cache tau for kernels (same across blocks)
        tau_cached = forward_kernel.get_tau_loc_glob()

        # Process each block sequentially
        for h, block_h in enumerate(blocks):
            if verbose > 1:
                print(f"   Processing block {h+1}/{len(blocks)} (size: {block_h.shape[1]})...")

            key, key_h = random.split(key)
            block_h_np = np.asarray(block_h)

            # Build block proposal by patching only the block columns into buffers
            # Reset buffers to current state for loc_update particles
            loc_buf[loc_update] = np.asarray(thetas.loc)[loc_update]
            loc_buf[loc_update[:, None], block_h_np] = loc_src[np.arange(len(loc_update))[:, None], block_h_np]
            proposed_thetas_loc_h = Theta(loc=loc_buf, glob=np.asarray(thetas.glob))

            zs_buf[loc_update] = zs[loc_update]
            zs_buf[loc_update[:, None], block_h_np] = proposed_zs_loc[loc_update[:, None], block_h_np]

            # Compute distances for block update
            if perm and K > 1:
                proposed_distances_loc_h, ys_index_loc_h, zs_index_loc_h, n_lsa_loc_h = optimal_index_distance(
                    model=model,
                    zs=zs_buf[loc_update],
                    y_obs=y_obs,
                    epsilon=epsilon,
                    ys_index=ys_index[loc_update] if ys_index is not None else None,
                    zs_index=zs_index[loc_update] if zs_index is not None else None,
                    verbose=verbose,
                    M=M,
                    L=L,
                    parallel=parallel,
                    cascade=cascade,
                )
                n_lsa += n_lsa_loc_h
            else:
                proposed_distances_loc_h = np.array(model.distance(zs_buf[loc_update], y_obs))
                ys_index_loc_h = ys_index[loc_update] if ys_index is not None else None
                zs_index_loc_h = zs_index[loc_update] if zs_index is not None else None

            # Compute acceptance probability for block
            forward_kernel_loc_h = kernel(
                model=model, thetas=thetas[loc_update], weights=weights[loc_update],
                ys_index=None if ys_index is None else ys_index[loc_update],
                zs_index=None if zs_index is None else zs_index[loc_update],
                verbose=verbose, M=M, L=L, tau_loc_glob=tau_cached
            )
            backward_kernel_loc_h = kernel(
                model=model, thetas=proposed_thetas_loc_h[loc_update], weights=weights[loc_update],
                ys_index=ys_index_loc_h, zs_index=zs_index_loc_h,
                verbose=verbose, M=M, L=L, tau_loc_glob=tau_cached
            )

            prior_logratio_h = np.minimum(
                np.asarray(model.prior_logpdf(proposed_thetas_loc_h[loc_update])) - np.asarray(model.prior_logpdf(thetas[loc_update])),
                703
            )
            kernel_logratio_h = np.asarray(backward_kernel_loc_h.logpdf(thetas[loc_update])) - np.asarray(forward_kernel_loc_h.logpdf(proposed_thetas_loc_h[loc_update]))

            accept_prob_h = (proposed_distances_loc_h < epsilon).astype(np.float64) * np.exp(prior_logratio_h + kernel_logratio_h)
            accept_prob_h = np.nan_to_num(np.minimum(accept_prob_h, 1))

            # Accept/reject block proposals
            rng_loc_h = np.random.default_rng(int(key_h[0]))
            uniform_h = rng_loc_h.uniform(size=np.sum(block_choice_loc))
            accept_h = uniform_h <= accept_prob_h

            # Update accepted particles for this block
            thetas[loc_update[accept_h]] = proposed_thetas_loc_h[loc_update[accept_h]]
            zs[loc_update[accept_h]] = zs_buf[loc_update[accept_h]]
            distance_values[loc_update[accept_h]] = proposed_distances_loc_h[accept_h]
            if perm and zs_index is not None and ys_index is not None and K > 1:
                zs_index[loc_update[accept_h]] = zs_index_loc_h[accept_h]
                ys_index[loc_update[accept_h]] = ys_index_loc_h[accept_h]

            # Update local counters
            n_accept_local += np.sum(accept_h) * block_h.shape[1]
            n_sim_local += np.sum(block_choice_loc) * block_h.shape[1]

            if verbose > 1:
                abc_reject = np.mean(proposed_distances_loc_h >= epsilon)
                mh_reject = np.mean(np.logical_and(proposed_distances_loc_h < epsilon, accept_prob_h < uniform_h))
                print(f"   Block {h+1}/{len(blocks)}: acceptance = {np.sum(accept_h)/np.sum(block_choice_loc):.2%} "
                      f"(ABC rejection: {abc_reject:.2%}, MH rejection: {mh_reject:.2%})")

    # === FINAL STATISTICS (CORRECTED) ===
    
    # Calculate individual acceptance rates
    accept_rate_global = n_accept_global / n_sim_global if n_sim_global > 0 else 0.0
    accept_rate_local = n_accept_local / n_sim_local if n_sim_local > 0 else 0.0
    
    # Calculate overall acceptance rate correctly
    total_accepted = n_accept_global + n_accept_local
    total_proposed = n_sim_global + n_sim_local
    accept_rate_overall = total_accepted / total_proposed if total_proposed > 0 else 0.0
    
    if verbose > 1:
        print(f"\n=== BLOCK GIBBS SUMMARY ===")
        print(f"Global updates: {n_accept_global}/{n_sim_global} accepted ({accept_rate_global:.1%})")
        print(f"Local updates:  {n_accept_local}/{n_sim_local} accepted ({accept_rate_local:.1%})")
        print(f"Overall: {total_accepted}/{total_proposed} accepted ({accept_rate_overall:.1%})")
        print(f"LSA problems solved: {n_lsa}")
    
    # Return structured result with corrected rates
    return BlockGibbsResult(
        thetas=thetas,
        zs=zs,
        distance_values=distance_values,
        ys_index=ys_index,
        zs_index=zs_index,
        n_lsa=n_lsa,
        accept_rate_global=accept_rate_global,
        accept_rate_local=accept_rate_local,
        n_simulations=total_proposed
    )


# === CONVENIENCE FUNCTIONS FOR RESULT HANDLING ===

def extract_standard_rates(result):
    """
    Extract acceptance rates from move results for compatibility.

    Parameters
    ----------
    result : MoveResult or BlockGibbsResult
        Result from move_smc or move_smc_gibbs_blocks.

    Returns
    -------
    tuple
        (accept_rate_overall, accept_rate_global, accept_rate_local)
    """
    if isinstance(result, MoveResult):
        return result.accept_rate, result.accept_rate, result.accept_rate
    elif isinstance(result, BlockGibbsResult):
        overall = (result.accept_rate_global + result.accept_rate_local) / 2
        return overall, result.accept_rate_global, result.accept_rate_local
    else:
        raise ValueError(f"Unknown result type: {type(result)}")


def calculate_overall_acceptance_rate(result):
    """
    Calculate the overall acceptance rate for BlockGibbsResult.

    Parameters
    ----------
    result : BlockGibbsResult
        Result from move_smc_gibbs_blocks.

    Returns
    -------
    float
        Overall acceptance rate (average of global and local rates).
    """
    if not isinstance(result, BlockGibbsResult):
        raise ValueError("This function only works with BlockGibbsResult")
    return (result.accept_rate_global + result.accept_rate_local) / 2


def format_move_summary(result, verbose=True):
    """
    Format a summary of move results for logging.
    
    Parameters
    ----------
    result : MoveResult or BlockGibbsResult
        Result from move function.
    verbose : bool, default=True
        Whether to include detailed statistics.
        
    Returns
    -------
    str
        Formatted summary string.
    """
    if isinstance(result, MoveResult):
        summary = f"Standard Move: {result.accept_rate:.2%} acceptance"
        if verbose:
            summary += f", {result.n_lsa} LSA problems, {result.n_simulations} simulations"
    elif isinstance(result, BlockGibbsResult):
        summary = f"Block Gibbs Move: Global={result.accept_rate_global:.2%}, Local={result.accept_rate_local:.2%}"
        if verbose:
            summary += f", {result.n_lsa} LSA problems, {result.n_simulations} simulations"
    else:
        summary = f"Unknown move type: {type(result)}"
    
    return summary


