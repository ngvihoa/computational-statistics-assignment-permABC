"""
Under-matching ABC algorithms for permutation-based inference.

This module implements permABC with under-matching strategy where we start
with fewer components (L < K) and progressively increase L towards K.
This approach can improve computational efficiency and convergence speed.
"""

from ..utils.functions import ess, resampling  # Fixed relative import
from ..core.distances import optimal_index_distance  # Fixed relative import
from ..core.moves import move_smc, move_smc_gibbs_blocks, calculate_overall_acceptance_rate  # Fixed relative import
from ..algorithms.smc import update_weights, _init_smc_tracking, _update_smc_tracking, _compute_smc_diagnostics, _compile_smc_results  # Fixed relative import
import numpy as np
from jax import random
import time
from scipy.special import gammaln   
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
from jax import random





def init_perm_under_matching(
    key: random.PRNGKey,
    model: Any,
    n_particles: int,
    y_obs: np.ndarray,
    epsilon: float,
    L_0: int,
    alpha_epsilon: float,
    verbose: int = 1,
    update_weight_distance: bool = True
) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    Optional[float], Optional[int], Optional[float]
]:
    """
    Initialize permutation-enhanced ABC with under-matching.
    
    Sets up the initial particle population with L_0 < K components and computes
    optimal permutation matching for the under-matched scenario.
    
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
    epsilon : float
        Initial tolerance (if np.inf, will be estimated).
    L_0 : int
        Initial number of components to match (L_0 < K).
    alpha_epsilon : float
        Quantile level for epsilon estimation if epsilon=np.inf.
    verbose : int, default=1
        Verbosity level.
    update_weight_distance : bool, default=True
        Whether to update distance weights.
        
    Returns
    -------
    tuple
        (thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon)
        
    Notes
    -----
    Under-matching strategy:
    - Start with L_0 < K components to match
    - Use optimal assignment for best L_0 components
    - Progressively increase L towards K
    - Provides faster initial convergence
    """
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    
    # Generate particles with full K components
    thetas = model.prior_generator(key_thetas, n_particles, K)
    zs = model.data_generator(key_zs, thetas)
    
    if verbose > 1:
        print("a) Simulation of the first particles:")
    
    # Update distance weights if requested
    if update_weight_distance: 
        model.update_weights_distance(zs, verbose)
    
    if verbose > 1:
        print(f"b) Computing the first distances with L = {L_0}:", end=" ")
    
    # Compute optimal assignment with L_0 < K components
    distance_values, ys_index, zs_index, n_lsa = optimal_index_distance(
        zs=zs, y_obs=y_obs, model=model, verbose=verbose, epsilon=epsilon, L=L_0
    )
    
    if verbose > 1:
        print("min = {:.2} max = {:.2} mean = {:.2}".format(
            np.min(distance_values), np.max(distance_values), np.mean(distance_values)
        ))
    
    # Estimate epsilon if not provided
    if epsilon == np.inf: 
        epsilon = np.quantile(distance_values, alpha_epsilon)
        print("Epsilon =", epsilon)

    # Set initial weights based on ABC criterion
    weights = np.where(distance_values <= epsilon, 1., 0.)
    
    if np.sum(weights) == 0.0:  
        print("All weights are null, stopping the algorithm")
        return None, None, None, None, None, None, None, None, None
    
    weights /= np.sum(weights)
    ess_val = ess(weights)
    
    if verbose > 1: 
        print("d) Setting the first weights: ESS =", round(ess_val))
    
    alive = np.where(weights > 0.0)[0]
    
    # Update distance weights with matched components
    if update_weight_distance: 
        if verbose > 1: 
            print("h) Update weights distance:", end=" ")
        
        zs_match = [[] for _ in range(K)]
        for i in alive:
            i = int(i)
            for k in range(L_0):
                zs_match[int(ys_index[i, k])].append(zs[i, int(zs_index[i, k])])
        
        len_K_match = [len(z) for z in zs_match]
        if np.min(len_K_match) > 25.:
            model.update_weights_distance(zs_match, verbose)
        
        distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(
            zs=zs[alive], y_obs=y_obs, model=model, verbose=verbose, epsilon=epsilon, L=L_0, 
            zs_index=zs_index[alive], ys_index=ys_index[alive]
        )
        
        weights = update_weights(weights, distance_values, epsilon)
        alive = np.where(weights > 0.0)[0]
        n_killed = np.sum(distance_values[alive] > epsilon)
        
        if verbose > 1:
            print(f"After update weights update: {len(alive)} particles alive ({n_killed} particles killed)")
    
    return thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon


def perm_abc_smc_um(
    key: random.PRNGKey,
    model: Any,
    n_particles: int,
    y_obs: np.ndarray,
    kernel: Any,
    L_0: int,
    epsilon: float = np.inf,
    alpha_epsilon: float = 0.95,
    alpha_L: float = 0.95,
    Final_iteration: int = 0,
    alpha_resample: float = 0.5,
    num_blocks_gibbs: int = 0,
    update_weights_distance: bool = False,
    verbose: int = 1,
    stopping_acc_rate: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Permutation-enhanced ABC-SMC with under-matching strategy.
    
    Implements ABC-SMC where we start with L_0 < K components and progressively
    increase L towards K. This under-matching approach can improve computational
    efficiency and convergence speed.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with permutation support.
    n_particles : int
        Number of particles to maintain.
    y_obs : numpy.ndarray
        Observed data.
    kernel : class
        Kernel class for MCMC proposals.
    L_0 : int
        Initial number of components to match (L_0 < K).
    epsilon : float, default=np.inf
        ABC tolerance (estimated if np.inf).
    alpha_epsilon : float, default=0.95
        Quantile for epsilon estimation.
    alpha_L : float, default=0.95
        Increase factor for L updates.
    Final_iteration : int, default=0
        Additional iterations after L=K.
    alpha_resample : float, default=0.5
        ESS threshold for resampling.
    num_blocks_gibbs : int, default=0
        Number of Gibbs blocks (0 = standard moves).
    update_weights_distance : bool, default=False
        Whether to adaptively update distance weights.
    verbose : int, default=1
        Verbosity level.
    stopping_acc_rate : float, default=0.0
        Minimum acceptance rate before stopping.
        
    Returns
    -------
    dict
        Results dictionary with algorithm evolution including:
        - All standard ABC-SMC outputs
        - 'L_values': Evolution of L parameter
        - 'Prop_killed': Proportion of particles killed each iteration
        
    Notes
    -----
    Under-matching algorithm progression:
    1. Start with L_0 < K components to match
    2. Each iteration: increase L = min(max(K - α_L * (K - L_old), L_old + 1), K)
    3. Use optimal permutation matching with current L
    4. Continue until L = K
    
    Benefits:
    - Faster initial convergence
    - Lower computational cost in early iterations
    - Better exploration when components are poorly identified
    - Gradual refinement of component assignments
    
    Examples
    --------
    >>> # 5-component model with under-matching
    >>> results = perm_abc_smc_um(
    ...     key=key, model=model, n_particles=1000, y_obs=data,
    ...     kernel=KernelRW, L_0=2, alpha_L=0.8
    ... )
    >>> 
    >>> # Check L evolution
    >>> L_evolution = results['L_values']
    >>> final_params = results['Thetas'][-1]
    """
    time_0 = time.time()
    K = model.K
    
    if y_obs.ndim == 1: 
        y_obs = y_obs.reshape(1, -1)
    
    if update_weights_distance:
        model.reset_weights_distance()
        
    # Initialize with under-matching
    init_result = init_perm_under_matching(
        key, model, n_particles, y_obs, epsilon, L_0, 
        verbose=verbose, alpha_epsilon=alpha_epsilon, 
        update_weight_distance=update_weights_distance
    )
    
    if init_result[0] is None:
        return None
    
    thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon = init_result
    alive = np.where(weights > 0.0)[0]
    
    # Initialize tracking with under-matching specific variables
    results_data = _init_smc_tracking(epsilon_init=epsilon)
    results_data.update({
        'Thetas': [thetas], 'Zs': [zs], 'Weights': [weights], 
        'Ys_index': [ys_index], 'Zs_index': [zs_index],
        'Ess': [ess_val], 'Dist': [distance_values], 
        'L_values': [L_0], 'Nsim': [n_particles * K], 'Nlsa': [n_lsa],
        'Time': [time.time() - time_0]
    })
    
    # Initialize diagnostics
    initial_diagnostics = _compute_smc_diagnostics(thetas, n_particles)
    results_data.update({
        'Unique_p': [initial_diagnostics['unique_part']], 
        'Unique_c': [initial_diagnostics['unique_comp']],
        'Prop_killed': [(n_particles - ess_val) / n_particles],
        'Acc_rate': [1.0]
    })
    
    L = L_0
    
    if verbose > 0:
        n_unique = len(np.unique(thetas.reshape_2d(), axis=0))
        print(f"Iteration 0: L = {L_0}, Epsilon = {epsilon}, ESS = {ess_val:0.0f}, "
              f"Acc. rate = 100%, Unique particles = {n_unique:0.0f}\n")
    
    t = 1
    apply_permutation = False
    failure = False
    
    # Main under-matching SMC loop
    while L < K or Final_iteration >= 0:
        old_L = L
        time_it = time.time()
        
        # Update L: progressive increase towards K
        L = min(max(int(K - (alpha_L * (K - old_L))), old_L + 1), K)
        
        if verbose > 1: 
            print(f"a) Update L: new L = {L}, old L = {old_L}")
        
        alive = np.where(weights > 0.0)[0]
        
        if verbose > 1: 
            print(f"b) Compute optimal distances: {len(np.unique(distance_values[alive]))} "
                  f"unique particles", end=" ")
        
        # Reset assignment indices for new L
        ys_index, zs_index = -np.ones((n_particles, L)), -np.ones((n_particles, L))
        
        # Compute optimal assignment with current L
        distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(
            zs=zs[alive], y_obs=y_obs, model=model, verbose=verbose, epsilon=epsilon, L=L
        )
        
        # Update weights
        old_ess = ess(weights)
        weights = update_weights(weights, distance_values, epsilon)
        
        if np.sum(weights) == 0.0:
            failure = True
            print("All weights are null, stopping the algorithm")
            break
        
        alive = np.where(weights > 0.0)[0]
        ess_val = ess(weights)
        
        if verbose > 1: 
            kill_rate = (old_ess - ess_val) / old_ess
            print(f"c) Update weights: Old ESS = {round(old_ess)}, "
                  f"New ESS = {round(ess_val)} ({kill_rate:.2%} of particles killed)")
        
        prop_killed = (old_ess - ess_val) / old_ess
        
        # Resampling if needed
        if verbose > 1: 
            print("f) Resampling:", end=" ")
            
        if ess_val < n_particles * alpha_resample or (L == K and ess_val < n_particles):
            key, key_resample = random.split(key)
            thetas, zs, distance_values, ys_index, zs_index = resampling(
                key_resample, weights, [thetas, zs, distance_values, ys_index, zs_index], 
                n_particles
            )
            weights, ess_val = np.ones(n_particles) / n_particles, n_particles
            alive = np.where(weights > 0.0)[0]
            
            if verbose > 0:
                unique_dist = len(np.unique(distance_values[alive]))
                print(f"Resampling... {unique_dist} unique particles left")
        else: 
            if verbose > 1: 
                print("No resampling")
        
        # MCMC moves
        if verbose > 1: 
            print("d) Move particles:")
            
        key, key_move = random.split(key)
        
        if num_blocks_gibbs == 0:
            # Standard move
            result = move_smc(
                key=key_move, model=model, thetas=thetas[alive], zs=zs[alive], 
                weights=weights[alive], ys_index=ys_index[alive], zs_index=zs_index[alive], 
                epsilon=epsilon, y_obs=y_obs, distance_values=distance_values[alive], 
                kernel=kernel, verbose=verbose, perm=True, L=L
            )
            
            # Update particles
            thetas[alive] = result.thetas
            zs[alive] = result.zs
            distance_values[alive] = result.distance_values
            ys_index[alive] = result.ys_index
            zs_index[alive] = result.zs_index
            
            # Extract statistics
            n_lsa = result.n_lsa
            acc_rate = result.accept_rate
            n_sim = result.n_simulations
            
        else:
            # Block Gibbs move
            result = move_smc_gibbs_blocks(
                key=key_move, model=model, thetas=thetas[alive], zs=zs[alive], 
                weights=weights[alive], ys_index=ys_index[alive], zs_index=zs_index[alive], 
                epsilon=epsilon, y_obs=y_obs, distance_values=distance_values[alive], 
                kernel=kernel, H=num_blocks_gibbs, verbose=verbose, perm=True, L=L
            )
            
            # Update particles
            thetas[alive] = result.thetas
            zs[alive] = result.zs
            distance_values[alive] = result.distance_values
            ys_index[alive] = result.ys_index
            zs_index[alive] = result.zs_index
            
            # Extract statistics
            n_lsa = result.n_lsa
            acc_rate = calculate_overall_acceptance_rate(result)
            n_sim = result.n_simulations
        
        # Update distance weights if requested
        if update_weights_distance: 
            if verbose > 1: 
                print("h) Update weights distance:", end=" ")
            
            zs_match = [[] for _ in range(K)]
            for i in alive:
                i = int(i)
                for k in range(L):
                    zs_match[int(ys_index[i, k])].append(zs[i, int(zs_index[i, k])])
            
            len_K_match = [len(z) for z in zs_match]
            if np.min(len_K_match) > 25.:
                model.update_weights_distance(zs_match, verbose)
            
            distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(
                zs=zs[alive], y_obs=y_obs, model=model, verbose=verbose, epsilon=epsilon, L=L, 
                zs_index=zs_index[alive], ys_index=ys_index[alive]
            )
            
            weights = update_weights(weights, distance_values, epsilon)
            alive = np.where(weights > 0.0)[0]
            n_killed = np.sum(distance_values[alive] > epsilon)
            
            if verbose > 1:
                print(f"After update weights update: {len(alive)} particles alive ({n_killed} particles killed)")
        
        # Check stopping conditions
        if L == K:
            Final_iteration -= 1
            
        if acc_rate < stopping_acc_rate or L == K:
            apply_permutation = True
            
        if acc_rate <= stopping_acc_rate and verbose > 0:
            print(f"Acceptance rate is too low ({acc_rate:.2%}), stopping for epsilon = {epsilon}")
            
        # Apply final permutation if stopping
        if apply_permutation and Final_iteration <= 0: 
            zs_index = np.array(zs_index, dtype=np.int32)
            thetas = thetas.apply_permutation(zs_index)
            zs = np.concatenate([
                zs[np.arange(n_particles)[:, np.newaxis], np.array(zs_index, dtype=np.int32)], 
                zs[:, K:]
            ], axis=1)
            zs_index = np.repeat([np.arange(model.K)], n_particles, axis=0)
        
        # Compute diagnostics
        diagnostics = _compute_smc_diagnostics(thetas, n_particles)
        
        # Display progress
        if verbose > 0:
            print(f"Iteration {t}: L = {L}, Epsilon = {epsilon:0.4f}, ESS = {ess_val:0.0f}, "
                  f"Acc. rate = {acc_rate:.2%}")
            print(f"Uniqueness rates: Particles = {diagnostics['unique_part']:.1%}, "
                  f"Components = {diagnostics['unique_comp']:.1%}, "
                  f"Global params = {len(np.unique(thetas.glob))/n_particles:.1%}")
        
        # Store results
        iteration_data = {
            'Thetas': thetas, 'Zs': zs, 'Weights': weights, 
            'Ys_index': ys_index, 'Zs_index': zs_index,
            'Ess': ess_val, 'Epsilon': epsilon, 'L_values': L, 'Dist': distance_values, 
            'Nsim': n_sim, 'Nlsa': n_lsa, 'Prop_killed': prop_killed,
            'Acc_rate': acc_rate, 'Time': time.time() - time_it
        }
        iteration_data.update(diagnostics)
        _update_smc_tracking(results_data, iteration_data)
        
        t += 1
        
        if verbose > 0: 
            print()
    
    # Handle failure case
    if failure: 
        return None
    
    # Compile final results
    final_results = _compile_smc_results(results_data, time.time() - time_0, include_permutation=True)
    
    # Add under-matching specific results
    final_results.update({
        "L_values": results_data['L_values'],
        "Prop_killed": results_data['Prop_killed'],
        "Final_iteration": Final_iteration
    })
    
    return final_results