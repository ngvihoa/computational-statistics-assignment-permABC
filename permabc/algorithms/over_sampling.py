"""
Over-sampling ABC algorithms for permutation-based inference.

This module implements permABC with over-sampling strategy where we simulate
more components (M > K) than observed and progressively reduce M towards K.
This approach can improve exploration and parameter recovery.
"""
from typing import Tuple, Any, Dict
from ..utils.functions import ess, resampling  # Fixed relative import
from ..core.distances import optimal_index_distance  # Fixed relative import
from ..core.moves import move_smc, move_smc_gibbs_blocks, calculate_overall_acceptance_rate  # Fixed relative import
from ..algorithms.smc import update_weights, _init_smc_tracking, _update_smc_tracking, _compute_smc_diagnostics, _compile_smc_results  # Fixed relative import
import numpy as np
from jax import random
import time
from scipy.special import gammaln   
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.functions import Theta


def init_perm_over_sampling(key: random.PRNGKey, model: Any, n_particles: int, y_obs: np.ndarray, 
                              epsilon: float, M_0: int, alpha_epsilon: float, verbose: int = 1, 
                              update_weight_distance: bool = True) -> Tuple[Theta, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, float]:

    """
    Initialize permutation-enhanced ABC with over-sampling.
    
    Sets up the initial particle population with M_0 > K components and computes
    optimal permutation matching for the over-sampled scenario.
    
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
    M_0 : int
        Initial number of components to simulate (M_0 > K).
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
    Over-sampling strategy:
    - Simulate M_0 > K components initially
    - Use optimal assignment to match K best components to observations
    - Remaining M_0-K components provide exploration diversity
    """
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    
    # Generate particles with M_0 components
    thetas = model.prior_generator(key_thetas, n_particles, M_0)
    zs = model.data_generator(key_zs, thetas)
    
    if verbose > 1:
        print("a) Simulation of the first particles:")
    
    # Update distance weights with K components + extra components
    if update_weight_distance: 
        # Use first K components and extra components beyond M_0
        combined_data = np.concatenate([zs[:, :K], zs[:, M_0:]], axis=1)
        model.update_weights_distance(combined_data, verbose)
    
    if verbose > 1:
        print("b) Computing the first distances:", end=" ")
    
    # Compute optimal assignment with M_0 components
    distance_values, ys_index, zs_index, n_lsa = optimal_index_distance(
        zs=zs, y_obs=y_obs, model=model, verbose=verbose, epsilon=epsilon, M=M_0
    )
    
    if verbose > 1:
        print("min = {:.2} max = {:.2} mean = {:.2}".format(
            np.min(distance_values), np.max(distance_values), np.mean(distance_values)
        ))
    
    # Estimate epsilon if not provided
    if epsilon == np.inf: 
        epsilon = np.quantile(distance_values, alpha_epsilon)
        print("Epsilon =", epsilon)

    # Update distance weights with permuted data
    if update_weight_distance: 
        # Use optimal permutation for K components + extra components
        zs_permuted = np.concatenate([
            zs[np.arange(n_particles)[:, np.newaxis], zs_index][:, :K], 
            zs[:, M_0:]
        ], axis=1)
        model.update_weights_distance(zs_permuted, verbose)
    
    # Set initial weights based on ABC criterion
    weights = np.where(distance_values <= epsilon, 1., 0.)
    weights /= np.sum(weights)
    ess_val = ess(weights)
    
    if verbose > 1: 
        print("d) Setting the first weights: ESS =", round(ess_val))
    
    return thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon

def duplicate_particles(key: random.PRNGKey, model: Any, weights: np.ndarray, thetas: Theta, 
                        zs: np.ndarray, ys_index: np.ndarray, zs_index: np.ndarray, 
                        distance_values: np.ndarray, old_M: int, new_M: int, alpha_M: float, 
                        verbose: int, n_duplicate: int = 0) -> Tuple[Theta, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Duplicate alive particles with random permutations for exploration.
    
    This function implements the duplication strategy to maintain diversity
    when reducing the number of components from old_M to new_M.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model.
    weights : numpy.ndarray
        Current particle weights.
    thetas : Theta
        Parameter particles.
    zs : numpy.ndarray
        Simulated data.
    ys_index : numpy.ndarray
        Row assignment indices.
    zs_index : numpy.ndarray
        Column assignment indices.
    distance_values : numpy.ndarray
        Current distances.
    old_M : int
        Previous number of components.
    new_M : int
        New number of components.
    alpha_M : float
        Reduction factor for M.
    verbose : int
        Verbosity level.
    n_duplicate : int, default=0
        Number of duplicates (if 0, computed automatically).
        
    Returns
    -------
    tuple
        Updated (thetas, zs, distance_values, ys_index, zs_index, weights)
        
    Notes
    -----
    Duplication helps maintain exploration capability when reducing M.
    Each surviving particle is duplicated with random permutations.
    """
    alive = np.where(weights > 0.0)[0]
    K = model.K
    n_particles = len(thetas[alive])
    
    # Compute number of duplicates if not provided
    if n_duplicate == 0:
        # Probability of survival when reducing from old_M to new_M
        proba_survive = np.exp(
            gammaln(old_M - K + 1) + gammaln(new_M + 1) - 
            gammaln(new_M - K + 1) - gammaln(old_M + 1)
        )
        n_duplicate = int(alpha_M / proba_survive)
    
    if n_duplicate == 1:
        if verbose > 1: 
            print("No duplication because n_duplicate = 1")
        return thetas, zs, distance_values, ys_index, zs_index, weights
    
    if verbose > 1:
        print(f"Duplicating the {n_particles} alive particles in {n_duplicate} copies...")

    # Create random permutations for duplicates
    permutation_duplicates = random.permutation(
        key, 
        np.repeat([np.arange(old_M)], n_duplicate * n_particles, axis=0), 
        axis=1, 
        independent=True
    )
    
    # CORRECTION : Duplicate and permute parameters
    new_thetas = thetas[alive].copy()
    new_thetas = new_thetas.duplicate(n_duplicate, permutation_duplicates)
    
    # CORRECTION : Duplicate and permute data avec la même logique
    new_zs = np.repeat(zs[alive], n_duplicate, axis=0)
    
    # Appliquer les mêmes permutations aux données zs
    # Séparer la partie à permuter (old_M colonnes) de la partie fixe (reste)
    zs_to_permute = new_zs[:, :old_M]  # Les old_M premières colonnes
    zs_fixed = new_zs[:, old_M:]       # Le reste des colonnes
    
    # Appliquer les permutations
    zs_permuted = np.array([
        zs_to_permute[i, perm] for i, perm in enumerate(permutation_duplicates)
    ])
    
    # Recombiner les parties permutées et fixes
    new_zs = np.concatenate([zs_permuted, zs_fixed], axis=1)
    
    # Duplicate assignment indices (pas de permutation nécessaire pour les indices)
    new_ys_index = np.repeat(ys_index[alive], n_duplicate, axis=0)
    new_zs_index = np.repeat(zs_index[alive], n_duplicate, axis=0)
    new_distance_values = np.repeat(distance_values[alive], n_duplicate, axis=0)
    new_weights = np.repeat(weights[alive], n_duplicate, axis=0)
    
    # CORRECTION : Remplacer complètement les arrays au lieu de les concaténer
    # avec les anciens qui contiennent des particules mortes
    
    # Récupérer les particules vivantes originales
    alive_thetas = thetas[alive]
    alive_zs = zs[alive]
    alive_distance_values = distance_values[alive]
    alive_ys_index = ys_index[alive] 
    alive_zs_index = zs_index[alive]
    alive_weights = weights[alive]
    
    # Combiner les particules vivantes originales avec leurs duplicatas
    thetas = alive_thetas.append(new_thetas)  # ou une méthode similaire selon votre classe Theta
    zs = np.concatenate([alive_zs, new_zs], axis=0)
    distance_values = np.concatenate([alive_distance_values, new_distance_values], axis=0)
    ys_index = np.concatenate([alive_ys_index, new_ys_index], axis=0)
    zs_index = np.concatenate([alive_zs_index, new_zs_index], axis=0)
    weights = np.concatenate([alive_weights, new_weights], axis=0)
    weights = np.where(weights > 0.0, 1., 0.)
    weights /= np.sum(weights)
    
    if verbose > 1: 
        print(f'Now particles of shape {thetas.loc.shape} and {zs.shape} with '
              f'{len(np.unique(thetas.reshape_2d(), axis=0)):0.0f} unique particles')
    
    return thetas, zs, distance_values, ys_index, zs_index, weights

def truncate_particles(thetas: Theta, zs: np.ndarray, new_M: int, old_M: int) -> Tuple[Theta, np.ndarray]:
    """
    Truncate particles from old_M to new_M components.
    
    Reduces the number of simulated components while preserving the 
    parameter structure and extra components beyond the main M.
    
    Parameters
    ----------
    thetas : Theta
        Parameter particles to truncate.
    zs : numpy.ndarray
        Simulated data to truncate.
    new_M : int
        New number of components.
    old_M : int
        Previous number of components.
        
    Returns
    -------
    tuple
        (new_thetas, new_zs) with reduced components.
    """
    new_thetas = thetas.copy()
    new_thetas = new_thetas.truncating(new_M, old_M)
    # Keep first new_M components and any extra components beyond old_M
    new_zs = np.concatenate([zs[:, :new_M], zs[:, old_M:]], axis=1)
    
    return new_thetas, new_zs
    

def perm_abc_smc_os(
    key: random.PRNGKey,
    model: Any,
    n_particles: int,
    y_obs: np.ndarray,
    kernel: Any,
    M_0: int,
    epsilon: float = np.inf,
    alpha_epsilon: float = 0.95,
    alpha_M: float = 0.95,
    Final_iteration: int = 0,
    alpha_resample: float = 0.5,
    num_blocks_gibbs: int = 0,
    update_weights_distance: bool = False,
    verbose: int = 1,
    duplicate: bool = False,
    n_duplicate: int = 0,
    stopping_accept_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Permutation-enhanced ABC-SMC with over-sampling strategy.
    
    Implements ABC-SMC where we start with M_0 > K components and progressively
    reduce M towards K. This over-sampling approach can improve exploration
    and parameter recovery in multi-component models.
    
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
    M_0 : int
        Initial number of components (M_0 > K).
    epsilon : float, default=np.inf
        ABC tolerance (estimated if np.inf).
    alpha_epsilon : float, default=0.95
        Quantile for epsilon estimation.
    alpha_M : float, default=0.95
        Reduction factor for M updates.
    Final_iteration : int, default=0
        Additional iterations after M=K.
    alpha_resample : float, default=0.5
        ESS threshold for resampling.
    num_blocks_gibbs : int, default=0
        Number of Gibbs blocks (0 = standard moves).
    update_weights_distance : bool, default=False
        Whether to adaptively update distance weights.
    verbose : int, default=1
        Verbosity level.
    duplicate : bool, default=False
        Whether to use particle duplication.
    n_duplicate : int, default=0
        Number of duplicates (computed if 0).
    stopping_accept_rate : float, default=0.0
        Minimum acceptance rate before stopping.
        
    Returns
    -------
    dict
        Results dictionary with algorithm evolution including:
        - All standard ABC-SMC outputs
        - 'M_values': Evolution of M parameter
        - 'Prop_killed': Proportion of particles killed each iteration
        
    Notes
    -----
    Over-sampling algorithm progression:
    1. Start with M_0 > K components
    2. Each iteration: reduce M = max(K, α_M * (M_old - K) + K)
    3. Use optimal permutation matching with current M
    4. Optional particle duplication for diversity
    5. Continue until M = K
    
    Benefits:
    - Better exploration of parameter space
    - Improved mixing in multi-modal posteriors
    - More robust parameter recovery
    - Handles model uncertainty better
    
    Examples
    --------
    >>> # 3-component model with over-sampling
    >>> results = perm_abc_smc_os(
    ...     key=key, model=model, n_particles=1000, y_obs=data,
    ...     kernel=KernelRW, M_0=6, alpha_M=0.8, duplicate=True
    ... )
    >>> 
    >>> # Check M evolution
    >>> M_evolution = results['M_values']
    >>> final_params = results['Thetas'][-1]
    """
    time_0 = time.time()
    K = model.K
    
    if y_obs.ndim == 1: 
        y_obs = y_obs.reshape(1, -1)
    
    if update_weights_distance:
        model.reset_weights_distance()
        
    # Initialize with over-sampling
    thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon = init_perm_over_sampling(
        key, model, n_particles, y_obs, epsilon, M_0, 
        verbose=verbose, alpha_epsilon=alpha_epsilon, 
        update_weight_distance=update_weights_distance
    )
    alive = np.where(weights > 0.0)[0]
    
    # Initialize tracking with over-sampling specific variables
    results_data = _init_smc_tracking(epsilon_init=epsilon)
    results_data.update({
        'Thetas': [thetas], 'Zs': [zs], 'Weights': [weights], 
        'Ys_index': [ys_index], 'Zs_index': [zs_index],
        'Ess': [ess_val], 'Dist': [distance_values], 
        'M_values': [M_0], 'Nsim': [n_particles * M_0], 'Nlsa': [n_lsa],
        'Time': [time.time() - time_0]
    })
    
    # Initialize diagnostics
    initial_diagnostics = _compute_smc_diagnostics(thetas, n_particles)
    results_data.update({
        'Unique_p': [initial_diagnostics['unique_part']], 
        'Unique_c': [initial_diagnostics['unique_comp']],
        'Unique_loc': [initial_diagnostics['unique_loc']],
        'Unique_glob': [initial_diagnostics['unique_glob']],
        'Prop_killed': [ess_val / n_particles]
    })
    
    M = M_0
    
    if verbose > 0:
        n_unique = len(np.unique(thetas.reshape_2d(), axis=0))
        print(f"Iteration 0: M = {M_0}, Epsilon = {epsilon}, ESS = {ess_val:0.0f}, "
              f"Acc. rate = 100%, Unique particles = {n_unique:0.0f}\n")
    
    t = 1
    apply_permutation = False
    
    # Main over-sampling SMC loop
    while M > K or Final_iteration >= 0:
        old_M = M
        time_it = time.time()
        
        # Update M: progressive reduction towards K
        M = max(min(int(K + (alpha_M * (old_M - K))), old_M - 1), K)
        
        if verbose > 1: 
            print(f"a) Update M: new M = {M}, old M = {old_M}")
    
        # Optional particle duplication
        if duplicate:
            if verbose > 1: 
                print("b) Duplicate particles")
            thetas, zs, distance_values, ys_index, zs_index, weights = duplicate_particles(
                key=key, model=model, weights=weights, thetas=thetas, zs=zs, 
                ys_index=ys_index, zs_index=zs_index, distance_values=distance_values, 
                old_M=old_M, new_M=M, alpha_M=alpha_M, verbose=verbose, n_duplicate=n_duplicate
            )

        alive = np.where(weights > 0.0)[0]
  
        # Truncate particles if reducing M
        if old_M > M:  # Only truncate if M is actually reduced
            old_thetas, old_zs = thetas.copy(), zs.copy()
            thetas, zs = truncate_particles(thetas, zs, M, old_M)
        
            if verbose > 1: 
                print(f"c) Truncate particles: before {old_thetas.loc.shape},{old_zs.shape} "
                      f"after {thetas.loc.shape},{zs.shape}")
        
        # Compute optimal distances with current M
        if verbose > 1: 
            print(f"d) Compute optimal distances: {len(np.unique(distance_values[alive]))} "
                  f"unique alive particles", end=" ")
        
        distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(
            zs=zs[alive], y_obs=y_obs, model=model, verbose=verbose, epsilon=epsilon, M=M
        )
    
        # Update weights
        old_ess = ess(weights)
        weights = update_weights(weights, distance_values, epsilon)
        alive = np.where(weights > 0.0)[0]
        ess_val = ess(weights)
        
        if verbose > 1: 
            kill_rate = (old_ess - ess_val) / old_ess
            print(f"e) Weights update/Killing particles: Old ESS = {round(old_ess)}, "
                  f"New ESS = {round(ess_val)} ({kill_rate:.2%} killed)")
        
        prop_killed = (old_ess - ess_val) / old_ess
        
        # Resampling if needed
        if verbose > 1: 
            print("f) Resampling:", end=" ")
            
        if (ess_val < n_particles * alpha_resample or 
            (M == K and ess_val < n_particles) or 
            (len(zs) > n_particles)):
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
        
        # MCMC moves - Store current M to track after moves
        current_M = M
        
        if verbose > 1: 
            print("g) Move particles:")
            
        key, key_move = random.split(key)
        
        if num_blocks_gibbs == 0:
            # Standard move - ensure M is correctly passed and maintained
            result = move_smc(
                key=key_move, model=model, thetas=thetas[alive], zs=zs[alive], 
                weights=weights[alive], ys_index=ys_index[alive], zs_index=zs_index[alive], 
                epsilon=epsilon, y_obs=y_obs, distance_values=distance_values[alive], 
                kernel=kernel, verbose=verbose, perm=True, M=current_M
            )
            
            # Update particles - ensure correct shapes
            thetas[alive] = result.thetas
            # Ensure zs maintains the correct shape after move
            if result.zs.shape[1] == zs[alive].shape[1]:
                zs[alive] = result.zs
            else:
                # Handle shape mismatch by reconstructing zs correctly
                if verbose > 1:
                    print(f"Shape mismatch in zs: expected {zs[alive].shape}, got {result.zs.shape}")
                # Keep the original zs structure and update only the relevant parts
                zs[alive, :result.zs.shape[1]] = result.zs
                
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
                kernel=kernel, H=num_blocks_gibbs, verbose=verbose, perm=True, M=current_M
            )
            
            # Update particles
            thetas[alive] = result.thetas
            # Handle potential shape mismatch in zs
            if result.zs.shape[1] == zs[alive].shape[1]:
                zs[alive] = result.zs
            else:
                if verbose > 1:
                    print(f"Shape mismatch in zs: expected {zs[alive].shape}, got {result.zs.shape}")
                zs[alive, :result.zs.shape[1]] = result.zs
                
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
            
            zs_permuted = zs[alive[:, None], zs_index[alive]]
            model.update_weights_distance(zs_permuted, verbose)
            
            distance_values[alive], ys_index[alive], zs_index[alive], n_lsa_update = optimal_index_distance(
                zs=zs[alive], y_obs=y_obs, model=model, verbose=verbose, 
                epsilon=epsilon, M=current_M, zs_index=zs_index[alive], ys_index=ys_index[alive]
            )
            
            weights = update_weights(weights, distance_values, epsilon)
            alive = np.where(weights > 0.0)[0]
            n_killed = np.sum(distance_values[alive] > epsilon)
            
            if verbose > 1:
                print(f"After weight update: {len(alive)} particles alive ({n_killed} killed)")
        
        # Check stopping conditions
        if M == K:
            Final_iteration -= 1
        
        if acc_rate < stopping_accept_rate or M == K:
            apply_permutation = True
            duplicate = False
            
        if acc_rate < stopping_accept_rate and verbose > 0:
            print(f"Acceptance rate too low ({acc_rate:.2%}), stopping at epsilon = {epsilon}")
        
        # Apply final permutation if stopping
        if apply_permutation and Final_iteration <= 0: 
            thetas = thetas.apply_permutation(zs_index)
            zs = np.concatenate([
                zs[np.arange(n_particles)[:, np.newaxis], zs_index], 
                zs[:, K:]
            ], axis=1)
            zs_index = np.repeat([np.arange(model.K)], n_particles, axis=0)
        
        # Compute diagnostics
        diagnostics = _compute_smc_diagnostics(thetas, n_particles)
        
        # Display progress
        if verbose > 0:
            print(f"Iteration {t}: M = {M}, Epsilon = {epsilon:0.4f}, ESS = {ess_val:0.0f}, "
                  f"Acc. rate = {acc_rate:.2%}")
            print(f"Uniqueness rates: Particles = {diagnostics['unique_part']:.1%}, "
                  f"Parameters = {diagnostics['unique_comp']:.1%}")
        
        # Store results
        iteration_data = {
            'Thetas': thetas, 'Zs': zs, 'Weights': weights, 
            'Ys_index': ys_index, 'Zs_index': zs_index,
            'Ess': ess_val, 'Epsilon': epsilon, 'M_values': M, 'Dist': distance_values, 
            'Nsim': n_sim, 'Nlsa': n_lsa, 'Prop_killed': prop_killed,
            'Acc_rate': acc_rate, 'Time': time.time() - time_it
        }
        iteration_data.update(diagnostics)
        _update_smc_tracking(results_data, iteration_data)
        
        t += 1
        
        if verbose > 0: 
            print()

    # Compile final results
    final_results = _compile_smc_results(results_data, time.time() - time_0, include_permutation=True)
    
    # Add over-sampling specific results
    final_results.update({
        "M_values": results_data['M_values'],
        "Prop_killed": results_data['Prop_killed'],
        "Final_iteration": Final_iteration
    })
    
    return final_results