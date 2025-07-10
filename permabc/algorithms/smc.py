"""
Sequential Monte Carlo (SMC) algorithms for ABC with permutation support.

This module implements ABC-SMC and permABC-SMC algorithms that use sequential
importance sampling to efficiently explore the posterior distribution through
a sequence of decreasing tolerance levels.
"""
from typing import Tuple, Any, List, Dict, Optional
from ..utils.functions import ess, resampling  # Fixed relative import
from ..core.distances import optimal_index_distance  # Fixed relative import
from ..core.moves import move_smc, move_smc_gibbs_blocks  # Fixed relative import
import numpy as np
from jax import random
import time
import matplotlib.pyplot as plt
# Add your custom types here if needed
Theta = Any  # Replace with actual Theta class if available




def init_smc(
    key: random.PRNGKey,
    model: Any,
    n_particles: int,
    y_obs: np.ndarray,
    update_weight_distance: bool = True,
    verbose: int = 1
) -> Tuple[Theta, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Initialize standard ABC-SMC algorithm.
    
    Sets up the initial particle population by sampling from the prior
    and computing initial distances without permutation optimization.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with required methods.
    n_particles : int
        Number of particles to initialize.
    y_obs : numpy.ndarray
        Observed data for distance computation.
    update_weight_distance : bool, default=True
        Whether to update model distance weights based on simulated data.
    verbose : int, default=1
        Verbosity level for output.
        
    Returns
    -------
    thetas : Theta
        Initial parameter particles.
    zs : numpy.ndarray
        Initial simulated data.
    distance_values : numpy.ndarray
        Initial distance values.
    weights : numpy.ndarray
        Initial particle weights (uniform).
    ess_val : float
        Effective sample size (equals n_particles initially).
        
    Notes
    -----
    This initialization is used for standard ABC-SMC without permutation.
    All particles start with equal weights and the ESS is maximum.
    """
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    
    # Sample initial parameters from prior
    thetas = model.prior_generator(key_thetas, n_particles, K)
    
    # Simulate initial data
    zs = model.data_generator(key_zs, thetas)
    
    # Update distance weights if requested
    if update_weight_distance: 
        model.update_weights_distance(zs, verbose)
    
    # Compute initial distances
    distance_values = model.distance(zs, y_obs)
    
    # Initialize uniform weights
    weights = np.ones(n_particles) / n_particles
    ess_val = n_particles
    
    return thetas, zs, distance_values, weights, ess_val


def init_perm_smc(
    key: random.PRNGKey,
    model: Any,
    n_particles: int,
    y_obs: np.ndarray,
    verbose: int = 1,
    update_weight_distance: bool = True,
    parallel: bool = False
) -> Tuple[Theta, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Initialize permutation-enhanced ABC-SMC algorithm.
    
    Sets up the initial particle population with optimal permutation matching
    to resolve label switching issues from the start.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with permutation support.
    n_particles : int
        Number of particles to initialize.
    y_obs : numpy.ndarray
        Observed data for distance computation.
    verbose : int, default=1
        Verbosity level for output.
    update_weight_distance : bool, default=True
        Whether to update model distance weights.
    parallel : bool, default=False
        Whether to use parallel LSA computation.
        
    Returns
    -------
    thetas : Theta
        Initial parameter particles.
    zs : numpy.ndarray
        Initial simulated data.
    distance_values : numpy.ndarray
        Initial optimal distances.
    ys_index : numpy.ndarray
        Initial row assignments for permutation.
    zs_index : numpy.ndarray
        Initial column assignments for permutation.
    weights : numpy.ndarray
        Initial particle weights (uniform).
    ess_val : float
        Effective sample size.
    n_lsa : int
        Number of LSA problems solved during initialization.
        
    Notes
    -----
    This initialization performs optimal permutation matching for all particles
    to establish the best possible starting configuration.
    """
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    
    # Sample initial parameters from prior
    thetas = model.prior_generator(key_thetas, n_particles, K)
    
    # Simulate initial data
    zs = model.data_generator(key_zs, thetas)
    
    if verbose > 1:
        print(f"Initial weight distance update: {update_weight_distance}")
    
    # Update distance weights if requested (pre-permutation)
    if update_weight_distance: 
        model.update_weights_distance(zs, verbose)
    
    # Compute optimal permutation matching
    distance_values, ys_index, zs_index, n_lsa = optimal_index_distance(
        zs=zs, 
        y_obs=y_obs, 
        model=model, 
        verbose=verbose, 
        epsilon=np.inf,  # Accept all particles initially
        parallel=parallel
    )
    
    # Update weights after permutation if requested
    if update_weight_distance: 
        zs_permuted = zs[np.arange(n_particles)[:, np.newaxis], zs_index]
        model.update_weights_distance(zs_permuted, verbose)
    
    # Initialize uniform weights
    weights = np.ones(n_particles) / n_particles
    ess_val = ess(weights)
    
    return thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa


def update_epsilon(
    alive_distances: np.ndarray,
    epsilon_target: float,
    alpha: float
) -> float:
    """
    Update tolerance level for ABC-SMC progression.
    
    Chooses the next epsilon as the alpha-quantile of current alive particles,
    ensuring gradual progression toward the target tolerance.
    
    Parameters
    ----------
    alive_distances : numpy.ndarray
        Distances of particles with positive weights.
    epsilon_target : float
        Final target tolerance.
    alpha : float
        Quantile level (e.g., 0.95 for 95th percentile).
        
    Returns
    -------
    new_epsilon : float
        Next tolerance level.
        
    Notes
    -----
    The new epsilon is the maximum of:
    - alpha-quantile of alive distances
    - target epsilon (to prevent overshooting)
    
    This ensures steady progress while preventing premature termination.
    """
    return max(np.quantile(alive_distances, alpha), epsilon_target)


def update_weights(
    weights: np.ndarray,
    distance_values: np.ndarray,
    epsilon: float
) -> np.ndarray:
    """
    Update particle weights based on ABC criterion.
    
    Sets weights to zero for particles with distance >= epsilon,
    renormalizes remaining weights to sum to 1.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Current particle weights.
    distance_values : numpy.ndarray
        Current distance values.
    epsilon : float
        Current tolerance level.
        
    Returns
    -------
    new_weights : numpy.ndarray
        Updated and normalized weights.
        
    Notes
    -----
    Particles with distance >= epsilon are killed (weight = 0).
    Surviving particles maintain their relative weights.
    """
    weights = weights * np.where(distance_values <= epsilon, 1., 0.)
    
    if np.sum(weights) == 0: 
        print("WARNING: All particles rejected - zero weights!")
        return weights
    
    return weights / np.sum(weights)


def abc_smc(
    key: random.PRNGKey,
    model: Any,
    n_particles: int,
    epsilon_target: float,
    y_obs: np.ndarray,
    kernel: Any,
    alpha_epsilon: float = 0.95,
    Final_iteration: int = 0,
    alpha_resample: float = 0.5,
    num_blocks_gibbs: int = 0,
    update_weights_distance: bool = False,
    verbose: int = 1,
    N_sim_max: float = np.inf,
    N_iteration_max: float = np.inf,
    stopping_accept_rate: float = 0.015,
    stopping_accept_rate_global: Optional[float] = None,
    stopping_accept_rate_local: Optional[float] = None,
    both_loc_glob: bool = False,
    stopping_epsilon_difference: float = 0.00
) -> Dict[str, Any]:
    """
    Standard ABC-SMC algorithm without permutation optimization.
    
    Implements the sequential Monte Carlo approach to ABC, progressively
    decreasing the tolerance level and updating particles through MCMC moves.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with required methods.
    n_particles : int
        Number of particles to maintain.
    epsilon_target : float
        Target tolerance level.
    y_obs : numpy.ndarray
        Observed data.
    kernel : class
        Kernel class for MCMC proposals.
    alpha_epsilon : float, default=0.95
        Quantile level for epsilon updates.
    Final_iteration : int, default=0
        Number of additional iterations after reaching target epsilon.
    alpha_resample : float, default=0.5
        ESS threshold for resampling (fraction of n_particles).
    num_blocks_gibbs : int, default=0
        Number of blocks for Gibbs sampling (0 = standard moves).
    update_weights_distance : bool, default=False
        Whether to adaptively update distance weights.
    verbose : int, default=1
        Verbosity level.
    N_sim_max : int, default=np.inf
        Maximum number of simulations.
    N_iteration_max : int, default=np.inf
        Maximum number of iterations.
    stopping_accept_rate : float, default=0.015
        Minimum acceptance rate before stopping.
    stopping_accept_rate_global : float, optional
        Minimum global acceptance rate (for Gibbs moves).
    stopping_accept_rate_local : float, optional
        Minimum local acceptance rate (for Gibbs moves).
    both_loc_glob : bool, default=False
        Whether to update both local and global parameters in Gibbs.
    stopping_epsilon_difference : float, default=0.001
        Minimum epsilon change to continue.
        
    Returns
    -------
    results : dict
        Dictionary containing algorithm results and diagnostics:
        - 'Thetas': Parameter evolution
        - 'Zs': Data evolution  
        - 'Weights': Weight evolution
        - 'Ess': ESS evolution
        - 'Eps_values': Epsilon evolution
        - 'Dist': Distance evolution
        - 'N_sim': Simulation counts
        - 'Time': Timing information
        - 'Acc_rate': Acceptance rates
        - And various other diagnostics
        
    Notes
    -----
    Algorithm outline:
    1. Initialize particles from prior
    2. For each iteration:
       a. Update epsilon (tolerance level)
       b. Update weights (kill particles with distance >= epsilon)
       c. Resample if ESS too low
       d. MCMC moves to refresh particles
       e. Update distance weights if requested
    3. Continue until target epsilon reached and stopping criteria met
    
    This is the standard ABC-SMC without permutation optimization.
    Use perm_abc_smc() for permutation-enhanced version.
    """
    time_0 = time.time()
    
    # Initialize distance weights if requested
    if update_weights_distance:
        model.reset_weights_distance()
    
    K = model.K
    if y_obs.ndim == 1: 
        y_obs = y_obs.reshape(1, -1)
    
    # Set default stopping rates
    if stopping_accept_rate_global is None:
        stopping_accept_rate_global = stopping_accept_rate
    if stopping_accept_rate_local is None:
        stopping_accept_rate_local = stopping_accept_rate
    
    # Initialize tracking variables
    Acc_rate = []
    Acc_rate_global = []
    Acc_rate_local = []
    
    # Initialize algorithm
    thetas, zs, distance_values, weights, ess_val = init_smc(
        key, model, n_particles, y_obs, verbose=verbose, 
        update_weight_distance=update_weights_distance
    )
    alive = np.where(weights > 0.0)[0]
    
    # Results tracking - Initialize with common structure
    results_data = _init_smc_tracking()
    results_data.update({
        'Thetas': [thetas], 'Zs': [zs], 'Weights': [weights], 
        'Ess': [ess_val], 'Dist': [distance_values], 
        'Nsim': [n_particles * model.K], 'Time': [time.time() - time_0]
    })
    
    # Initialize diagnostics
    initial_diagnostics = _compute_smc_diagnostics(thetas, n_particles)
    results_data.update({
        'Unique_p': [initial_diagnostics['unique_part']], 
        'Unique_c': [initial_diagnostics['unique_comp']],
        'Unique_loc': [initial_diagnostics['unique_loc']], 
        'Unique_glob': [initial_diagnostics['unique_glob']]
    })
    
    epsilon = np.inf
    
    if verbose > 0:
        n_unique = len(np.unique(thetas.reshape_2d(), axis=0))
        print(f"Iteration 0: Epsilon = {epsilon}, ESS = {ess_val:0.0f}, "
              f"Acc. rate = 100%, Unique particles = {n_unique:0.0f}\n")
    
    t = 1  # Start iteration counter at 1 for consistency
    
    # Main SMC loop
    while epsilon > epsilon_target or Final_iteration >= 0:
        if verbose > 1:
            print(f"Particles <= epsilon-1: {np.sum(distance_values[alive] <= epsilon - 1)}")
        
        time_it = time.time()
        old_epsilon = epsilon
        
        if verbose > 1:
            unique_dist = len(np.unique(distance_values[alive]))
            print(f"Unique distances: {unique_dist}/{len(distance_values[alive])} particles alive")
            print(f"Distance range: [{np.min(distance_values[alive]):.3f}, {np.max(distance_values[alive]):.3f}]")
        
        # Update epsilon
        epsilon = update_epsilon(distance_values[alive], epsilon_target, alpha_epsilon)
        
        if verbose > 1: 
            print(f"a) Update Epsilon: new epsilon = {epsilon:.4f}")

        # Update weights
        weights = update_weights(weights, distance_values, epsilon)
        alive = np.where(weights > 0.0)[0]
        old_ess = ess_val
        ess_val = np.round(ess(weights))
        
        if verbose > 1: 
            kill_rate = (old_ess - ess_val) / old_ess
            print(f"b) Update weights: Old ESS = {old_ess:0.0f}, New ESS = {ess_val:0.0f} "
                  f"({kill_rate:.2%} particles killed)")
        
        # Resampling if needed
        if verbose > 1: 
            print("c) Resampling: ", end="")
            
        if ess_val < n_particles * alpha_resample or (epsilon == epsilon_target and ess_val < n_particles):
            key, key_resample = random.split(key)
            thetas, zs, distance_values = resampling(
                key_resample, weights, [thetas, zs, distance_values]
            )
            weights, ess_val = np.ones(n_particles) / n_particles, n_particles
            alive = np.where(weights > 0.0)[0]
            
            if verbose > 0:
                unique_dist = len(np.unique(distance_values[alive]))
                print(f"Resampling... {unique_dist} unique particles left")
        elif verbose > 1:
            print("No resampling needed")
        
        # MCMC moves
        key, key_move = random.split(key)
        if verbose > 1: 
            print("d) Move particles: ", end="")

        if num_blocks_gibbs == 0:
            # Standard move
            result = move_smc(
                key=key_move, model=model, thetas=thetas[alive], zs=zs[alive], 
                weights=weights[alive], epsilon=epsilon, y_obs=y_obs, 
                distance_values=distance_values[alive], kernel=kernel, 
                verbose=verbose, perm=False
            )
            
            # Update particles
            thetas[alive] = result.thetas
            zs[alive] = result.zs
            distance_values[alive] = result.distance_values
            
            # Extract rates
            acc_rate = result.accept_rate
            acc_rate_global = acc_rate_local = acc_rate
            n_sim = result.n_simulations
            
        else:
            # Block Gibbs move
            result = move_smc_gibbs_blocks(
                key=key_move, model=model, thetas=thetas[alive], zs=zs[alive], 
                weights=weights[alive], epsilon=epsilon, y_obs=y_obs, 
                distance_values=distance_values[alive], kernel=kernel, 
                H=num_blocks_gibbs, verbose=verbose, both_loc_glob=both_loc_glob, 
                perm=False
            )
            
            # Update particles
            thetas[alive] = result.thetas
            zs[alive] = result.zs
            distance_values[alive] = result.distance_values
            
            # Extract corrected rates
            acc_rate_global = result.accept_rate_global
            acc_rate_local = result.accept_rate_local
            # Use corrected overall acceptance rate calculation
            acc_rate = calculate_overall_acceptance_rate(result)
            n_sim = result.n_simulations

        # Update distance weights if requested
        if update_weights_distance: 
            if verbose > 1: 
                print("e) Update weights distance: ", end="")
            model.update_weights_distance(zs, verbose)
            distance_values[alive] = model.distance(zs[alive], y_obs)
            weights = update_weights(weights, distance_values, epsilon)
            alive = np.where(weights > 0.0)[0]
            n_killed = np.sum(distance_values[alive] > epsilon)
            if verbose > 1:
                print(f"After weight update: {len(alive)} particles alive ({n_killed} killed)")

        # Compute diagnostics
        diagnostics = _compute_smc_diagnostics(thetas, n_particles)
        
        # Display progress
        if verbose > 0:
            if num_blocks_gibbs > 0:
                print(f"Iteration {t}: Epsilon = {epsilon:0.4f}, ESS = {ess_val:0.0f}, "
                      f"Acc. rate = {acc_rate:.2%} (Global: {acc_rate_global:.2%}, Local: {acc_rate_local:.2%})")
            else:
                print(f"Iteration {t}: Epsilon = {epsilon:0.4f}, ESS = {ess_val:0.0f}, "
                      f"Acc. rate = {acc_rate:.2%}")
            print(f"Uniqueness rates: Particles = {diagnostics['unique_part']:.1%}, "
                  f"Parameters = {diagnostics['unique_comp']:.1%}, "
                  f"Local = {diagnostics['unique_loc']:.1%}, "
                  f"Global = {diagnostics['unique_glob']:.1%}")

        # Store results using common function
        iteration_data = {
            'Thetas': thetas, 'Zs': zs, 'Weights': weights, 'Ess': ess_val,
            'Epsilon': epsilon, 'Dist': distance_values, 'Nsim': n_sim,
            'Acc_rate': acc_rate, 'Acc_rate_global': acc_rate_global, 'Acc_rate_local': acc_rate_local,
            'Time': time.time() - time_it
        }
        iteration_data.update(diagnostics)
        _update_smc_tracking(results_data, iteration_data)
        
        t += 1
        
        if verbose > 1 and N_sim_max < np.inf:
            print(f"SIMULATIONS: {np.sum(results_data['Nsim'])}/{N_sim_max}")
        
        if epsilon == epsilon_target:
            Final_iteration -= 1
            
        if verbose > 1:
            print(f"Stopping epsilon diff = {stopping_epsilon_difference}, "
                  f"Difference = {old_epsilon - epsilon:.6f}")
        
        # Check stopping conditions
        stop_overall = acc_rate < stopping_accept_rate
        stop_global = acc_rate_global < stopping_accept_rate_global
        stop_local = acc_rate_local < stopping_accept_rate_local
        stop_epsilon = epsilon == epsilon_target
        stop_nsim = np.sum(results_data['Nsim']) >= N_sim_max
        stop_niter = t >= N_iteration_max
        stop_eps_diff = (old_epsilon - epsilon) < stopping_epsilon_difference
        
        if stop_overall or stop_global or stop_local or stop_epsilon or stop_nsim or stop_niter or stop_eps_diff:
            epsilon_target = epsilon
            
        # Display stopping reasons
        if stop_overall and verbose > 0: 
            print(f"Overall acceptance rate too low ({acc_rate:.2%} < {stopping_accept_rate:.2%}), stopping at epsilon = {epsilon}")
        if stop_global and verbose > 0: 
            print(f"Global acceptance rate too low ({acc_rate_global:.2%} < {stopping_accept_rate_global:.2%}), stopping at epsilon = {epsilon}")
        if stop_local and verbose > 0: 
            print(f"Local acceptance rate too low ({acc_rate_local:.2%} < {stopping_accept_rate_local:.2%}), stopping at epsilon = {epsilon}")
        if stop_nsim and verbose > 0: 
            print("Maximum number of simulations reached")
        if stop_niter and verbose > 0: 
            print("Maximum number of iterations reached")
        if stop_eps_diff and verbose > 0:
            print(f"Epsilon change too small ({old_epsilon - epsilon:.6f} < {stopping_epsilon_difference})")
        
        if verbose > 0: 
            print()
    
    # Compile final results
    return _compile_smc_results(results_data, time.time() - time_0)


def perm_abc_smc(
    key: random.PRNGKey,
    model: Any,
    n_particles: int,
    epsilon_target: float,
    y_obs: np.ndarray,
    kernel: Any,
    alpha_epsilon: float = 0.95,
    Final_iteration: int = 0,
    alpha_resample: float = 0.5,
    num_blocks_gibbs: int = 0,
    update_weights_distance: bool = False,
    verbose: int = 1,
    N_sim_max: float = np.inf,
    N_iteration_max: float = np.inf,
    stopping_accept_rate: float = 0.015,
    stopping_accept_rate_global: Optional[float] = None,
    stopping_accept_rate_local: Optional[float] = None,
    both_loc_glob: bool = False,
    stopping_epsilon_difference: float = 0.00,
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Permutation-enhanced ABC-SMC algorithm.
    
    Implements ABC-SMC with optimal permutation matching at each iteration
    to handle label switching and improve parameter inference in multi-component models.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    model : object
        Statistical model with permutation support.
    n_particles : int
        Number of particles to maintain.
    epsilon_target : float
        Target tolerance level.
    y_obs : numpy.ndarray
        Observed data.
    kernel : class
        Kernel class for MCMC proposals.
    alpha_epsilon : float, default=0.95
        Quantile level for epsilon updates.
    Final_iteration : int, default=0
        Number of additional iterations after reaching target epsilon.
    alpha_resample : float, default=0.5
        ESS threshold for resampling.
    num_blocks_gibbs : int, default=0
        Number of blocks for Gibbs sampling.
    update_weights_distance : bool, default=False
        Whether to adaptively update distance weights.
    verbose : int, default=1
        Verbosity level.
    N_sim_max : int, default=np.inf
        Maximum number of simulations.
    N_iteration_max : int, default=np.inf
        Maximum number of iterations.
    stopping_accept_rate : float, default=0.015
        Minimum acceptance rate before stopping.
    stopping_accept_rate_global : float, optional
        Minimum global acceptance rate.
    stopping_accept_rate_local : float, optional
        Minimum local acceptance rate.
    both_loc_glob : bool, default=False
        Whether to update both local and global parameters in Gibbs.
    stopping_epsilon_difference : float, default=0.001
        Minimum epsilon change to continue.
    parallel : bool, default=False
        Whether to use parallel LSA computation.
        
    Returns
    -------
    results : dict
        Dictionary containing algorithm results and diagnostics including:
        - All standard ABC-SMC outputs
        - 'Ys_index': Row assignment evolution
        - 'Zs_index': Column assignment evolution  
        - 'N_lsa': LSA problem counts
        
    Notes
    -----
    This algorithm extends standard ABC-SMC with:
    
    1. **Optimal Initialization**: Uses permutation matching from the start
    2. **Permutation-Aware Moves**: MCMC moves consider permutation optimization
    3. **Smart Distance Computation**: Uses efficient LSA with smart acceptance
    4. **Final Permutation**: Applies optimal permutation at algorithm end
    
    Key differences from standard ABC-SMC:
    - Maintains permutation indices (ys_index, zs_index) throughout
    - Uses optimal_index_distance() instead of standard distance computation
    - Can apply final permutation to resolve remaining label switching
    
    Benefits:
    - Better parameter recovery in mixture models
    - Reduced label switching issues
    - Improved posterior approximation quality
    - More efficient exploration of permutation-invariant spaces
    
    Examples
    --------
    >>> from permabc.algorithms import perm_abc_smc
    >>> from permabc.core import KernelRW
    >>> 
    >>> results = perm_abc_smc(
    ...     key=rng_key, model=mixture_model, n_particles=1000,
    ...     epsilon_target=0.1, y_obs=observations, kernel=KernelRW,
    ...     verbose=1, parallel=True
    ... )
    >>> 
    >>> final_thetas = results['Thetas'][-1]
    >>> final_permutations = results['Zs_index'][-1]
    """
    time_0 = time.time()
    
    # Initialize distance weights if requested
    if update_weights_distance:
        model.reset_weights_distance()
    
    K = model.K
    if y_obs.ndim == 1: 
        y_obs = y_obs.reshape(1, -1)
    
    # Set default stopping rates
    if stopping_accept_rate_global is None:
        stopping_accept_rate_global = stopping_accept_rate
    if stopping_accept_rate_local is None:
        stopping_accept_rate_local = stopping_accept_rate
    
    # Initialize tracking variables
    Acc_rate = []
    Acc_rate_global = []
    Acc_rate_local = []
    
    # Initialize with permutation optimization
    thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa = init_perm_smc(
        key, model, n_particles, y_obs, verbose=verbose, 
        update_weight_distance=update_weights_distance, parallel=parallel
    )
    alive = np.where(weights > 0.0)[0]
    
    # Initialize with consistent structure
    results_data = _init_smc_tracking()
    results_data.update({
        'Thetas': [thetas], 'Zs': [zs], 'Weights': [weights], 
        'Ys_index': [ys_index], 'Zs_index': [zs_index],
        'Ess': [ess_val], 'Dist': [distance_values], 
        'Nsim': [n_particles * model.K], 'Nlsa': [n_lsa], 'Time': [time.time() - time_0],
    })

    # Initialize diagnostics
    initial_diagnostics = _compute_smc_diagnostics(thetas, n_particles)
    results_data.update({
        'Unique_p': [initial_diagnostics['unique_part']], 
        'Unique_c': [initial_diagnostics['unique_comp']],
        'Unique_loc': [initial_diagnostics['unique_loc']], 
        'Unique_glob': [initial_diagnostics['unique_glob']]
    })
    
    epsilon = np.inf
    
    if verbose > 0:
        n_unique = len(np.unique(thetas.reshape_2d(), axis=0))
        print(f"Iteration 0: Epsilon = {epsilon}, ESS = {ess_val:0.0f}, "
              f"Acc. rate = 100%, Unique particles = {n_unique:0.0f}\n")
    
    t = 1  # Start at 1 for consistency with abc_smc
    apply_permutation = False
    
    # Main permutation-enhanced SMC loop
    while epsilon > epsilon_target or Final_iteration >= 0:
        time_it = time.time()
        old_epsilon = epsilon
        
        # Update epsilon
        epsilon = update_epsilon(distance_values[alive], epsilon_target, alpha_epsilon)
        if verbose > 1: 
            print(f"a) Update Epsilon: new epsilon = {epsilon:.4f}")
        
        # Update weights
        old_ess = ess_val
        weights = update_weights(weights, distance_values, epsilon)
        alive = np.where(weights > 0.0)[0]
        ess_val = ess(weights)
        
        if verbose > 1: 
            kill_rate = (old_ess - ess_val) / old_ess
            print(f"b) Update weights: Old ESS = {old_ess:0.0f}, New ESS = {ess_val:0.0f} "
                  f"({kill_rate:.2%} particles killed)")
        
        # Resampling if needed
        if verbose > 1: 
            print("c) Resampling: ", end="")
            
        if ess_val < n_particles * alpha_resample or (epsilon == epsilon_target and ess_val < n_particles):
            key, key_resample = random.split(key)
            thetas, zs, distance_values, ys_index, zs_index = resampling(
                key_resample, weights, [thetas, zs, distance_values, ys_index, zs_index]
            )
            weights, ess_val = np.ones(n_particles) / n_particles, n_particles
            alive = np.where(weights > 0.0)[0]
            
            if verbose > 0:
                unique_dist = len(np.unique(distance_values[alive]))
                print(f"Resampling... {unique_dist} unique particles left")
        elif verbose > 1:
            print("No resampling needed")
        
        # Permutation-aware MCMC moves
        key, key_move = random.split(key)
        if verbose > 1: 
            print("d) Move particles: ", end="")

        if num_blocks_gibbs == 0:
            # Standard permutation-aware move
            result = move_smc(
                key=key_move, model=model, thetas=thetas[alive], zs=zs[alive], 
                weights=weights[alive], ys_index=ys_index[alive], zs_index=zs_index[alive], 
                epsilon=epsilon, y_obs=y_obs, distance_values=distance_values[alive], 
                kernel=kernel, verbose=verbose, perm=True, parallel=parallel
            )
            
            # Update particles and permutations
            thetas[alive] = result.thetas
            zs[alive] = result.zs
            distance_values[alive] = result.distance_values
            ys_index[alive] = result.ys_index
            zs_index[alive] = result.zs_index
            
            # Extract statistics
            n_lsa = result.n_lsa
            acc_rate = result.accept_rate
            acc_rate_global = acc_rate_local = acc_rate
            n_sim = result.n_simulations
            
        else:
            # Block Gibbs with permutation awareness
            result = move_smc_gibbs_blocks(
                key=key_move, model=model, thetas=thetas[alive], zs=zs[alive], 
                weights=weights[alive], ys_index=ys_index[alive], zs_index=zs_index[alive], 
                epsilon=epsilon, y_obs=y_obs, distance_values=distance_values[alive], 
                kernel=kernel, H=num_blocks_gibbs, verbose=verbose, 
                both_loc_glob=both_loc_glob, perm=True, parallel=parallel
            )
            
            # Update particles and permutations
            thetas[alive] = result.thetas
            zs[alive] = result.zs
            distance_values[alive] = result.distance_values
            ys_index[alive] = result.ys_index
            zs_index[alive] = result.zs_index
            
            # Extract statistics with corrected rates
            n_lsa = result.n_lsa
            acc_rate_global = result.accept_rate_global
            acc_rate_local = result.accept_rate_local
            # Use corrected overall acceptance rate calculation
            acc_rate = calculate_overall_acceptance_rate(result)
            n_sim = result.n_simulations

        # Update distance weights with permuted data if requested
        if update_weights_distance: 
            if verbose > 1: 
                print("e) Update weights distance: ", end="")
            
            # Use permuted data for weight updates
            zs_permuted = zs[alive[:, None], zs_index[alive]]
            model.update_weights_distance(zs_permuted, verbose=verbose)
            
            # Recompute distances with updated weights
            distance_values[alive], ys_index[alive], zs_index[alive], n_lsa_update = optimal_index_distance(
                zs=zs[alive], y_obs=y_obs, model=model, verbose=verbose, 
                epsilon=epsilon, zs_index=zs_index[alive], ys_index=ys_index[alive],
                parallel=parallel
            )
            
            weights = update_weights(weights, distance_values, epsilon)
            alive = np.where(weights > 0.0)[0]
            n_killed = np.sum(distance_values[alive] > epsilon)
            
            if verbose > 1:
                print(f"After weight update: {len(alive)} particles alive ({n_killed} killed)")

        # Compute diagnostics
        diagnostics = _compute_smc_diagnostics(thetas, n_particles)
        
        # Display progress
        if verbose > 0:
            if num_blocks_gibbs > 0:
                print(f"Iteration {t}: Epsilon = {epsilon:0.4f}, ESS = {ess_val:0.0f}, "
                      f"Acc. rate = {acc_rate:.2%} (Global: {acc_rate_global:.2%}, Local: {acc_rate_local:.2%})")
            else:
                print(f"Iteration {t}: Epsilon = {epsilon:0.4f}, ESS = {ess_val:0.0f}, "
                      f"Acc. rate = {acc_rate:.2%}")
            print(f"Uniqueness rates: Particles = {diagnostics['unique_part']:.1%}, "
                  f"Parameters = {diagnostics['unique_comp']:.1%}, "
                  f"Local = {diagnostics['unique_loc']:.1%}, "
                  f"Global = {diagnostics['unique_glob']:.1%}")
        
       
        
        if verbose > 1 and N_sim_max < np.inf:
            print(f"SIMULATIONS: {np.sum(results_data['Nsim'])}/{N_sim_max}")
        
        if epsilon == epsilon_target:
            Final_iteration -= 1
            
        if verbose > 1:
            print(f"Stopping epsilon diff = {stopping_epsilon_difference}, "
                  f"Difference = {old_epsilon - epsilon:.6f}")
        
        # Check stopping conditions
        stop_overall = acc_rate < stopping_accept_rate
        stop_global = acc_rate_global < stopping_accept_rate_global
        stop_local = acc_rate_local < stopping_accept_rate_local
        stop_epsilon = epsilon == epsilon_target
        stop_nsim = np.sum(results_data['Nsim']) >= N_sim_max
        stop_niter = t >= N_iteration_max
        stop_eps_diff = (old_epsilon - epsilon) < stopping_epsilon_difference
        
        if stop_overall or stop_global or stop_local or stop_epsilon or stop_nsim or stop_niter or stop_eps_diff:
            epsilon_target = epsilon
            apply_permutation = True
            
        # Display stopping reasons
        if stop_overall and verbose > 0: 
            print(f"Overall acceptance rate too low ({acc_rate:.2%} < {stopping_accept_rate:.2%}), stopping at epsilon = {epsilon}")
        if stop_global and verbose > 0: 
            print(f"Global acceptance rate too low ({acc_rate_global:.2%} < {stopping_accept_rate_global:.2%}), stopping at epsilon = {epsilon}")
        if stop_local and verbose > 0: 
            print(f"Local acceptance rate too low ({acc_rate_local:.2%} < {stopping_accept_rate_local:.2%}), stopping at epsilon = {epsilon}")
        if stop_nsim and verbose > 0: 
            print("Maximum number of simulations reached")
        if stop_niter and verbose > 0: 
            print("Maximum number of iterations reached")
        if stop_eps_diff and verbose > 0:
            print(f"Epsilon change too small ({old_epsilon - epsilon:.6f} < {stopping_epsilon_difference})")
            
        # Apply final permutation if stopping
        if apply_permutation and Final_iteration <= 0: 
            if verbose > 0: 
                print("Applying final optimal permutation...")
            
            # Apply optimal permutation to resolve any remaining label switching
            thetas = thetas.copy().apply_permutation(zs_index)
            zs = np.concatenate([
                zs[np.arange(n_particles)[:, np.newaxis], zs_index], 
                zs[:, K:]  # Keep any extra components unchanged
            ], axis=1)
            
            # Reset permutation indices to identity
            zs_index = np.repeat([np.arange(model.K)], n_particles, axis=0)
         # Store results using common function
        iteration_data = {
            'Thetas': thetas, 'Zs': zs, 'Weights': weights, 
            'Ys_index': ys_index, 'Zs_index': zs_index,
            'Ess': ess_val, 'Epsilon': epsilon, 'Dist': distance_values, 
            'Nsim': n_sim, 'Nlsa': n_lsa,
            'Acc_rate': acc_rate, 'Acc_rate_global': acc_rate_global, 'Acc_rate_local': acc_rate_local,
            'Time': time.time() - time_it
        }
        iteration_data.update(diagnostics)
        _update_smc_tracking(results_data, iteration_data)
        
        t += 1
        if verbose > 0: 
            print()
    
    # Compile final results
    return _compile_smc_results(results_data, time.time() - time_0, include_permutation=True)


# === COMMON UTILITY FUNCTIONS FOR SMC ALGORITHMS ===

def _init_smc_tracking(
    epsilon_init: float = np.inf
) -> Dict[str, List[Any]]:
    """
    Initialize common tracking structure for SMC algorithms.
    
    Parameters
    ----------
    epsilon_init : float, default=np.inf
        Initial epsilon value.
        
    Returns
    -------
    tracking_dict : dict
        Initialized tracking dictionary.
    """
    return {
        'Epsilon': [epsilon_init],
        'Acc_rate': [1.0],
        'Acc_rate_global': [1.0], 
        'Acc_rate_local': [1.0],
        'unique_part': [1.0],
        'unique_comp': [1.0],
        'unique_loc': [1.0],
        'unique_glob': [1.0],
    }

def _update_smc_tracking(
    results_data: Dict[str, List[Any]],
    iteration_data: Dict[str, Any]
) -> None:
    """
    Update SMC tracking with iteration data.
    
    Parameters
    ----------
    results_data : dict
        Main results dictionary to update.
    iteration_data : dict
        Data from current iteration.
    """
    for key, value in iteration_data.items():
        if key in results_data:
            results_data[key].append(value)
        else:
            # Handle new keys for diagnostics
            results_data[key] = results_data.get(key, []) + [value]


def _compute_smc_diagnostics(
    thetas: Theta,
    n_particles: int
) -> Dict[str, float]:
    ...
    """
    Compute diagnostic metrics for SMC algorithms.
    
    Parameters
    ----------
    thetas : Theta
        Parameter particles.
    n_particles : int
        Number of particles.
        
    Returns
    -------
    diagnostics : dict
        Diagnostic metrics.
    """
    reshaped_thetas = thetas.reshape_2d()
    
    return {
        'unique_part': len(np.unique(reshaped_thetas, axis=0)) / n_particles,
        'unique_comp': len(np.unique(reshaped_thetas)) / np.prod(reshaped_thetas.shape),
        'unique_loc': len(np.unique(thetas.loc)) / np.prod(thetas.loc.shape),
        'unique_glob': len(np.unique(thetas.glob)) / np.prod(thetas.glob.shape)
    }


def _compile_smc_results(
    results_data: Dict[str, List[Any]],
    total_time: float,
    include_permutation: bool = False
) -> Dict[str, Any]:    
    """
    Compile final results dictionary for SMC algorithms.
    
    Parameters
    ----------
    results_data : dict
        Tracked results throughout algorithm.
    total_time : float
        Total execution time.
    include_permutation : bool, default=False
        Whether to include permutation-specific results.
        
    Returns
    -------
    results : dict
        Final compiled results.
    """
    # Standard results
    compiled_results = {
        "Thetas": results_data['Thetas'], 
        "Zs": results_data['Zs'], 
        "Weights": results_data['Weights'], 
        "Ess": results_data['Ess'], 
        "Eps_values": results_data['Epsilon'], 
        "Dist": results_data['Dist'], 
        "N_sim": results_data['Nsim'], 
        "Time": results_data['Time'], 
        "Acc_rate": results_data['Acc_rate'], 
        "Acc_rate_global": results_data['Acc_rate_global'], 
        "Acc_rate_local": results_data['Acc_rate_local'], 
        "unique_part": results_data['unique_part'], 
        "unique_comp": results_data['unique_comp'], 
        "unique_loc": results_data['unique_loc'], 
        "unique_glob": results_data['unique_glob'], 
        "time_final": total_time
    }
    
    # Add permutation-specific results if requested
    if include_permutation:
        compiled_results.update({
            "Ys_index": results_data['Ys_index'], 
            "Zs_index": results_data['Zs_index'], 
            "N_lsa": results_data['Nlsa']
        })
    
    return compiled_results

