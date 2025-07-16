"""
Core components for permABC algorithms.

This module contains the fundamental building blocks for permutation-based
Approximate Bayesian Computation algorithms:

- **distances**: Optimal distance computation with Linear Sum Assignment
- **kernels**: Proposal kernels for ABC-SMC (Random Walk, Truncated)
- **lsa**: Linear Sum Assignment solver with parallelization
- **moves**: MCMC moves for ABC algorithms (standard and block Gibbs)

These components work together to provide the core functionality for:
- permABC-SMC: Sequential Monte Carlo with permutation optimization
- permABC-UM: Under-matching variants for computational efficiency
- permABC-OS: Over-sampling strategies

Example Usage
-------------
>>> from permabc.core import optimal_index_distance, KernelRW, move_smc
>>> from permabc.models import GaussianModel
>>> 
>>> # Create model and initialize
>>> model = GaussianModel(K=3)
>>> kernel = KernelRW
>>> 
>>> # Compute optimal distances with permutation
>>> distances, y_idx, z_idx, n_lsa = optimal_index_distance(
...     model=model, zs=simulations, y_obs=observations, epsilon=0.1
... )
>>> 
>>> # Perform MCMC move
>>> new_thetas, new_zs, new_distances, _, _, _, accept_rate, _ = move_smc(
...     key=rng_key, model=model, thetas=thetas, zs=zs, 
...     weights=weights, epsilon=0.1, y_obs=observations,
...     distance_values=distances, kernel=kernel, perm=True
... )
"""

# Linear Sum Assignment
from .lsa import solve_lsa

# Distance computation functions
from .distances import (
    optimal_index_distance,
    compute_total_distance,
    remove_under_matching,
    compute_distances_with_current_assignment,
    compute_local_distances_for_assignment,
)

# Kernel classes for parameter proposals
from .kernels import (
    Kernel,           # Base class
    KernelRW,         # Random Walk kernel
    KernelTruncatedRW # Truncated Random Walk for bounded parameters
)

# MCMC moves for ABC algorithms
from .moves import (
    move_smc,                # Standard Metropolis-Hastings move
    move_smc_gibbs_blocks,   # Block-wise Gibbs move
    create_block             # Block creation utility
)

# Define public API
__all__ = [
    # === LINEAR SUM ASSIGNMENT ===
    "solve_lsa",
    
    # === DISTANCE COMPUTATION ===
    "optimal_index_distance",
    "compute_total_distance", 
    "remove_under_matching",
    "compute_distances_with_current_assignment",
    "compute_local_distances_for_assignment",
    "optimal_index_distance_old",
    
    # === PROPOSAL KERNELS ===
    "Kernel",
    "KernelRW", 
    "KernelTruncatedRW",
    
    # === MCMC MOVES ===
    "move_smc",
    "move_smc_gibbs_blocks",
    "create_block"
]

# Module metadata
__version__ = "0.1.0"
__author__ = "permABC Development Team"

# Convenience imports for common workflows
def get_default_kernel():
    """
    Get the default kernel class for most ABC-SMC applications.
    
    Returns
    -------
    KernelRW
        Random Walk kernel class, suitable for unbounded parameters.
        
    Notes
    -----
    For bounded parameters, consider using KernelTruncatedRW instead.
    """
    return KernelRW

def get_bounded_kernel():
    """
    Get the recommended kernel for bounded parameter spaces.
    
    Returns
    -------
    KernelTruncatedRW
        Truncated Random Walk kernel class for bounded parameters.
    """
    return KernelTruncatedRW

# Performance and debugging utilities
class CoreConfig:
    """
    Configuration settings for core permABC components.
    """
    
    # LSA solver settings
    LSA_PARALLEL_THRESHOLD = 40  # Matrix size threshold for parallelization
    LSA_CHUNK_SIZE_SMALL = 20    # Chunk size for problems ≤ 1000 matrices
    LSA_CHUNK_SIZE_LARGE = 50    # Chunk size for problems > 1000 matrices
    
    # Distance computation settings
    DISTANCE_SMART_ACCEPTANCE = True  # Use smart acceptance strategy
    DISTANCE_NUMERICAL_INFINITY = 1e12  # Replace infinite values
    
    # Kernel settings
    KERNEL_VARIANCE_MIN_SAMPLES = 25  # Minimum samples for variance estimation
    KERNEL_VARIANCE_SCALING = 2.0     # Scaling factor: sqrt(2 * empirical_var)
    
    # Move settings
    MOVE_PRIOR_RATIO_CLIP = 703       # Clip log prior ratios for numerical stability
    
    @classmethod
    def set_lsa_parallel_threshold(cls, threshold):
        """Set matrix size threshold for LSA parallelization."""
        cls.LSA_PARALLEL_THRESHOLD = threshold
    
    @classmethod
    def set_kernel_variance_scaling(cls, scaling):
        """Set variance scaling factor for kernel proposals."""
        cls.KERNEL_VARIANCE_SCALING = scaling
    
    @classmethod
    def disable_smart_acceptance(cls):
        """Disable smart acceptance optimization in distance computation."""
        cls.DISTANCE_SMART_ACCEPTANCE = False
    
    @classmethod
    def enable_smart_acceptance(cls):
        """Enable smart acceptance optimization in distance computation."""
        cls.DISTANCE_SMART_ACCEPTANCE = True

# Make configuration available at module level
config = CoreConfig()