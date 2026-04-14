"""
Assignment problem solvers and distance computation for permABC.

This package handles the core computational bottleneck of permABC:
finding the optimal permutation between simulated and observed compartments.

Subpackages
-----------
solvers : Individual assignment solvers (LSA, Hilbert, Sinkhorn, Swap)

Modules
-------
distances : Distance utilities (compute_total_distance, etc.)
dispatch : Entry point (optimal_index_distance) and smart strategies
"""

# Main entry point
from .dispatch import optimal_index_distance

# Distance utilities
from .distances import (
    compute_total_distance,
    remove_under_matching,
    compute_distances_with_current_assignment,
    compute_local_distances_for_assignment,
)

# Solver dispatchers (convenience)
from .dispatch import do_swap, do_hilbert, do_hilbert_solve

# Individual solvers
from .solvers.lsa import solve_lsa, solve_lsa_custom
from .solvers.sinkhorn import sinkhorn_assignment

__all__ = [
    "optimal_index_distance",
    "compute_total_distance",
    "remove_under_matching",
    "compute_distances_with_current_assignment",
    "compute_local_distances_for_assignment",
    "do_swap",
    "do_hilbert",
    "do_hilbert_solve",
    "solve_lsa",
    "solve_lsa_custom",
    "sinkhorn_assignment",
]
