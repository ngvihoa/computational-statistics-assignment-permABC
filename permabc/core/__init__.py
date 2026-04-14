"""
Backward-compatibility shim for permabc.core.

The core functionality has been reorganized into:
  - permabc.assignment  — assignment solvers and distance computation
  - permabc.sampling    — kernels and MCMC moves

This module re-exports everything so that existing imports like
``from permabc.core import KernelRW`` continue to work.
"""

# Assignment
from ..assignment import (
    optimal_index_distance,
    compute_total_distance,
    remove_under_matching,
    compute_distances_with_current_assignment,
    compute_local_distances_for_assignment,
    do_swap,
    do_hilbert,
    do_hilbert_solve,
    solve_lsa,
    sinkhorn_assignment,
)

# Sampling
from ..sampling import (
    Kernel,
    KernelRW,
    KernelTruncatedRW,
    move_smc,
    move_smc_gibbs_blocks,
    calculate_overall_acceptance_rate,
    create_block,
)

__all__ = [
    "solve_lsa",
    "sinkhorn_assignment",
    "optimal_index_distance",
    "compute_total_distance",
    "remove_under_matching",
    "compute_distances_with_current_assignment",
    "compute_local_distances_for_assignment",
    "do_swap",
    "do_hilbert",
    "do_hilbert_solve",
    "Kernel",
    "KernelRW",
    "KernelTruncatedRW",
    "move_smc",
    "move_smc_gibbs_blocks",
    "calculate_overall_acceptance_rate",
    "create_block",
]
