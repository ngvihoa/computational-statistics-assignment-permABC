"""
Backward-compatibility shim — real code is in permabc.assignment.

All public names are re-exported so ``from permabc.core.distances import X``
continues to work.
"""

# Re-export everything from the new locations
from ..assignment.dispatch import (
    optimal_index_distance,
    do_hilbert,
    do_hilbert_solve,
)

from ..assignment.solvers.swap import do_swap, _HAS_NUMBA

from ..assignment.distances import (
    compute_total_distance,
    remove_under_matching,
    compute_distances_with_current_assignment,
    compute_local_distances_for_assignment,
)

from ..assignment.solvers.hilbert import _HAS_CGAL
