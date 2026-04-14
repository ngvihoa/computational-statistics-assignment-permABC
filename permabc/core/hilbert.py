"""
Backward-compatibility shim — real code is in permabc.assignment.solvers.hilbert.
"""
from ..assignment.solvers.hilbert import (
    hilbert_distance,
    solve_hilbert,
    hilbert_distance_cgal,
    solve_hilbert_cgal,
    _HAS_CGAL,
    _points_to_hilbert_indices,
)
