"""
Backward-compatibility shim — the ctypes wrapper is kept here since the
compiled .so files live in this directory.

The main LSA solver code is in permabc.assignment.solvers.lsa which
references these .so files via a path back to permabc/core/.
"""
# The actual ctypes wrapper can still be imported directly from this file
# for advanced usage (warm-start API etc.)

import ctypes
from pathlib import Path
import numpy as np

_here = Path(__file__).resolve().parent
_lib_path = _here / "librectangular_lsap_custom.so"

if not _lib_path.exists():
    raise ImportError(
        f"Shared library not found: {_lib_path}\n"
        "Run:  bash permabc/core/build_lsa.sh"
    )

_lib = ctypes.CDLL(str(_lib_path))

_lib.solve_rectangular_linear_sum_assignment_ws.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_lib.solve_rectangular_linear_sum_assignment_ws.restype = ctypes.c_int


def linear_sum_assignment(cost_matrix, init_col4row=None, return_info=False):
    """Solve a linear sum assignment problem (ctypes wrapper)."""
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64, order="C")
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2-D")

    nr, nc = cost_matrix.shape
    transposed = nr > nc
    if transposed:
        cost_matrix = np.ascontiguousarray(cost_matrix.T)
        nr, nc = nc, nr
        if init_col4row is not None:
            init_col4row = None

    row4col = np.full(nc, -1, dtype=np.int32)
    col4row = np.full(nr, -1, dtype=np.int32)

    use_init = 0
    init_ptr = None
    if init_col4row is not None:
        hint = np.asarray(init_col4row, dtype=np.int32)
        if hint.shape != (nr,):
            raise ValueError(f"init_col4row must have shape ({nr},), got {hint.shape}")
        use_init = 1
        init_ptr = hint.ctypes.data_as(ctypes.c_void_p)

    warm_supported = ctypes.c_int(0)
    warm_used = ctypes.c_int(0)

    ret = _lib.solve_rectangular_linear_sum_assignment_ws(
        nr, nc, cost_matrix,
        row4col, col4row,
        use_init, init_ptr,
        ctypes.byref(warm_supported),
        ctypes.byref(warm_used),
    )

    _ERRORS = {
        -1: "Invalid dimensions (nr<=0 or nc<=0)",
        -2: "Transpose case failed unexpectedly",
        -3: "Non-finite value in cost matrix",
        -4: "Augmenting path failed (internal error)",
    }
    if ret != 0:
        raise RuntimeError(f"LSA solver error {ret}: {_ERRORS.get(ret, 'unknown')}")

    if transposed:
        row_ind = col4row.astype(np.int64)
        col_ind = np.arange(nr, dtype=np.int64)
        order = np.argsort(col_ind)
        row_ind, col_ind = col_ind[order], row_ind[order]
    else:
        row_ind = np.arange(nr, dtype=np.int64)
        col_ind = col4row.astype(np.int64)

    if return_info:
        info = {
            "warm_start_requested": bool(use_init),
            "warm_start_supported": bool(warm_supported.value),
            "warm_start_pairs_used": int(warm_used.value),
        }
        return row_ind, col_ind, info

    return row_ind, col_ind
