"""
Linear Sum Assignment (LSA) solver with parallel processing.

This module provides:
  - solve_lsa: scipy-based solver (fallback / reference)
  - solve_lsa_custom: Jonker-Volgenant C solver with optional warm-start
    (uses librectangular_lsap_custom.so via lsa_ctypes)
"""

from scipy.optimize import linear_sum_assignment as _scipy_lsa
import ctypes
from pathlib import Path
import numpy as np
import multiprocessing as mp
import concurrent.futures

# ---------------------------------------------------------------------------
# Try to load the custom C solver; fall back gracefully
# ---------------------------------------------------------------------------
_HAS_CUSTOM_LSA = False
_c_solve = None

# The compiled .so lives in permabc/core/ (next to the C source)
_core_dir = Path(__file__).resolve().parent.parent.parent / "core"

try:
    _lib_path = _core_dir / "librectangular_lsap_custom.so"
    if _lib_path.exists():
        _lib = ctypes.CDLL(str(_lib_path))
        _c_solve = _lib.solve_rectangular_linear_sum_assignment_ws
        _c_solve.argtypes = [
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        _c_solve.restype = ctypes.c_int
        _HAS_CUSTOM_LSA = True
except Exception:
    pass


# ===================================================================
# scipy-based solver (unchanged legacy path)
# ===================================================================

def solve_lsa(dist_matrices, indices=None, parallel=True, n_jobs=-1):
    """Solve N LSA problems using scipy (no warm-start)."""

    def solve_chunk(chunk):
        return [_scipy_lsa(matrix) for matrix in chunk]

    if indices is None:
        indices = np.arange(dist_matrices.shape[0])
    n_matrices = len(indices)

    mat_dim = np.max(dist_matrices.shape[1:])
    if parallel and (mat_dim > 100 or (n_matrices >= 200 and mat_dim >= 20)):
        n_cpu = mp.cpu_count() if n_jobs == -1 else n_jobs
        if n_matrices <= 1000:
            chunk_size = max(20, n_matrices // (n_cpu * 2))
        else:
            chunk_size = max(50, n_matrices // (n_cpu * 3))
        chunks = [dist_matrices[i:i+chunk_size]
                  for i in range(0, len(dist_matrices), chunk_size)]
        if len(chunks) == 1:
            return solve_chunk(chunks[0])
        max_workers = min(n_cpu, len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(solve_chunk, chunks))
        results = []
        for cr in chunk_results:
            results.extend(cr)
    else:
        results = solve_chunk(dist_matrices)

    ys_idx, zs_idx = zip(*results)
    return np.array(ys_idx, dtype=np.int32), np.array(zs_idx, dtype=np.int32)


# ===================================================================
# Custom JV solver with warm-start support
# ===================================================================

def _solve_chunk_custom(matrices, hints, nr, nc):
    """Solve a chunk of LSA problems with minimal per-call overhead."""
    n = len(matrices)
    K = min(nr, nc)

    row_template = np.arange(K, dtype=np.int32)
    ys_chunk = np.tile(row_template, (n, 1))
    zs_chunk = np.empty((n, K), dtype=np.int32)

    row4col = np.empty(nc, dtype=np.int32)
    col4row = np.empty(nr, dtype=np.int32)
    ws = ctypes.c_int(0)
    wu = ctypes.c_int(0)
    ws_ref = ctypes.byref(ws)
    wu_ref = ctypes.byref(wu)
    total_warm = 0

    if hints is None:
        for idx in range(n):
            row4col[:] = -1
            col4row[:] = -1
            _c_solve(nr, nc, matrices[idx], row4col, col4row,
                     0, None, ws_ref, wu_ref)
            zs_chunk[idx] = col4row[:K]
    else:
        for idx in range(n):
            row4col[:] = -1
            col4row[:] = -1
            _c_solve(nr, nc, matrices[idx], row4col, col4row,
                     1, hints[idx].ctypes.data_as(ctypes.c_void_p),
                     ws_ref, wu_ref)
            total_warm += wu.value
            zs_chunk[idx] = col4row[:K]

    return ys_chunk, zs_chunk, total_warm


def solve_lsa_custom(dist_matrices, init_col4row=None, parallel=True, n_jobs=-1):
    """
    Solve N LSA problems using the custom Jonker-Volgenant C solver.

    Parameters
    ----------
    dist_matrices : np.ndarray, shape (N, nr, nc)
    init_col4row : np.ndarray or None, shape (N, nr)
        Per-particle warm-start hint.
    parallel : bool
    n_jobs : int

    Returns
    -------
    ys_idx : np.ndarray, shape (N, min(nr,nc))   int32
    zs_idx : np.ndarray, shape (N, min(nr,nc))   int32
    """
    if not _HAS_CUSTOM_LSA:
        raise ImportError(
            "Custom LSA library not available. "
            "Run:  bash permabc/core/build_lsa.sh"
        )

    dist_matrices = np.ascontiguousarray(dist_matrices, dtype=np.float64)
    N, nr, nc = dist_matrices.shape

    if init_col4row is not None:
        init_col4row = np.ascontiguousarray(init_col4row, dtype=np.int32)

    n_cpu = mp.cpu_count() if n_jobs == -1 else n_jobs

    if parallel and N > n_cpu * 2 and nr > 20:
        chunk_size = max(16, N // (n_cpu * 2))
        mat_chunks = [dist_matrices[i:i+chunk_size]
                      for i in range(0, N, chunk_size)]
        hint_chunks = (
            [init_col4row[i:i+chunk_size] for i in range(0, N, chunk_size)]
            if init_col4row is not None
            else [None] * len(mat_chunks)
        )

        def _worker(args):
            return _solve_chunk_custom(args[0], args[1], nr, nc)

        max_workers = min(n_cpu, len(mat_chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            chunk_results = list(ex.map(_worker, zip(mat_chunks, hint_chunks)))

        ys_idx = np.concatenate([r[0] for r in chunk_results])
        zs_idx = np.concatenate([r[1] for r in chunk_results])
        total_warm = sum(r[2] for r in chunk_results)
    else:
        ys_idx, zs_idx, total_warm = _solve_chunk_custom(
            dist_matrices, init_col4row, nr, nc,
        )

    if init_col4row is not None and total_warm < N * nr * 0.1:
        import sys
        print(f"  [LSA custom warm] WARNING: only {total_warm}/{N*nr} pairs seeded "
              f"({total_warm/(N*nr):.1%}) — warm-start ineffective", file=sys.stderr)

    return ys_idx, zs_idx
