"""
Hilbert curve sort as a replacement for the Linear Sum Assignment solver.

Two backends:
  - Pure-Python (PCA + 2D bit-interleaving) — always available.
  - CGAL (via cgal_hilbert pybind11 module) — exact d-dimensional Hilbert sort,
    much faster and more accurate for d >= 2.

Implements the Hilbert distance H_p described in:
    Bernton, Jacob, Gerber & Robert (2019) — "Approximate Bayesian Computation
    with the Wasserstein Distance", JRSS-B 81(2), section 2.3.2.

Cost: O(K log K)  vs  O(K³) for the Hungarian algorithm.
Property: H_p >= W_p  (Hilbert coupling is feasible for the transport problem).
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Try to load the CGAL backend (compiled .so lives in permabc/core/)
# ---------------------------------------------------------------------------
_HAS_CGAL = False
_cgal = None

try:
    from pathlib import Path
    import importlib.util
    _core_dir = Path(__file__).resolve().parent.parent.parent / "core"
    # Try loading the CGAL module from core/
    import sys
    _old_path = sys.path[:]
    sys.path.insert(0, str(_core_dir))
    try:
        import cgal_hilbert as _cgal
        _HAS_CGAL = True
    except ImportError:
        pass
    finally:
        sys.path[:] = _old_path
except Exception:
    pass


# ---------------------------------------------------------------------------
# Low-level: 2-D Hilbert curve index (standard bit-interleaving algorithm)
# ---------------------------------------------------------------------------

def _rot2d(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    """Rotate/flip a quadrant for the 2D Hilbert curve."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _xy_to_hilbert(x: int, y: int, order: int) -> int:
    """Convert integer 2D coordinates to a Hilbert curve distance (index)."""
    d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot2d(2 * s, x, y, rx, ry)
        s >>= 1
    return d


def _points_to_hilbert_indices(points: np.ndarray, n_bits: int = 12) -> np.ndarray:
    """
    Map K points in R^d to Hilbert curve scalar indices.

    Steps:
      1. Rank-normalize each dimension to {0, ..., 2^n_bits - 1}
      2. Reduce to 2D via PCA if d > 2
      3. Apply the 2D Hilbert xy->index bijection
    """
    K, d = points.shape
    max_val = (1 << n_bits) - 1

    def rank_norm(col):
        ranks = np.argsort(np.argsort(col))
        if K <= 1:
            return np.array([max_val // 2], dtype=np.int64)
        return np.round(ranks * max_val / (K - 1)).astype(np.int64)

    if d == 1:
        return rank_norm(points[:, 0]).astype(float)

    if d > 2:
        centered = points - points.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            coords_2d = centered @ Vt[:2].T
        except np.linalg.LinAlgError:
            coords_2d = centered[:, :2]
    else:
        coords_2d = points.copy()

    gx = rank_norm(coords_2d[:, 0])
    gy = rank_norm(coords_2d[:, 1])
    gx = np.clip(gx, 0, max_val)
    gy = np.clip(gy, 0, max_val)

    return np.array([_xy_to_hilbert(int(gx[i]), int(gy[i]), n_bits)
                     for i in range(K)], dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_hilbert(
    zs: np.ndarray,
    y_ref: np.ndarray,
    n_bits: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Hilbert-sort assignment between K simulated and K observed
    components for N particles.

    Parameters
    ----------
    zs : np.ndarray, shape (N, K, d)
    y_ref : np.ndarray, shape (K, d)
    n_bits : int, default 12

    Returns
    -------
    ys_idx : np.ndarray, shape (N, K) int32
    zs_idx : np.ndarray, shape (N, K) int32
    """
    N, K, d = zs.shape

    h_y = _points_to_hilbert_indices(y_ref, n_bits=n_bits)
    ys_order = np.argsort(h_y).astype(np.int32)

    ys_idx = np.tile(ys_order, (N, 1))
    zs_idx = np.empty((N, K), dtype=np.int32)

    for i in range(N):
        h_z = _points_to_hilbert_indices(zs[i], n_bits=n_bits)
        zs_idx[i] = np.argsort(h_z).astype(np.int32)

    return ys_idx, zs_idx


def hilbert_distance(
    zs: np.ndarray,
    y_ref: np.ndarray,
    weights: np.ndarray,
    n_bits: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hilbert distances + assignment indices.

    Returns
    -------
    distances : np.ndarray, shape (N,)
    ys_idx : np.ndarray, shape (N, K)
    zs_idx : np.ndarray, shape (N, K)
    """
    ys_idx, zs_idx = solve_hilbert(zs, y_ref, n_bits=n_bits)

    N, K, d = zs.shape
    particle_idx = np.arange(N)[:, None]
    y_assigned = y_ref[ys_idx]
    z_assigned = zs[particle_idx, zs_idx]
    w = weights[ys_idx]

    diff = y_assigned - z_assigned
    distances = np.sqrt(
        np.sum(w[:, :, None] ** 2 * diff ** 2, axis=(1, 2))
    )
    return distances, ys_idx, zs_idx


# ===================================================================
# CGAL backend — exact Hilbert sort in any dimension
# ===================================================================

def _cgal_sort(points: np.ndarray) -> np.ndarray:
    """Dispatch to the correct CGAL function based on dimensionality."""
    if not _HAS_CGAL:
        raise ImportError(
            "CGAL Hilbert module not available. "
            "Run:  bash permabc/core/build_cgal.sh"
        )
    pts = np.ascontiguousarray(points, dtype=np.float64)
    d = pts.shape[1]
    if d == 2:
        return np.asarray(_cgal.hilbert_sort_2d(pts), dtype=np.int64)
    elif d == 3:
        return np.asarray(_cgal.hilbert_sort_3d(pts), dtype=np.int64)
    else:
        return np.asarray(_cgal.hilbert_sort_nd(pts), dtype=np.int64)


def solve_hilbert_cgal(
    zs: np.ndarray,
    y_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """CGAL-based Hilbert sort assignment (exact d-dimensional)."""
    N, K, d = zs.shape

    ys_order = _cgal_sort(y_ref).astype(np.int32)
    ys_idx = np.tile(ys_order, (N, 1))
    zs_idx = np.empty((N, K), dtype=np.int32)

    for i in range(N):
        zs_idx[i] = _cgal_sort(zs[i]).astype(np.int32)

    return ys_idx, zs_idx


def hilbert_distance_cgal(
    zs: np.ndarray,
    y_ref: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CGAL-based Hilbert distance + assignment."""
    ys_idx, zs_idx = solve_hilbert_cgal(zs, y_ref)

    N, K, d = zs.shape
    particle_idx = np.arange(N)[:, None]
    y_assigned = y_ref[ys_idx]
    z_assigned = zs[particle_idx, zs_idx]
    w = weights[ys_idx]

    diff = y_assigned - z_assigned
    distances = np.sqrt(
        np.sum(w[:, :, None] ** 2 * diff ** 2, axis=(1, 2))
    )
    return distances, ys_idx, zs_idx
