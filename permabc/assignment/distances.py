"""
Distance computation utilities for permABC.

Pure utility functions that compute distances given assignment indices,
without any assignment strategy logic.
"""

import numpy as np
from typing import Tuple


def remove_under_matching(
    ys_index: np.ndarray,
    zs_index: np.ndarray,
    M: int,
    L: int,
    K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove out-of-bounds assignments for under-matching (L < K)."""
    valid = (zs_index < M) & (ys_index < K)  # (N, cols) bool mask

    N = ys_index.shape[0]
    new_ys = np.zeros((N, L), dtype=np.int32)
    new_zs = np.zeros((N, L), dtype=np.int32)

    # Number of valid entries per row — typically == L for well-formed inputs
    counts = valid.sum(axis=1)
    max_count = int(counts.max()) if N > 0 else 0

    if max_count == ys_index.shape[1] and max_count == L:
        # Fast path: all entries valid, just truncate
        return ys_index[:, :L].astype(np.int32), zs_index[:, :L].astype(np.int32)

    # General path: compact valid entries to the left per row
    for i in range(N):
        sel = np.where(valid[i])[0]
        n = min(len(sel), L)
        new_ys[i, :n] = ys_index[i, sel[:n]]
        new_zs[i, :n] = zs_index[i, sel[:n]]

    return new_ys, new_zs


def compute_total_distance(
    zs_index: np.ndarray,
    ys_index: np.ndarray,
    local_dist_matrices: np.ndarray,
    global_distances: np.ndarray,
) -> np.ndarray:
    """Total distance for each particle given assignment indices."""
    mats = np.asarray(local_dist_matrices)
    ys = np.asarray(ys_index)
    zs = np.asarray(zs_index)
    glob = np.asarray(global_distances)
    pidx = np.arange(ys.shape[0])[:, None]
    local_sums = np.sum(mats[pidx, ys, zs], axis=1)
    return np.sqrt(local_sums + glob)


def compute_distances_with_current_assignment(
    model, zs, y_obs, ys_index, zs_index, M, L,
) -> np.ndarray:
    """Evaluate distances using existing assignment (no matrix build)."""
    global_distances = model.distance_global(zs, y_obs)
    local_distances = compute_local_distances_for_assignment(
        model, zs, y_obs, ys_index, zs_index, M, L,
    )
    return np.sqrt(local_distances + global_distances)


def compute_local_distances_for_assignment(
    model, zs, y_obs, ys_index, zs_index, M, L,
) -> np.ndarray:
    """Pointwise local distances for a given assignment (no cost matrix)."""
    K = model.K
    num_particles = ys_index.shape[0]
    y_obs_slice = y_obs[0, :K]
    weights_slice = model.weights_distance[:K]
    particle_indices = np.arange(num_particles)[:, None]
    ys_index = ys_index.astype(np.int32)
    zs_index = zs_index.astype(np.int32)
    y_assigned = y_obs_slice[ys_index]
    z_assigned = zs[particle_indices, zs_index]
    weights_assigned = weights_slice[ys_index]
    diff = y_assigned - z_assigned
    squared_diffs = np.sum(diff ** 2, axis=2)
    weighted_distances = weights_assigned ** 2 * squared_diffs
    return np.sum(weighted_distances, axis=1)
