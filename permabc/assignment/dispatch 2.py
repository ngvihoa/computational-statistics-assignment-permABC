"""
Assignment dispatch: entry point and smart strategies for permABC.

This module contains:
- optimal_index_distance: main entry point for all assignment methods
- _full_assignment: first-time assignment (always LSA)
- _smart_progressive_cascade: generic progressive refinement cascade
"""

import numpy as np
from typing import Optional, Tuple, List

from .solvers.lsa import solve_lsa
from .solvers.hilbert import (
    hilbert_distance, solve_hilbert,
    hilbert_distance_cgal, solve_hilbert_cgal, _HAS_CGAL,
)
from .solvers.sinkhorn import sinkhorn_assignment
from .solvers.swap import do_swap, _HAS_NUMBA

from .distances import (
    remove_under_matching,
    compute_total_distance,
    compute_distances_with_current_assignment,
)


# ===================================================================
# Hilbert dispatchers
# ===================================================================

def do_hilbert(zs_slice, y_ref, weights):
    """Hilbert assignment — CGAL if available, else PCA+2D."""
    if _HAS_CGAL:
        return hilbert_distance_cgal(zs_slice, y_ref, weights)
    return hilbert_distance(zs_slice, y_ref, weights, n_bits=12)


def do_hilbert_solve(zs_slice, y_ref):
    """Hilbert index-only solve — CGAL if available, else PCA+2D."""
    if _HAS_CGAL:
        return solve_hilbert_cgal(zs_slice, y_ref)
    return solve_hilbert(zs_slice, y_ref, n_bits=12)


# ===================================================================
# Internal helpers for building cost matrices and running assignments
# ===================================================================

def _build_cost_and_global(model, zs, y_obs, M, L):
    """Build local cost matrices and global distances."""
    local_mats = np.asarray(model.distance_matrices_loc(zs, y_obs, M, L))
    global_d = np.asarray(model.distance_global(zs, y_obs))
    local_mats = np.where(np.isinf(local_mats), 1e12, local_mats)
    return local_mats, global_d


def _apply_lsa(local_mats, global_d, M, L, K, parallel):
    """Run LSA solver and return (distances, ys, zs)."""
    ys, zs = solve_lsa(local_mats, parallel=parallel)
    if L < K:
        ys, zs = remove_under_matching(ys, zs, M, L, K)
    dists = compute_total_distance(zs, ys, local_mats, global_d)
    return dists, ys, zs


def _apply_sinkhorn(local_mats, global_d, M, L, K, reg=None):
    """Run Sinkhorn solver and return (distances, ys, zs)."""
    ys, zs = sinkhorn_assignment(local_mats, reg=reg)
    if L < K:
        ys, zs = remove_under_matching(ys, zs, M, L, K)
    dists = compute_total_distance(zs, ys, local_mats, global_d)
    return dists, ys, zs


def _apply_hilbert(model, zs_full, y_obs, M, K):
    """Run Hilbert solve and return (distances, ys, zs, global_d)."""
    y_ref = y_obs[0, :K]
    weights = np.asarray(model.weights_distance[:K])
    global_d = np.asarray(model.distance_global(zs_full, y_obs))

    if M == K:
        zs_slice = zs_full[:, :K]
        h_dist, ys, zs_idx = do_hilbert(zs_slice, y_ref, weights)
        dists = np.sqrt(h_dist ** 2 + global_d)
        return dists, ys, zs_idx, global_d

    # M != K: sort separately and pair first K entries
    N = zs_full.shape[0]
    zs_slice = zs_full[:, :M]
    ys_order, _ = do_hilbert_solve(y_ref[None, :], y_ref)
    ys_order = ys_order[0]

    zs_idx_all = np.empty((N, M), dtype=np.int32)
    for i in range(N):
        _, zs_i = do_hilbert_solve(zs_slice[i:i+1], zs_slice[i])
        zs_idx_all[i] = zs_i[0]

    ys = np.tile(ys_order[:K], (N, 1)).astype(np.int32)
    zs_idx = zs_idx_all[:, :K].astype(np.int32)

    pidx = np.arange(N)[:, None]
    y_assigned = y_ref[ys]
    z_assigned = zs_slice[pidx, zs_idx]
    w = weights[ys]
    diff = y_assigned - z_assigned
    local_sq = np.sum(w[:, :, None] ** 2 * diff ** 2, axis=(1, 2))
    dists = np.sqrt(local_sq + global_d)
    return dists, ys, zs_idx, global_d


def _apply_swap(local_mats, global_d, ys_idx, zs_idx, M, L, K, max_sweeps=2):
    """Run swap refinement on existing indices and return (distances, ys, zs)."""
    ys, zs = do_swap(local_mats, ys_idx, zs_idx, max_sweeps)
    if L < K:
        ys, zs = remove_under_matching(ys, zs, M, L, K)
    dists = compute_total_distance(zs, ys, local_mats, global_d)
    return dists, ys, zs


# ===================================================================
# Main entry point
# ===================================================================

# Module-level storage for last smart assignment cascade stats.
# Populated by _smart_progressive_cascade after each call.
# Access via: from permabc.assignment.dispatch import last_smart_stats
last_smart_stats = {}


def optimal_index_distance(
    model,
    zs: np.ndarray,
    y_obs: np.ndarray,
    epsilon: float = 0,
    ys_index: Optional[np.ndarray] = None,
    zs_index: Optional[np.ndarray] = None,
    verbose: int = 0,
    M: int = 0,
    L: int = 0,
    parallel: bool = True,
    cascade: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute optimal assignment distances using a progressive cascade.

    Parameters
    ----------
    cascade : list of str, optional
        Progressive cascade steps, e.g. ``["identity", "hilbert", "swap", "lsa"]``.
        First iteration (no previous indices) always uses LSA regardless.
        Valid steps: ``"identity"``, ``"hilbert"``, ``"sinkhorn"``, ``"swap"``, ``"lsa"``.
        If None, defaults to ``["identity", "swap", "lsa"]``.

    Returns
    -------
    (distances, ys_index, zs_index, n_assignment)
        ``n_assignment`` = number of particles that needed full reassignment.
    """
    if cascade is None:
        cascade = ["identity", "swap", "lsa"]

    zs = np.asarray(zs)
    y_obs = np.asarray(y_obs)
    if zs.ndim == 4 and zs.shape[1] == 1:
        zs = zs[:, 0]
    if y_obs.ndim == 4 and y_obs.shape[0] == 1:
        y_obs = y_obs[0]
    if y_obs.ndim == 2:
        y_obs = y_obs[None, ...]

    N = zs.shape[0]
    K = model.K
    if M == 0: M = K
    if L == 0: L = K
    has_prev = ys_index is not None and zs_index is not None

    if not has_prev:
        # First iteration: always LSA
        return _full_assignment_lsa(model, zs, y_obs, N, K, M, L, parallel, verbose)

    return _smart_progressive_cascade(
        model, zs, y_obs, epsilon,
        ys_index.copy(), zs_index.copy(),
        N, K, M, L, cascade, parallel, verbose,
    )


# ------------------------------------------------------------------
# Full assignment: first iteration, always LSA
# ------------------------------------------------------------------

def _full_assignment_lsa(model, zs, y_obs, N, K, M, L, parallel, verbose):
    """First-time assignment: always LSA (used by cascade API)."""
    local_mats, global_d = _build_cost_and_global(model, zs, y_obs, M, L)
    dists, ys, zs_idx = _apply_lsa(local_mats, global_d, M, L, K, parallel)
    return dists, ys, zs_idx, N


# ------------------------------------------------------------------
# Generic progressive cascade
# ------------------------------------------------------------------

def _smart_progressive_cascade(model, zs, y_obs, epsilon, ys_index, zs_index,
                                N, K, M, L, steps, parallel, verbose):
    """Progressive cascade: try steps in order, fallback on remaining.

    Steps are applied in order. Each step processes only the particles
    still rejected after all previous steps.

    Valid steps: ``"identity"``, ``"hilbert"``, ``"sinkhorn"``, ``"swap"``, ``"lsa"``.

    Parameters
    ----------
    steps : list of str
        Cascade steps in order, e.g. ``["identity", "hilbert", "swap", "lsa"]``.
        ``"identity"`` should be first (tries previous sigma_t).
    """
    global last_smart_stats

    stats = {"n_total": N}

    # Always start by computing distances with current assignment
    cur_dists = compute_distances_with_current_assignment(
        model, zs, y_obs, ys_index, zs_index, M, L,
    )

    # Determine initial rejected set
    if "identity" in steps:
        # Identity = keep current sigma if distance < epsilon
        accepted = cur_dists < epsilon
        all_rejected = ~accepted
        n_acc = int(np.sum(accepted))
        stats["n_accepted_identity"] = n_acc
        stats["rate_identity"] = n_acc / N
        if verbose > 1:
            print(f"  cascade identity: {n_acc}/{N} accepted")
    else:
        # No identity step: all particles need reassignment
        all_rejected = np.ones(N, dtype=bool)

    # --- Pre-compute cost matrices once for all cost-matrix steps ---
    # Steps that need local_mats/global_d: sinkhorn, swap, lsa
    _COST_STEPS = {"sinkhorn", "swap", "lsa"}
    need_cost = bool(_COST_STEPS & set(steps))
    # Indices of particles that will need cost matrices (all non-identity rejected)
    _cost_idx = None       # particle indices into zs (N-space)
    _cost_local = None     # (len(_cost_idx), nr, nc) cost matrices
    _cost_global = None    # (len(_cost_idx),) global distances

    def _ensure_cost_cache():
        """Lazily compute cost matrices for all currently-rejected particles."""
        nonlocal _cost_idx, _cost_local, _cost_global
        if _cost_idx is not None:
            return
        rej = np.where(all_rejected)[0]
        if len(rej) == 0:
            _cost_idx = np.array([], dtype=np.intp)
            _cost_local = np.empty((0, 0, 0))
            _cost_global = np.empty(0)
            return
        _cost_idx = rej
        _cost_local, _cost_global = _build_cost_and_global(
            model, zs[rej], y_obs, M, L)

    def _get_cost_for(idx_rej):
        """Return (local_mats, global_d) for a subset of rejected particles."""
        _ensure_cost_cache()
        if len(idx_rej) == len(_cost_idx) and np.array_equal(idx_rej, _cost_idx):
            return _cost_local, _cost_global
        # Map idx_rej back to rows in cached arrays
        inv = np.searchsorted(_cost_idx, idx_rej)
        return _cost_local[inv], _cost_global[inv]

    # Process remaining steps on rejected particles
    for step in steps:
        if step == "identity":
            continue  # already handled above

        idx_rej = np.where(all_rejected)[0]
        n_rej = len(idx_rej)
        if n_rej == 0:
            break

        is_last = (step == steps[-1])

        if step == "hilbert":
            zs_sub = zs[idx_rej]
            h_dists, ys_h, zs_h, _ = _apply_hilbert(model, zs_sub, y_obs, M, K)

            cur_dists[idx_rej] = h_dists
            ys_index[idx_rej] = ys_h
            zs_index[idx_rej] = zs_h

            if is_last:
                all_rejected[idx_rej] = False
                stats["n_hilbert"] = n_rej
            else:
                accepted_h = h_dists < epsilon
                all_rejected[idx_rej[accepted_h]] = False
                n_acc_h = int(np.sum(accepted_h))
                stats["n_accepted_hilbert"] = n_acc_h
                stats["rate_hilbert"] = n_acc_h / max(n_rej, 1)
                if verbose > 1:
                    print(f"  cascade hilbert: {n_acc_h}/{n_rej} accepted")

        elif step == "sinkhorn":
            local_mats_sub, global_d_sub = _get_cost_for(idx_rej)
            dists_sk, ys_sk, zs_sk = _apply_sinkhorn(
                local_mats_sub, global_d_sub, M, L, K)

            cur_dists[idx_rej] = dists_sk
            ys_index[idx_rej] = ys_sk
            zs_index[idx_rej] = zs_sk

            if is_last:
                all_rejected[idx_rej] = False
                stats["n_sinkhorn"] = n_rej
            else:
                accepted_sk = dists_sk < epsilon
                all_rejected[idx_rej[accepted_sk]] = False
                n_acc_sk = int(np.sum(accepted_sk))
                stats["n_accepted_sinkhorn"] = n_acc_sk
                stats["rate_sinkhorn"] = n_acc_sk / max(n_rej, 1)
                if verbose > 1:
                    print(f"  cascade sinkhorn: {n_acc_sk}/{n_rej} accepted")

        elif step == "swap":
            local_mats_sub, global_d_sub = _get_cost_for(idx_rej)
            dists_sw, ys_sw, zs_sw = _apply_swap(
                local_mats_sub, global_d_sub,
                ys_index[idx_rej], zs_index[idx_rej],
                M, L, K,
            )

            cur_dists[idx_rej] = dists_sw
            ys_index[idx_rej] = ys_sw
            zs_index[idx_rej] = zs_sw

            if is_last:
                all_rejected[idx_rej] = False
                stats["n_swap"] = n_rej
            else:
                accepted_sw = dists_sw < epsilon
                all_rejected[idx_rej[accepted_sw]] = False
                n_acc_sw = int(np.sum(accepted_sw))
                stats["n_accepted_swap"] = n_acc_sw
                stats["rate_swap"] = n_acc_sw / max(n_rej, 1)
                if verbose > 1:
                    print(f"  cascade swap: {n_acc_sw}/{n_rej} accepted")

        elif step == "lsa":
            local_mats_sub, global_d_sub = _get_cost_for(idx_rej)
            dists_lsa, ys_lsa, zs_lsa = _apply_lsa(
                local_mats_sub, global_d_sub, M, L, K, parallel)

            cur_dists[idx_rej] = dists_lsa
            ys_index[idx_rej] = ys_lsa
            zs_index[idx_rej] = zs_lsa
            all_rejected[idx_rej] = False
            stats["n_lsa"] = n_rej
            stats["rate_lsa"] = n_rej / N
            if verbose > 1:
                print(f"  cascade lsa: {n_rej} particles")

        else:
            raise ValueError(f"Unknown cascade step '{step}'")

    n_reassigned = N - stats.get("n_accepted_identity", 0)
    last_smart_stats = stats
    if L < K:
        ys_index, zs_index = remove_under_matching(ys_index, zs_index, M, L, K)
    return cur_dists, ys_index, zs_index, n_reassigned
