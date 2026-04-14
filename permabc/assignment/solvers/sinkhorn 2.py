"""
Sinkhorn-based approximate assignment solver.

Computes an approximate solution to the linear sum assignment problem using
entropic regularization (Sinkhorn-Knopp algorithm).  The continuous transport
plan is rounded to a permutation via row-wise argmax.

Complexity per problem: O(K^2 * n_iter), where n_iter is typically 50-200.

Three backends (auto-dispatched):
  - JAX (preferred): vmap + jit + lax.while_loop — batched over particles
  - Numba: @njit(parallel=True) with prange — parallel over particles
  - NumPy (fallback): Python loop over particles
"""

import numpy as np

# ---------------------------------------------------------------------------
# Optional backend imports
# ---------------------------------------------------------------------------
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import jit, vmap, lax
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from scipy.special import logsumexp


# ===================================================================
# NumPy backend (fallback)
# ===================================================================

def _sinkhorn_log_single(cost, reg, max_iter=200, tol=1e-6):
    """Log-domain Sinkhorn for a single (K, K) cost matrix."""
    K = cost.shape[0]
    log_K = -cost / reg
    f = np.zeros(K)
    g = np.zeros(K)
    log_r = np.full(K, -np.log(K))

    for it in range(1, max_iter + 1):
        f = log_r - logsumexp(log_K + g[None, :], axis=1)
        g_new = log_r - logsumexp(log_K + f[:, None], axis=0)
        if np.max(np.abs(g_new - g)) < tol:
            g = g_new
            break
        g = g_new

    log_P = f[:, None] + log_K + g[None, :]
    P = np.exp(log_P)
    return P, it


def _sinkhorn_standard_single(cost, reg, max_iter=200, tol=1e-6):
    """Standard-domain Sinkhorn for a single (K, K) cost matrix."""
    K = cost.shape[0]
    K_mat = np.exp(-cost / reg)
    r = np.ones(K) / K
    u = np.ones(K)

    for it in range(1, max_iter + 1):
        v = r / (K_mat.T @ u)
        u_new = r / (K_mat @ v)
        if not np.all(np.isfinite(u_new)):
            return _sinkhorn_log_single(cost, reg, max_iter, tol)
        if np.max(np.abs(u_new - u)) / (np.max(np.abs(u_new)) + 1e-30) < tol:
            u = u_new
            break
        u = u_new

    P = (u[:, None] * K_mat) * v[None, :]
    return P, it


def _plan_to_assignment(P):
    """Round a (K, K) transport plan to a permutation via row-wise argmax."""
    K = P.shape[0]
    zs_idx = np.argmax(P, axis=1).astype(np.int32)

    # Handle collisions: if two rows map to the same column, resolve greedily
    used = np.zeros(K, dtype=bool)
    for i in range(K):
        if used[zs_idx[i]]:
            order = np.argsort(-P[i])
            for j in order:
                if not used[j]:
                    zs_idx[i] = j
                    break
        used[zs_idx[i]] = True

    ys_idx = np.arange(K, dtype=np.int32)
    return ys_idx, zs_idx


def _sinkhorn_numpy(cost_matrices, reg, max_iter, tol):
    """NumPy fallback: Python loop over particles."""
    N, K, _ = cost_matrices.shape
    ys_idx = np.zeros((N, K), dtype=np.int32)
    zs_idx = np.zeros((N, K), dtype=np.int32)

    for i in range(N):
        P, _ = _sinkhorn_standard_single(cost_matrices[i], reg, max_iter, tol)
        ys_i, zs_i = _plan_to_assignment(P)
        ys_idx[i] = ys_i
        zs_idx[i] = zs_i

    return ys_idx, zs_idx


# ===================================================================
# JAX backend (preferred)
# ===================================================================

if _HAS_JAX:

    @jit
    def _sinkhorn_jax_single(cost, reg, max_iter, tol):
        """
        JAX Sinkhorn for a single (K, K) cost matrix.
        Uses lax.while_loop for convergence — fully JIT-compatible.
        """
        K = cost.shape[0]
        log_K = -cost / reg
        log_r = jnp.full(K, -jnp.log(K))

        def body(state):
            f, g, it, _ = state
            f_new = log_r - jax.scipy.special.logsumexp(log_K + g[None, :], axis=1)
            g_new = log_r - jax.scipy.special.logsumexp(log_K + f_new[:, None], axis=0)
            diff = jnp.max(jnp.abs(g_new - g))
            return f_new, g_new, it + 1, diff

        def cond(state):
            _, _, it, diff = state
            return (it < max_iter) & (diff > tol)

        f0 = jnp.zeros(K)
        g0 = jnp.zeros(K)
        # Run one iteration to initialize diff
        f1, g1, _, diff0 = body((f0, g0, 0, jnp.inf))
        f_final, g_final, _, _ = lax.while_loop(cond, body, (f1, g1, 1, diff0))

        log_P = f_final[:, None] + log_K + g_final[None, :]
        P = jnp.exp(log_P)
        return P

    @jit
    def _round_plan_jax_single(P):
        """
        Round a (K, K) transport plan to a permutation.
        Greedy collision resolution using lax.fori_loop.
        """
        K = P.shape[0]
        # Initial argmax
        zs = jnp.argmax(P, axis=1)

        def resolve_row(i, state):
            zs, used = state
            col = zs[i]
            is_collision = used[col]

            # Build preference order for this row (descending P value)
            order = jnp.argsort(-P[i])
            # Find first unused column in preference order
            unused_mask = ~used[order]
            # argmax on bool gives first True
            first_free_idx = jnp.argmax(unused_mask)
            alt_col = order[first_free_idx]

            new_col = jnp.where(is_collision, alt_col, col)
            zs = zs.at[i].set(new_col)
            used = used.at[new_col].set(True)
            return zs, used

        used0 = jnp.zeros(K, dtype=bool)
        zs_final, _ = lax.fori_loop(0, K, resolve_row, (zs, used0))
        return zs_final

    # Batched versions via vmap
    _sinkhorn_jax_batch_plans = None  # lazily compiled

    def _sinkhorn_jax(cost_matrices, reg, max_iter, tol):
        """JAX backend: vmap over batch dimension."""
        global _sinkhorn_jax_batch_plans

        cost_jax = jnp.asarray(cost_matrices)
        N, K, _ = cost_jax.shape
        reg_jax = jnp.float64(reg)
        max_iter_static = int(max_iter)
        tol_jax = jnp.float64(tol)

        # vmap the Sinkhorn solver over the batch dimension
        plans = vmap(
            lambda c: _sinkhorn_jax_single(c, reg_jax, max_iter_static, tol_jax)
        )(cost_jax)  # (N, K, K)

        # vmap the rounding over the batch dimension
        zs_all = vmap(_round_plan_jax_single)(plans)  # (N, K)

        # Convert back to numpy
        zs_np = np.asarray(zs_all, dtype=np.int32)
        ys_np = np.tile(np.arange(K, dtype=np.int32), (N, 1))
        return ys_np, zs_np


# ===================================================================
# Numba backend
# ===================================================================

if _HAS_NUMBA:

    @nb.njit(cache=True)
    def _logsumexp_rows(M):
        """Row-wise logsumexp for (K, K) matrix — returns (K,) vector."""
        K = M.shape[0]
        out = np.empty(K)
        for i in range(K):
            m = M[i, 0]
            for j in range(1, M.shape[1]):
                if M[i, j] > m:
                    m = M[i, j]
            s = 0.0
            for j in range(M.shape[1]):
                s += np.exp(M[i, j] - m)
            out[i] = m + np.log(s)
        return out

    @nb.njit(cache=True)
    def _logsumexp_cols(M):
        """Column-wise logsumexp for (K, K) matrix — returns (K,) vector."""
        K = M.shape[1]
        out = np.empty(K)
        for j in range(K):
            m = M[0, j]
            for i in range(1, M.shape[0]):
                if M[i, j] > m:
                    m = M[i, j]
            s = 0.0
            for i in range(M.shape[0]):
                s += np.exp(M[i, j] - m)
            out[j] = m + np.log(s)
        return out

    @nb.njit(cache=True)
    def _sinkhorn_numba_single(cost, reg, max_iter, tol):
        """
        Standard-domain Sinkhorn for a single (K, K) cost matrix — Numba.
        Falls back to log-domain if overflow/underflow.
        Uses vectorized numpy ops that Numba compiles efficiently.
        """
        K = cost.shape[0]
        K_mat = np.exp(-cost / reg)
        r = np.full(K, 1.0 / K)
        u = np.ones(K)

        for it in range(max_iter):
            Ktu = np.dot(K_mat.T, u)
            v = r / Ktu
            Kv = np.dot(K_mat, v)
            u_new = r / Kv

            # Check for numerical issues — fallback to log domain
            has_nan = False
            for i in range(K):
                if not np.isfinite(u_new[i]):
                    has_nan = True
                    break
            if has_nan:
                # Log-domain fallback
                log_K = -cost / reg
                log_r = -np.log(np.float64(K))
                f = np.zeros(K)
                g = np.zeros(K)
                for it2 in range(max_iter):
                    # f[i] = log_r - logsumexp_j(log_K[i,j] + g[j])
                    M_fg = log_K + g.reshape(1, K)  # broadcast
                    f = log_r - _logsumexp_rows(M_fg)
                    # g[j] = log_r - logsumexp_i(log_K[i,j] + f[i])
                    M_ff = log_K + f.reshape(K, 1)  # broadcast
                    g_new = log_r - _logsumexp_cols(M_ff)
                    max_diff = np.max(np.abs(g_new - g))
                    g = g_new
                    if max_diff < tol:
                        break
                P = np.empty((K, K))
                for i in range(K):
                    for j in range(K):
                        P[i, j] = np.exp(f[i] + log_K[i, j] + g[j])
                return P

            max_diff = 0.0
            for i in range(K):
                d = abs(u_new[i] - u[i])
                if d > max_diff:
                    max_diff = d
            rel_tol = max_diff / (np.max(np.abs(u_new)) + 1e-30)
            u = u_new
            if rel_tol < tol:
                break

        # P = diag(u) @ K_mat @ diag(v)
        P = np.empty((K, K))
        for i in range(K):
            for j in range(K):
                P[i, j] = u[i] * K_mat[i, j] * v[j]
        return P

    @nb.njit(cache=True)
    def _round_plan_numba(P):
        """Greedy argmax rounding — Numba."""
        K = P.shape[0]
        zs = np.empty(K, dtype=np.int32)

        # argmax per row
        for i in range(K):
            best = np.int32(0)
            best_val = P[i, 0]
            for j in range(1, K):
                if P[i, j] > best_val:
                    best_val = P[i, j]
                    best = np.int32(j)
            zs[i] = best

        # Resolve collisions greedily
        used = np.zeros(K, dtype=nb.boolean)
        for i in range(K):
            if used[zs[i]]:
                order = np.argsort(-P[i])
                for idx in range(K):
                    j = np.int32(order[idx])
                    if not used[j]:
                        zs[i] = j
                        break
            used[zs[i]] = True
        return zs

    @nb.njit(parallel=True, cache=True)
    def _sinkhorn_numba_batch(cost_matrices, reg, max_iter, tol):
        """Parallel Sinkhorn over batch dimension — Numba."""
        N = cost_matrices.shape[0]
        K = cost_matrices.shape[1]
        ys_idx = np.zeros((N, K), dtype=np.int32)
        zs_idx = np.zeros((N, K), dtype=np.int32)

        for i in nb.prange(N):
            P = _sinkhorn_numba_single(cost_matrices[i], reg, max_iter, tol)
            zs_idx[i] = _round_plan_numba(P)
            for k in range(K):
                ys_idx[i, k] = np.int32(k)

        return ys_idx, zs_idx

    def _sinkhorn_numba(cost_matrices, reg, max_iter, tol):
        """Numba backend entry point."""
        cost_matrices = np.ascontiguousarray(cost_matrices, dtype=np.float64)
        return _sinkhorn_numba_batch(cost_matrices, reg, int(max_iter), tol)


# ===================================================================
# Public API — auto-dispatch
# ===================================================================

def sinkhorn_assignment(cost_matrices, reg=None, max_iter=200, tol=1e-6,
                        backend="auto"):
    """
    Solve a batch of assignment problems using Sinkhorn-Knopp.

    Parameters
    ----------
    cost_matrices : np.ndarray, shape (N, K, K)
        Batch of cost matrices.
    reg : float or None
        Entropic regularization parameter.  If None, defaults to
        0.01 * mean(cost_matrices) which gives near-exact assignments.
    max_iter : int
        Maximum Sinkhorn iterations.
    tol : float
        Convergence tolerance.
    backend : str
        Backend to use: "auto" (default), "jax", "numba", or "numpy".

    Returns
    -------
    ys_idx : np.ndarray, shape (N, K), int32
    zs_idx : np.ndarray, shape (N, K), int32
    """
    cost_matrices = np.asarray(cost_matrices, dtype=np.float64)
    N, K, K2 = cost_matrices.shape
    assert K == K2, f"Cost matrices must be square, got ({K}, {K2})"

    if reg is None:
        mean_cost = np.mean(cost_matrices[np.isfinite(cost_matrices)])
        reg = max(0.01 * mean_cost, 1e-10)

    if backend == "auto":
        # Numba is consistently faster than JAX for Sinkhorn
        # (parallel prange + compiled standard-domain iterations)
        if _HAS_NUMBA:
            backend = "numba"
        elif _HAS_JAX:
            backend = "jax"
        else:
            backend = "numpy"

    if backend == "jax":
        if not _HAS_JAX:
            raise ImportError("JAX not available")
        return _sinkhorn_jax(cost_matrices, reg, max_iter, tol)
    elif backend == "numba":
        if not _HAS_NUMBA:
            raise ImportError("Numba not available")
        return _sinkhorn_numba(cost_matrices, reg, max_iter, tol)
    else:
        return _sinkhorn_numpy(cost_matrices, reg, max_iter, tol)
