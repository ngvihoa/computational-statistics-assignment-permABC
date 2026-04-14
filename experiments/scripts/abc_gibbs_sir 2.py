"""
ABC-Gibbs sampler for the SIR_real_world model.

Block Gibbs that alternates:
  1. (I0_k, R0_k, gamma_k) | R0, y   for each k = 1..K  (local distance d(y_k, z_k))
  2. R0 | locals, y                                       (global distance sum_k d(y_k, z_k))

Each ABC step draws M candidates from the prior conditional, simulates data,
and keeps the candidate minimizing the relevant distance.

Usage:
    from abc_gibbs_sir import run_gibbs_sampler_sir
    mus, r0s, eps_loc, eps_glob, times, n_sim = run_gibbs_sampler_sir(
        key, model, y_obs_2d, T=500, M_loc=50, M_glob=100
    )
"""
import numpy as np
import time as _time
from jax import random
import jax.numpy as jnp

from permabc.utils.functions import Theta
from permabc.models.SIR import simulate_sir_jax


# =====================================================================
# Distance helpers
# =====================================================================

def distance_component_k(z_k, y_k):
    """Distance for one region k: sum of squared differences over time.
    z_k, y_k: (n_obs,)
    """
    return float(np.sum((z_k - y_k) ** 2))


def distance_total(z, y):
    """Total distance across K regions. z, y: (K, n_obs)."""
    return float(np.sum((z - y) ** 2))


# =====================================================================
# Core Gibbs functions
# =====================================================================

def setup_gibbs_functions_sir(model):
    """Create ABC-Gibbs step functions for SIR_real_world.

    Parameters
    ----------
    model : SIR_real_world
        Model instance.

    Returns
    -------
    ABClocals : callable — update local params for each k
    ABCglobal : callable — update R0 global
    """
    K = model.K
    n_obs = model.n_obs
    n_pop = model.n_pop
    support_loc = np.array(model.support_par_loc)   # (3, 2): I0, R0_init, gamma
    support_glob = np.array(model.support_par_glob)  # (1, 2): R0

    def ABClocals(rng, jax_key, M, y_obs_2d, r0_current):
        """ABC step for (I0_k, R0_init_k, gamma_k) | R0, y, for each k independently.

        For each k:
          - draw M candidates from the prior Uniform(low, high)
          - simulate SIR with those local params + current R0
          - keep the candidate minimizing d(z_k, y_k)

        Parameters
        ----------
        rng : np.random.Generator
        jax_key : jax PRNGKey (for stochastic SIR simulation)
        M : int
        y_obs_2d : (K, n_obs)
        r0_current : float

        Returns
        -------
        locals_new : (K, 3) array
        eps_per_k : (K,) array
        n_sim : int
        """
        locals_new = np.zeros((K, 3))
        eps_per_k = np.zeros(K)

        for k in range(K):
            # Draw M candidates for local params of region k from uniform prior
            loc_candidates = rng.uniform(
                low=support_loc[:, 0],
                high=support_loc[:, 1],
                size=(M, 3),
            )  # (M, 3): I0, R0_init, gamma

            I0_cand = jnp.array(loc_candidates[:, 0:1])    # (M, 1)
            R0_init_cand = jnp.array(loc_candidates[:, 1:2])  # (M, 1)
            gamma_cand = jnp.array(loc_candidates[:, 2:3])  # (M, 1)
            r0_arr = jnp.full((M, 1), r0_current)
            beta_cand = r0_arr * gamma_cand
            S0_cand = jnp.maximum(0.0, n_pop - I0_cand - R0_init_cand)

            jax_key_k = random.fold_in(jax_key, k)
            zs_k = simulate_sir_jax(
                S0_cand, I0_cand, R0_init_cand,
                beta_cand, gamma_cand,
                n_pop=n_pop, n_obs=n_obs,
                noise_key=jax_key_k,
            )  # (M, 1, n_obs)
            zs_k = np.array(zs_k[:, 0, :])  # (M, n_obs)

            obs_k = y_obs_2d[k]  # (n_obs,)
            dists = np.sum((zs_k - obs_k[None, :]) ** 2, axis=1)  # (M,)
            best = np.argmin(dists)
            locals_new[k] = loc_candidates[best]
            eps_per_k[k] = dists[best]

        n_sim = K * M
        return locals_new, eps_per_k, n_sim

    def ABCglobal(rng, jax_key, M, y_obs_2d, locals_current):
        """ABC step for R0 | locals, y.

        Draw M candidates R0 ~ Uniform(low_r0, high_r0),
        simulate all K regions using current locals + candidate R0,
        compute total distance, keep argmin.

        Parameters
        ----------
        rng : np.random.Generator
        jax_key : jax PRNGKey
        M : int
        y_obs_2d : (K, n_obs)
        locals_current : (K, 3) array

        Returns
        -------
        r0_new : float
        eps_total : float
        n_sim : int
        """
        r0_candidates = rng.uniform(
            low=support_glob[0, 0],
            high=support_glob[0, 1],
            size=M,
        )

        I0_fixed = jnp.array(locals_current[None, :, 0].repeat(M, axis=0))   # (M, K)
        R0_init_fixed = jnp.array(locals_current[None, :, 1].repeat(M, axis=0))
        gamma_fixed = jnp.array(locals_current[None, :, 2].repeat(M, axis=0))
        r0_arr = jnp.array(r0_candidates[:, None])  # (M, 1)
        beta = r0_arr * gamma_fixed  # (M, K)
        S0 = jnp.maximum(0.0, n_pop - I0_fixed - R0_init_fixed)

        zs = simulate_sir_jax(
            S0, I0_fixed, R0_init_fixed,
            beta, gamma_fixed,
            n_pop=n_pop, n_obs=n_obs,
            noise_key=jax_key,
        )  # (M, K, n_obs)
        zs = np.array(zs)

        dists = np.sum((zs - y_obs_2d[None, :, :]) ** 2, axis=(1, 2))  # (M,)
        best = np.argmin(dists)
        return float(r0_candidates[best]), float(dists[best]), M

    return ABClocals, ABCglobal


def run_gibbs_sampler_sir(key, model, y_obs_2d, T, M_loc, M_glob):
    """Run ABC-Gibbs sampler for SIR_real_world.

    Parameters
    ----------
    key : jax.random.PRNGKey
    model : SIR_real_world
    y_obs_2d : (K, n_obs) array — observed data
    T : int — number of Gibbs iterations
    M_loc : int — number of prior candidates per component
    M_glob : int — number of prior candidates for R0

    Returns
    -------
    locals_chain : (T+1, K, 3) — chain of local params
    r0_chain : (T+1,) — chain of R0 values
    eps_loc : (T, K) — per-component distances at each iteration
    eps_glob : (T,) — total distance at each iteration
    times : (T,) — wall-clock time per iteration
    n_sim_per_iter : int — simulations per iteration
    """
    K = model.K
    ABClocals, ABCglobal = setup_gibbs_functions_sir(model)

    locals_chain = np.zeros((T + 1, K, 3))
    r0_chain = np.zeros(T + 1)
    eps_loc = np.zeros((T, K))
    eps_glob = np.zeros(T)
    times = np.zeros(T)

    support_loc = np.array(model.support_par_loc)
    support_glob = np.array(model.support_par_glob)

    # Initialize from prior
    rng = np.random.default_rng(int(key[0]))
    locals_chain[0] = rng.uniform(
        low=support_loc[:, 0], high=support_loc[:, 1], size=(K, 3)
    )
    r0_chain[0] = rng.uniform(low=support_glob[0, 0], high=support_glob[0, 1])

    n_sim_per_iter = K * M_loc + M_glob

    for t in range(T):
        t0 = _time.perf_counter()

        key, key_loc, key_glob = random.split(key, 3)

        # Step 1: update locals for each k
        locals_chain[t + 1], eps_loc[t], _ = ABClocals(
            rng, key_loc, M_loc, y_obs_2d, r0_chain[t]
        )
        # Step 2: update R0 global
        r0_chain[t + 1], eps_glob[t], _ = ABCglobal(
            rng, key_glob, M_glob, y_obs_2d, locals_chain[t + 1]
        )

        times[t] = _time.perf_counter() - t0
        if (t + 1) % 100 == 0 or t == 0:
            print(f"  Gibbs iter {t+1}/{T}: R0={r0_chain[t+1]:.3f}, "
                  f"eps_glob={eps_glob[t]:.1f}, "
                  f"eps_loc_mean={np.mean(eps_loc[t]):.1f}, "
                  f"time={times[t]:.2f}s")

    return locals_chain, r0_chain, eps_loc, eps_glob, times, n_sim_per_iter
