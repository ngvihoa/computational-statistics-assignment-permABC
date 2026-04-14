"""
ABC-Gibbs sampler for the GaussianWithNoSummaryStats model.

Block Gibbs that alternates:
  1. μ_k | σ², y   for each k = 1..K  (ABC step on local params)
  2. σ²  | μ, y                        (ABC step on global param)

Each ABC step draws M candidates from the prior conditional and keeps
the one minimizing a distance to the observed data.

Adapted from the ABC-Gibbs implementation in fig5_posterior_comparison_perm_smc_vs_gibbs.py
(which targets GaussianWithCorrelatedParams) to GaussianWithNoSummaryStats.

Usage:
    from abc_gibbs_gaussian import run_gibbs_for_benchmark
    rows = run_gibbs_for_benchmark(key, model, y_obs, K, N_sim_budget)
"""
import sys
import numpy as np
import time as _time
from pathlib import Path
from jax import random
from scipy.stats import invgamma

from permabc.utils.functions import Theta


# =====================================================================
# Core Gibbs functions
# =====================================================================

def _distance_component(sim_k, obs_k):
    """Squared distance between sorted sim and obs for one component.
    sim_k, obs_k: (n_obs,)
    """
    return float(np.sum((np.sort(sim_k) - np.sort(obs_k)) ** 2))


def _distance_all_components(sim, obs):
    """Total distance across K components. sim, obs: (K, n_obs)."""
    return float(np.sum((np.sort(sim, axis=1) - np.sort(obs, axis=1)) ** 2))


def setup_gibbs_functions(model):
    """Create ABC-Gibbs step functions for GaussianWithNoSummaryStats.

    Parameters
    ----------
    model : GaussianWithNoSummaryStats
        Model instance with attributes: K, n_obs, mu_0, sigma_0, alpha, beta.

    Returns
    -------
    ABCmus : callable(rng, M, y_obs_2d, sigma2_current) -> (mus_new, eps_per_k)
    ABCsigma2 : callable(rng, M, y_obs_2d, mus_current) -> (sigma2_new, eps_total)
    """
    K = model.K
    n_obs = model.n_obs
    mu_0 = model.mu_0
    sigma_0 = model.sigma_0
    alpha = model.alpha
    beta = model.beta

    def ABCmus(rng, M, y_obs_2d, sigma2_current):
        """ABC step for mu_k | sigma2, y.

        For each component k, draw M candidates mu_k ~ N(mu_0, sigma_0^2),
        simulate n_obs observations N(mu_k, sigma2_current), sort them,
        keep the candidate minimizing distance to y_obs[k].

        Parameters
        ----------
        rng : np.random.Generator
        M : int
        y_obs_2d : (K, n_obs)
        sigma2_current : float

        Returns
        -------
        mus_new : (K,) array
        eps_per_k : (K,) array — per-component distances
        """
        sigma = np.sqrt(max(sigma2_current, 1e-15))
        mus_new = np.zeros(K)
        eps_per_k = np.zeros(K)

        for k in range(K):
            # Draw M candidates from prior N(mu_0, sigma_0^2)
            mu_candidates = rng.normal(mu_0, sigma_0, size=M)
            # Simulate n_obs observations for each candidate
            noise = rng.normal(size=(M, n_obs))
            sims = mu_candidates[:, None] + sigma * noise  # (M, n_obs)
            sims.sort(axis=1)
            # Distance to sorted observed data for component k
            obs_k = np.sort(y_obs_2d[k])
            dists = np.sum((sims - obs_k[None, :]) ** 2, axis=1)  # (M,)
            best = np.argmin(dists)
            mus_new[k] = mu_candidates[best]
            eps_per_k[k] = dists[best]

        return mus_new, eps_per_k

    def ABCsigma2(rng, M, y_obs_2d, mus_current):
        """ABC step for sigma2 | mu, y.

        Draw M candidates sigma2 ~ InvGamma(alpha, beta),
        simulate all K components using current mus and candidate sigma2,
        sort observations, compute total distance, keep argmin.

        Parameters
        ----------
        rng : np.random.Generator
        M : int
        y_obs_2d : (K, n_obs)
        mus_current : (K,) array

        Returns
        -------
        sigma2_new : float
        eps_total : float
        """
        # Draw M candidates from InvGamma prior
        sigma2_candidates = invgamma.rvs(alpha, scale=beta, size=M, random_state=rng)
        sigma2_candidates = np.maximum(sigma2_candidates, 1e-15)

        # Simulate data for each candidate
        dists = np.empty(M)
        obs_sorted = np.sort(y_obs_2d, axis=1)  # (K, n_obs)

        for m in range(M):
            sigma_m = np.sqrt(sigma2_candidates[m])
            noise = rng.normal(size=(K, n_obs))
            sims = mus_current[:, None] + sigma_m * noise  # (K, n_obs)
            sims.sort(axis=1)
            dists[m] = np.sum((sims - obs_sorted) ** 2)

        best = np.argmin(dists)
        return float(sigma2_candidates[best]), float(dists[best])

    return ABCmus, ABCsigma2


def run_gibbs_sampler(key, model, y_obs_2d, T, M_mu, M_sigma2):
    """Run ABC-Gibbs sampler for GaussianWithNoSummaryStats.

    Parameters
    ----------
    key : jax.random.PRNGKey
    model : GaussianWithNoSummaryStats
    y_obs_2d : (K, n_obs) array — observed data (sorted within components)
    T : int — number of Gibbs iterations
    M_mu : int — number of prior candidates for each mu_k step
    M_sigma2 : int — number of prior candidates for sigma2 step

    Returns
    -------
    mus : (T+1, K) — chain of mu values
    sigma2s : (T+1,) — chain of sigma2 values
    eps_mus : (T, K) — per-component distances at each iteration
    eps_sigma2 : (T,) — total distance at each iteration
    times : (T,) — wall-clock time per iteration
    n_sim_per_iter : int — simulations per iteration (K*M_mu + M_sigma2)
    """
    K = model.K
    ABCmus, ABCsigma2 = setup_gibbs_functions(model)

    mus = np.zeros((T + 1, K))
    sigma2s = np.zeros(T + 1)
    eps_mus = np.zeros((T, K))
    eps_sigma2 = np.zeros(T)
    times = np.zeros(T)

    # Initialize from prior
    rng = np.random.default_rng(int(key[0]))
    mus[0] = rng.normal(model.mu_0, model.sigma_0, size=K)
    sigma2s[0] = invgamma.rvs(model.alpha, scale=model.beta, random_state=rng)

    n_sim_per_iter = K * M_mu + M_sigma2

    for t in range(T):
        t0 = _time.perf_counter()
        # Step 1: update each mu_k | sigma2, y
        mus[t + 1], eps_mus[t] = ABCmus(rng, M_mu, y_obs_2d, sigma2s[t])
        # Step 2: update sigma2 | mu, y
        sigma2s[t + 1], eps_sigma2[t] = ABCsigma2(rng, M_sigma2, y_obs_2d, mus[t + 1])
        times[t] = _time.perf_counter() - t0

    return mus, sigma2s, eps_mus, eps_sigma2, times, n_sim_per_iter


# =====================================================================
# Benchmark interface
# =====================================================================

def run_gibbs_for_benchmark(key, model, y_obs, K, N_sim_budget,
                            T_iter=1000, n_points=10):
    """Run ABC-Gibbs at multiple budget levels and return one diagnostic per level.

    For each of *n_points* log-spaced budgets up to *N_sim_budget*, runs an
    independent Gibbs chain of T_iter iterations (no burn-in).
    M_mu is derived from the per-run budget:  M_mu = budget / (T_iter * (K+2)),
    and M_sigma2 = 2 * M_mu.

    Parameters
    ----------
    key : jax.random.PRNGKey
    model : GaussianWithNoSummaryStats
    y_obs : observed data
    K : int
    N_sim_budget : int — maximum total simulation budget
    T_iter : int — number of iterations per run (default 1000)
    n_points : int — number of budget levels (default 10)

    Returns
    -------
    records : list of dicts with benchmark metrics (one per budget level)
    """
    # diagnostics.py is in the same directory as this file
    _scripts_dir = str(Path(__file__).resolve().parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from diagnostics import (
        empirical_kl_sigma2, build_sigma2_reference_bins,
        expected_neg_log_joint_true,
    )

    T = T_iter

    # Prepare y_obs
    y2d = np.asarray(y_obs)
    if y2d.ndim == 3 and y2d.shape[0] == 1:
        y2d = y2d[0]

    sigma2_edges = build_sigma2_reference_bins(model, y_obs, nbins=80)

    # Log-spaced budgets from min_budget to N_sim_budget
    min_M_mu = 10
    min_budget = T * (K + 2) * min_M_mu
    budgets = np.unique(np.geomspace(
        max(min_budget, N_sim_budget / 100),
        N_sim_budget,
        n_points,
    ).astype(int))

    print(f"  ABC-Gibbs: T={T}, {len(budgets)} budget levels, "
          f"max_budget={N_sim_budget:,}")

    records = []
    for i, budget in enumerate(budgets):
        M_mu = max(min_M_mu, int(budget / (T * (K + 2))))
        M_sigma2 = 2 * M_mu
        n_sim_per_iter = K * M_mu + M_sigma2
        actual_sim = T * n_sim_per_iter

        # Fresh key for each independent run
        key, subkey = random.split(key)

        mus, sigma2s, eps_mus, eps_sigma2, times, _ = run_gibbs_sampler(
            subkey, model, y2d, T, M_mu, M_sigma2,
        )
        time_total = float(np.sum(times))

        # Use full chain (no burn-in)
        chain_mus = mus[1:]       # (T, K) — skip initialization
        chain_s2 = sigma2s[1:]    # (T,)
        n_samples = len(chain_mus)

        thetas_gibbs = Theta(
            loc=chain_mus[:, :, None],
            glob=chain_s2[:, None],
        )
        weights = np.ones(n_samples) / n_samples

        kl_s2 = empirical_kl_sigma2(model, y_obs, thetas_gibbs,
                                     weights=weights, edges=sigma2_edges,
                                     direction="q_vs_p")
        score = expected_neg_log_joint_true(model, y_obs, thetas_gibbs,
                                            weights=weights, perm=None)
        eps_val = float(np.mean(np.sqrt(eps_sigma2)))

        # Normalize per 1000 unique particles
        n_unique = min(n_samples, len(np.unique(chain_s2)))
        denom = K * max(n_unique, 1)
        n_sim_norm = actual_sim / denom * 1000
        time_norm = time_total / denom * 1000

        records.append({
            "method": "ABC-Gibbs",
            "n_sim": n_sim_norm,
            "time": time_norm,
            "n_sim_raw": actual_sim,
            "time_raw": time_total,
            "epsilon": eps_val,
            "kl_sigma2": kl_s2,
            "score_joint": score,
        })
        print(f"    [{i+1}/{len(budgets)}] M_mu={M_mu}, M_s2={M_sigma2}, "
              f"sim={actual_sim:,}, KL_s2={kl_s2:.4f}, score={score:.1f}")

    return records
