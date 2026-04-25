"""ABC-Gibbs sampler for Bernoulli rain probability models.

The sampler alternates:
1. component-wise intercept updates alpha_k | beta, y_k
2. global coefficient updates beta | alpha, y

Each ABC step draws candidates from the corresponding prior, simulates binary
rain observations, and keeps the candidate with the smallest Hamming distance.
"""

from __future__ import annotations

import time as _time
from typing import Tuple

import numpy as np
from jax import random
from scipy.special import expit


def _rng_from_key(key) -> np.random.Generator:
    key_np = np.asarray(key)
    seed = int(key_np[0]) * 1_000_003 + int(key_np[1])
    return np.random.default_rng(seed % (2**32 - 1))


def _simulate_component(rng: np.random.Generator, alpha: np.ndarray, beta: np.ndarray, x_k: np.ndarray) -> np.ndarray:
    eta = alpha[:, None] + x_k[None, :, :] @ beta
    probs = expit(eta)
    return rng.binomial(1, probs).astype(np.float32)


def _simulate_global(rng: np.random.Generator, alpha: np.ndarray, beta: np.ndarray, x_cov: np.ndarray) -> np.ndarray:
    eta = alpha[None, :, None] + np.einsum("knf,mf->mkn", x_cov, beta)
    probs = expit(eta)
    return rng.binomial(1, probs).astype(np.float32)


def setup_gibbs_functions_weather(model):
    """Create ABC-Gibbs update functions for BernoulliLogitWithCovariates."""
    k_comp = model.K
    n_features = model.n_features
    x_cov = np.asarray(model.X_cov, dtype=np.float32)

    def abc_locals(rng: np.random.Generator, m_loc: int, y_obs_2d: np.ndarray, beta_current: np.ndarray):
        alpha_new = np.zeros((k_comp, 1), dtype=np.float64)
        eps_per_k = np.zeros(k_comp, dtype=np.float64)

        for k in range(k_comp):
            candidates = rng.normal(model.mu_alpha, model.sigma_alpha, size=m_loc)
            z_k = _simulate_component(rng, candidates, beta_current, x_cov[k])
            dists = np.sum(z_k != y_obs_2d[k][None, :], axis=1)
            best = int(np.argmin(dists))
            alpha_new[k, 0] = candidates[best]
            eps_per_k[k] = float(dists[best])

        return alpha_new, eps_per_k, k_comp * m_loc

    def abc_global(rng: np.random.Generator, m_glob: int, y_obs_2d: np.ndarray, alpha_current: np.ndarray):
        beta_candidates = rng.normal(model.mu_beta, model.sigma_beta, size=(m_glob, n_features))
        z = _simulate_global(rng, alpha_current[:, 0], beta_candidates, x_cov)
        dists = np.sum(z != y_obs_2d[None, :, :], axis=(1, 2))
        best = int(np.argmin(dists))
        return beta_candidates[best].astype(np.float64), float(dists[best]), k_comp * m_glob

    return abc_locals, abc_global


def run_gibbs_sampler_weather(
    key,
    model,
    y_obs_2d: np.ndarray,
    T: int,
    M_loc: int,
    M_glob: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run ABC-Gibbs for Bernoulli rain probability inference."""
    y_obs_2d = np.asarray(y_obs_2d, dtype=np.float32)
    if y_obs_2d.ndim == 1:
        y_obs_2d = y_obs_2d[None, :]

    k_comp = model.K
    n_features = model.n_features
    abc_locals, abc_global = setup_gibbs_functions_weather(model)
    rng = _rng_from_key(key)

    alpha_chain = np.zeros((T + 1, k_comp, 1), dtype=np.float64)
    beta_chain = np.zeros((T + 1, n_features), dtype=np.float64)
    eps_loc = np.zeros((T, k_comp), dtype=np.float64)
    eps_glob = np.zeros(T, dtype=np.float64)
    times = np.zeros(T, dtype=np.float64)

    alpha_chain[0, :, 0] = rng.normal(model.mu_alpha, model.sigma_alpha, size=k_comp)
    beta_chain[0] = rng.normal(model.mu_beta, model.sigma_beta, size=n_features)
    n_sim_per_iter = k_comp * (M_loc + M_glob)

    for t in range(T):
        t0 = _time.perf_counter()
        key, _, _ = random.split(key, 3)

        alpha_chain[t + 1], eps_loc[t], _ = abc_locals(
            rng,
            M_loc,
            y_obs_2d,
            beta_chain[t],
        )
        beta_chain[t + 1], eps_glob[t], _ = abc_global(
            rng,
            M_glob,
            y_obs_2d,
            alpha_chain[t + 1],
        )

        times[t] = _time.perf_counter() - t0
        if (t + 1) % 100 == 0 or t == 0:
            print(
                f"  Gibbs iter {t + 1}/{T}: eps_glob={eps_glob[t]:.1f}, "
                f"eps_loc_mean={np.mean(eps_loc[t]):.1f}, time={times[t]:.2f}s"
            )

    return alpha_chain, beta_chain, eps_loc, eps_glob, times, n_sim_per_iter
