#!/usr/bin/env python3
"""
Performance comparison runner for ABC methods.

This script runs comprehensive performance comparisons between various ABC algorithms.
Used by fig4 and fig6 with different osum settings.

Usage:
    python run_performance_comparison.py --K 20 --K_outliers 4 --osum
    python run_performance_comparison.py --K 20 --K_outliers 0 --no-osum --seed 42
    python run_performance_comparison.py --rerun experiments/results/performance_K_20_outliers_4_osum_True_seed_42.pkl
    python run_performance_comparison.py --K 5 --methods osum --seed 42   # run only OS/UM, append to existing CSV
    python run_performance_comparison.py --K 5 --methods smc osum --seed 42  # run SMC + OS/UM only
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
import pickle
import argparse
import time
import datetime
import csv
import errno
from jax import random
from pathlib import Path

from permabc.algorithms.smc import abc_smc, perm_abc_smc
try:
    from permabc.algorithms.pmc import abc_pmc
except Exception:  # pragma: no cover
    abc_pmc = None
try:
    from permabc.algorithms.over_sampling import perm_abc_smc_os
except Exception:  # pragma: no cover
    perm_abc_smc_os = None
try:
    from permabc.algorithms.under_matching import perm_abc_smc_um
except Exception:  # pragma: no cover
    perm_abc_smc_um = None
from permabc.core.kernels import KernelTruncatedRW
from permabc.core.distances import optimal_index_distance
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta

from diagnostics import (
    build_sigma2_reference_bins,
    empirical_kl_sigma2,
    empirical_kl_sigma2_abc_vs_true,
    empirical_kl_sigma2_true_vs_abc,
    expected_neg_log_joint_true,
    empirical_kl_mu_avg,
    empirical_w2_sigma2,
    empirical_w2_mu_avg,
    sliced_w2_joint,
)

N_PARTICLES_DEFAULT = 1000


# =============================================================================
# Utilities
# =============================================================================

FACTOR = 1.
def select_indices(L, m, factor=FACTOR):
    """Extract m elements with exponential spacing."""
    n = len(L)
    if m >= n:
        return list(range(n))
    if m <= 0:
        return []

    indices = []
    for i in range(m):
        t = i / (m - 1) if m > 1 else 0
        if factor != 1:
            mapped_t = (factor ** t - 1) / (factor - 1)
        else:
            mapped_t = t
        indice = int(mapped_t * (n - 1))
        indices.append(indice)

    return sorted(list(set(indices)))


def _safe_array_at(seq, idx):
    """Return seq[idx] if available, else None."""
    if seq is None or idx < 0 or idx >= len(seq):
        return None
    return seq[idx]


# =============================================================================
# DataFrame helpers (work with both pandas DataFrames and list-of-dicts)
# =============================================================================

def _df_unique_methods(df):
    if pd is not None and hasattr(df, "columns"):
        return list(df["method"].unique())
    return sorted({r["method"] for r in df})


def _df_filter_method(df, method):
    if pd is not None and hasattr(df, "columns"):
        return df[df["method"] == method]
    return [r for r in df if r["method"] == method]


def _df_sort_by(df, key):
    if pd is not None and hasattr(df, "sort_values"):
        return df.sort_values(key)
    return sorted(df, key=lambda r: r[key])


def _df_col(df, key):
    if pd is not None and hasattr(df, "__getitem__") and not isinstance(df, list):
        return np.asarray(df[key])
    return np.asarray([r[key] for r in df], dtype=float)


# =============================================================================
# Experiment setup
# =============================================================================

def setup_experiment(K=20, K_outliers=4, seed=42, N_points=1000000, N_particles=1000):
    """Setup experimental parameters and generate synthetic data with outliers."""
    stopping_rate = 0.0

    n = 10
    sigma0 = 10
    alpha, beta = 5, 5

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)

    model = GaussianWithNoSummaryStats(K=K, n_obs=n, sigma_0=sigma0, alpha=alpha, beta=beta)
    true_theta = model.prior_generator(subkey, 1)
    true_theta = Theta(loc=true_theta.loc, glob=np.array([1.])[None, :])

    for i in range(K_outliers):
        key, subkey = random.split(key)
        sign = int(np.asarray(random.choice(subkey, a=np.array([-1, 1]), shape=(1,)))[0])
        key, subkey = random.split(key)
        if sign == 1:
            outlier_val = float(np.asarray(random.uniform(subkey, shape=(1,), minval=-3 * sigma0, maxval=-2 * sigma0))[0])
        else:
            outlier_val = float(np.asarray(random.uniform(subkey, shape=(1,), minval=2 * sigma0, maxval=3 * sigma0))[0])
        loc = np.array(true_theta.loc, copy=True)
        loc[0, i, 0] = outlier_val
        true_theta = Theta(loc=loc, glob=true_theta.glob)

    key, subkey = random.split(key)
    y_obs = model.data_generator(subkey, true_theta)

    return model, y_obs, true_theta, key, N_points, N_particles, stopping_rate, K


# =============================================================================
# Algorithm runners
# =============================================================================

def run_vanilla_methods(key, model, N_points, y_obs):
    """Run vanilla ABC methods for baseline comparison."""
    print("Running Vanilla methods...")

    # Vanilla ABC
    model.reset_weights_distance()
    time_0 = time.time()
    key, key_theta, key_data = random.split(key, 3)
    thetas = model.prior_generator(key_theta, N_points)
    zs = model.data_generator(key_data, thetas)
    dists = model.distance(zs, y_obs)
    time_van = time.time() - time_0
    print(f"Time for vanilla ABC: {time_van:.2f}s")

    # permABC Vanilla
    model.reset_weights_distance()
    time_0 = time.time()
    key, key_theta, key_data = random.split(key, 3)
    thetas = model.prior_generator(key_theta, N_points)
    zs = model.data_generator(key_data, thetas)
    dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs)
    time_perm_van = time.time() - time_0
    print(f"Time for permABC Vanilla: {time_perm_van:.2f}s")

    results = {
        'vanilla_abc': {'dists': dists, 'time': time_van, 'thetas': thetas, 'zs': zs},
        'vanilla_perm': {'dists': dists_perm, 'time': time_perm_van, 'thetas': thetas, 'zs': zs,
                         'ys_index': ys_index, 'zs_index': zs_index}
    }
    return key, results


def run_smc_methods(key, model, N_particles, y_obs, stopping_rate, N_points, K,
                    out_perm_pilot=None):
    """Run SMC-based ABC methods (ABC-SMC + permABC-SMC).

    If out_perm_pilot is provided (from budget discovery), it is reused as the
    permABC-SMC result instead of running it a second time.
    """
    kernel = KernelTruncatedRW
    results = {}

    # ABC SMC
    print("Running ABC SMC...")
    key, subkey = random.split(key)
    model.reset_weights_distance()
    out_smc = abc_smc(
        key=subkey, model=model, n_particles=N_particles, epsilon_target=0, y_obs=y_obs,
        kernel=kernel, verbose=1, Final_iteration=0, update_weights_distance=False,
        stopping_accept_rate=stopping_rate, N_sim_max=N_points*K
    )
    results['abc_smc'] = out_smc

    # permABC SMC — reuse budget discovery run if available
    if out_perm_pilot is not None:
        print("Reusing permABC-SMC from budget discovery (no double run).")
        results['perm_abc_smc'] = out_perm_pilot
    else:
        print("Running permABC SMC...")
        key, subkey = random.split(key)
        model.reset_weights_distance()
        out_perm_smc = perm_abc_smc(
            key=subkey, model=model, n_particles=N_particles, epsilon_target=0, y_obs=y_obs,
            kernel=kernel, verbose=1, Final_iteration=0, update_weights_distance=False,
            stopping_accept_rate=stopping_rate, N_sim_max=N_points*K
        )
        results['perm_abc_smc'] = out_perm_smc

    return key, results


def run_pmc_method(key, model, N_particles, y_obs, stopping_rate, N_points, K):
    """Run ABC-PMC method."""
    print("Running ABC PMC...")
    if abc_pmc is None:
        print("  Skipping ABC PMC (abc_pmc import failed: likely numba/numpy incompatibility).")
        return key, None

    key, subkey = random.split(key)
    model.reset_weights_distance()
    out_pmc = abc_pmc(
        key=subkey, model=model, n_particles=N_particles, epsilon_target=0, y_obs=y_obs,
        alpha=0.95, verbose=1, update_weights_distance=False,
        stopping_accept_rate=stopping_rate, N_sim_max=N_points*K
    )
    return key, out_pmc


def run_osum_methods(key, model, N_particles, y_obs, K):
    """
    Run over-sampling (OS) and under-matching (UM) variants.

    Over-sampling: for each M0 > K, calibrate epsilon at a fixed quantile then
    run perm_abc_smc_os.  Under-matching: for each L0 < K, same calibration
    then run perm_abc_smc_um.
    """
    print("Running OSUM methods...")

    alpha_epsilon = 0.95
    alpha_M = 0.9
    alpha_L = 0.9
    N_epsilon = 10000
    kernel = KernelTruncatedRW

    results = {'over_sampling': {}, 'under_matching': {}}

    if perm_abc_smc_os is None or perm_abc_smc_um is None:
        raise ImportError(
            "OS/UM algorithms could not be imported (often due to numba/numpy incompatibility). "
            "Install compatible deps or run without --osum."
        )

    # Over-sampling
    M0s = np.array([1.5*K, 2*K, 5*K, 7*K, 10*K, 15*K, 20*K, 25*K], dtype=int)
    os_results = {
        'M0': [], 'epsilons': [], 'time': [], 'n_sim': [], 'unique': [], 'full_results': [],
    }

    for M0 in M0s:
        print(f"  Over-sampling M0 = {M0}")
        try:
            key, subkey = random.split(key)
            thetas = model.prior_generator(subkey, N_epsilon, M0)
            key, subkey = random.split(key)
            zs = model.data_generator(subkey, thetas)
            dists_perm, ys_index, zs_index, _ = optimal_index_distance(
                model=model, zs=zs, y_obs=y_obs, epsilon=0, verbose=0, M=M0
            )
            epsilon = np.quantile(dists_perm, alpha_epsilon)

            model.reset_weights_distance()
            key, subkey = random.split(key)
            out_os = perm_abc_smc_os(
                key=subkey, model=model, n_particles=N_particles, y_obs=y_obs,
                kernel=kernel, M_0=M0, epsilon=epsilon, alpha_M=alpha_M,
                update_weights_distance=False, verbose=1, Final_iteration=0, duplicate=True
            )

            os_results['M0'].append(int(M0))
            if out_os is not None:
                os_results['epsilons'].append(float(epsilon))
                os_results['n_sim'].append(float(np.sum(out_os["N_sim"])))
                os_results['unique'].append(float(out_os["unique_part"][-1]))
                os_results['time'].append(float(out_os["time_final"]))
                os_results['full_results'].append(out_os)
            else:
                os_results['epsilons'].append(np.nan)
                os_results['n_sim'].append(np.nan)
                os_results['unique'].append(np.nan)
                os_results['time'].append(np.nan)
                os_results['full_results'].append(None)
        except Exception as e:
            print(f"    Failed: {e}")
            os_results['M0'].append(int(M0))
            os_results['epsilons'].append(np.nan)
            os_results['n_sim'].append(np.nan)
            os_results['unique'].append(np.nan)
            os_results['time'].append(np.nan)
            os_results['full_results'].append(None)

    # Under-matching
    L0s = np.array(np.linspace(2, K, K), dtype=int)
    um_results = {'epsilons': [], 'time': [], 'n_sim': [], 'unique': [], 'full_results': []}

    for L0 in L0s:
        print(f"  Under-matching L0 = {L0}")
        try:
            key, subkey = random.split(key)
            thetas = model.prior_generator(subkey, N_epsilon)
            key, subkey = random.split(key)
            zs = model.data_generator(subkey, thetas)
            dists_perm, ys_index, zs_index, _ = optimal_index_distance(
                model=model, zs=zs, y_obs=y_obs, epsilon=0, verbose=0, L=L0
            )
            epsilon = np.quantile(dists_perm, alpha_epsilon)

            key, subkey = random.split(key)
            model.reset_weights_distance()
            out_um = perm_abc_smc_um(
                key=subkey, model=model, n_particles=N_particles, y_obs=y_obs,
                kernel=kernel, L_0=L0, epsilon=epsilon, alpha_L=alpha_L,
                update_weights_distance=False, verbose=1, Final_iteration=0
            )

            if out_um is not None:
                um_results['epsilons'].append(out_um["Eps_values"][-1])
                um_results['n_sim'].append(np.sum(out_um["N_sim"]))
                um_results['unique'].append(out_um["unique_part"][-1])
                um_results['time'].append(out_um["time_final"])
                um_results['full_results'].append(out_um)
        except Exception as e:
            print(f"    Failed: {e}")
            um_results['full_results'].append(None)

    for key_name in ['M0', 'epsilons', 'time', 'n_sim', 'unique']:
        os_results[key_name] = np.asarray(os_results[key_name], dtype=float)
    for key_name in ['epsilons', 'time', 'n_sim', 'unique']:
        um_results[key_name] = np.array(um_results[key_name])

    results['over_sampling'] = {'M0s': M0s, 'results': os_results}
    results['under_matching'] = {'L0s': L0s, 'results': um_results}

    return key, results


# =============================================================================
# Result processing
# =============================================================================

def _compute_row_diagnostics(model, y_obs, thetas_i, weights_i, perm_i, sigma2_edges):
    """Compute all diagnostic metrics for one population snapshot."""
    import time as _time
    result = {}
    for name, fn, kw in [
        ("kl_sigma2", empirical_kl_sigma2_abc_vs_true,
         dict(weights=weights_i, edges=sigma2_edges)),
        ("kl_sigma2_true_vs_abc", empirical_kl_sigma2_true_vs_abc,
         dict(weights=weights_i, edges=sigma2_edges)),
        ("score_joint", expected_neg_log_joint_true,
         dict(weights=weights_i, perm=perm_i)),
        ("kl_mu_avg", empirical_kl_mu_avg,
         dict(weights=weights_i, perm=perm_i)),
        ("w2_sigma2", empirical_w2_sigma2,
         dict(weights=weights_i)),
        ("w2_mu_avg", empirical_w2_mu_avg,
         dict(weights=weights_i, perm=perm_i)),
        ("sw2_joint", sliced_w2_joint,
         dict(weights=weights_i, perm=perm_i)),
    ]:
        t0 = _time.time()
        result[name] = fn(model, y_obs, thetas_i, **kw)
        dt = _time.time() - t0
        if dt > 1.0:
            print(f"    {name}: {result[name]:.4f} ({dt:.1f}s)", flush=True)
    return result


def _append_smc_rows(target_rows, out, display_name, model, y_obs, K, N_particles,
                     n_sample=1000, perm_key="Zs_index", sigma2_edges=None,
                     last_only=False, epsilon_override=None, extra_fields=None):
    """
    Append per-population rows for SMC-like outputs with diagnostics.

    If ``last_only=True``, only the final population is appended (used for OS).
    """
    if out is None:
        return
    needed = ["N_sim", "Eps_values", "Time", "Thetas", "Weights", "unique_part"]
    if any(k not in out for k in needed):
        return

    n_sim_arr = np.asarray(out["N_sim"], dtype=float)
    eps_arr = np.asarray(out["Eps_values"], dtype=float)
    time_arr = np.asarray(out["Time"], dtype=float)
    unique_arr = np.asarray(out["unique_part"], dtype=float)

    if n_sim_arr.size < 2 or eps_arr.size < 2 or time_arr.size < 2 or unique_arr.size < 2:
        return

    n_sim_cum = np.cumsum(n_sim_arr[1:])
    epsilons = eps_arr[1:]
    time_cum = np.cumsum(time_arr[1:])
    unique = unique_arr[1:]

    thetas_list = out["Thetas"]
    weights_list = out["Weights"]
    perm_list = out.get(perm_key, None)

    n_steps = min(len(n_sim_cum), len(epsilons), len(time_cum), len(unique),
                  max(0, len(thetas_list) - 1))
    if n_steps < 1:
        return

    step_range = [n_steps - 1] if last_only else range(n_steps)

    for i in step_range:
        unique_count = max(float(unique[i]), 1e-12) * max(float(N_particles), 1.0)
        # n_sim_cum counts compartment-sims (draws * K), divide by K to get draws
        n_sim_norm = n_sim_cum[i] / (K * unique_count) * n_sample
        # time_cum is wall-clock seconds, no K conversion needed
        time_norm = time_cum[i] / unique_count * n_sample

        thetas_i = _safe_array_at(thetas_list, i + 1)
        weights_i = _safe_array_at(weights_list, i + 1)
        perm_i = _safe_array_at(perm_list, i + 1)

        if thetas_i is None:
            continue

        # Skip diagnostics for UM intermediate steps where L < K
        loc_i = thetas_i.loc if hasattr(thetas_i, 'loc') else thetas_i
        n_comp = loc_i.shape[1] if hasattr(loc_i, 'shape') and loc_i.ndim >= 2 else K
        if n_comp < K:
            print(f"  Skipping diagnostics {display_name} step {i+1}/{n_steps} (L={n_comp} < K={K})", flush=True)
            continue

        print(f"  Diagnostics {display_name} step {i+1}/{n_steps} (eps={epsilons[i]:.4f})", flush=True)
        diag = _compute_row_diagnostics(model, y_obs, thetas_i, weights_i, perm_i, sigma2_edges)

        eps_out = float(epsilon_override) if epsilon_override is not None else float(epsilons[i])
        row = {
            "method": display_name,
            "n_sim": float(n_sim_norm),
            "time": float(time_norm),
            "n_sim_raw": float(n_sim_cum[i]),
            "time_raw": float(time_cum[i]),
            "epsilon": eps_out,
            **diag,
        }
        if extra_fields:
            row.update(extra_fields)
        target_rows.append(row)


def process_results(vanilla_results, smc_results, osum_results, model, y_obs,
                    K, N_particles, include_osum=True):
    """Process all results for comparison."""
    N_sample = 1000
    sigma2_edges = build_sigma2_reference_bins(model, y_obs, nbins=80)

    # Check if vanilla results are available
    has_vanilla = (vanilla_results is not None
                   and vanilla_results.get('vanilla_abc') is not None
                   and vanilla_results.get('vanilla_perm') is not None)

    processed_results = {
        'method': [], 'n_sim': [], 'time': [], 'epsilon': [],
        'kl_sigma2': [], 'score_joint': [], 'kl_sigma2_true_vs_abc': [],
        'kl_mu_avg': [], 'w2_sigma2': [], 'w2_mu_avg': [], 'sw2_joint': [],
        'n_sim_raw': [], 'time_raw': [],
    }

    if has_vanilla:
        dists = vanilla_results['vanilla_abc']['dists']
        dists_perm = vanilla_results['vanilla_perm']['dists']
        time_van = vanilla_results['vanilla_abc']['time']
        time_perm_van = vanilla_results['vanilla_perm']['time']
        N_points = len(dists)

        time_by_sim_van = time_van / N_points
        time_by_sim_perm_van = time_perm_van / N_points
        alphas = np.logspace(0, -3, 10)
        n_sim_van = 1 / alphas * N_sample
        n_sim_perm_van = 1 / alphas * N_sample

        processed_results['method'] = ['ABC-Vanilla'] * len(alphas) + ['permABC-Vanilla'] * len(alphas)
        processed_results['n_sim'] = list(np.concatenate([n_sim_van, n_sim_perm_van]))
        processed_results['time'] = list(np.concatenate([n_sim_van * time_by_sim_van, n_sim_perm_van * time_by_sim_perm_van]))
        processed_results['epsilon'] = list(np.concatenate([np.quantile(dists, alphas), np.quantile(dists_perm, alphas)]))

        # Diagnostics for vanilla rejection ABC
        thetas_van = vanilla_results['vanilla_abc']['thetas']
        thetas_perm = vanilla_results['vanilla_perm']['thetas']
        perm_van = vanilla_results['vanilla_perm'].get('zs_index', None)

        n_diag = N_particles
        rng_diag = np.random.default_rng(42)

        for j, eps_th in enumerate(np.quantile(dists, alphas)):
            print(f"  Diagnostics ABC-Vanilla {j+1}/{len(alphas)} (eps={eps_th:.4f})", flush=True)
            mask = dists <= eps_th
            thetas_acc = thetas_van[mask]
            if len(thetas_acc) > n_diag:
                idx = rng_diag.choice(len(thetas_acc), n_diag, replace=False)
                thetas_acc = thetas_acc[idx]
            diag = _compute_row_diagnostics(model, y_obs, thetas_acc, None, None, sigma2_edges)
            processed_results['kl_sigma2'].append(diag['kl_sigma2'])
            processed_results['score_joint'].append(diag['score_joint'])
            processed_results['kl_sigma2_true_vs_abc'].append(diag['kl_sigma2_true_vs_abc'])
            processed_results['kl_mu_avg'].append(diag['kl_mu_avg'])
            processed_results['w2_sigma2'].append(diag['w2_sigma2'])
            processed_results['w2_mu_avg'].append(diag['w2_mu_avg'])
            processed_results['sw2_joint'].append(diag['sw2_joint'])

        for j, eps_th in enumerate(np.quantile(dists_perm, alphas)):
            print(f"  Diagnostics permABC-Vanilla {j+1}/{len(alphas)} (eps={eps_th:.4f})", flush=True)
            mask = dists_perm <= eps_th
            thetas_acc = thetas_perm[mask]
            perm_mask = perm_van[mask] if perm_van is not None else None
            if len(thetas_acc) > n_diag:
                idx = rng_diag.choice(len(thetas_acc), n_diag, replace=False)
                thetas_acc = thetas_acc[idx]
                perm_mask = perm_mask[idx] if perm_mask is not None else None
            diag = _compute_row_diagnostics(model, y_obs, thetas_acc, None, perm_mask, sigma2_edges)
            processed_results['kl_sigma2'].append(diag['kl_sigma2'])
            processed_results['score_joint'].append(diag['score_joint'])
            processed_results['kl_sigma2_true_vs_abc'].append(diag['kl_sigma2_true_vs_abc'])
            processed_results['kl_mu_avg'].append(diag['kl_mu_avg'])
            processed_results['w2_sigma2'].append(diag['w2_sigma2'])
            processed_results['w2_mu_avg'].append(diag['w2_mu_avg'])
            processed_results['sw2_joint'].append(diag['sw2_joint'])

        # n_sim_raw in compartment-simulations (consistent with SMC where N_sim = draws * K)
        n_sim_raw_van = n_sim_van * K
        time_raw_van_arr = n_sim_van * time_by_sim_van
        n_sim_raw_perm = n_sim_perm_van * K
        time_raw_perm_arr = n_sim_perm_van * time_by_sim_perm_van
        processed_results['n_sim_raw'] = list(n_sim_raw_van) + list(n_sim_raw_perm)
        processed_results['time_raw'] = list(time_raw_van_arr) + list(time_raw_perm_arr)

    # SMC results
    smc_data = []
    for name_key, display_name in [('abc_smc', 'ABC-SMC'), ('abc_pmc', 'ABC-PMC'), ('perm_abc_smc', 'permABC-SMC')]:
        if name_key in smc_results and smc_results[name_key] is not None:
            _append_smc_rows(
                target_rows=smc_data, out=smc_results[name_key],
                display_name=display_name, model=model, y_obs=y_obs,
                K=K, N_particles=N_particles, n_sample=N_sample,
                sigma2_edges=sigma2_edges,
            )

    # OSUM results
    osum_data = []
    if include_osum and osum_results:
        os_results = osum_results['over_sampling']['results']
        if len(os_results.get('full_results', [])) > 0:
            m0_arr = os_results.get('M0')
            for idx, out_full in enumerate(os_results['full_results']):
                if out_full is None:
                    continue
                cal_eps = os_results['epsilons'][idx]
                if not np.isfinite(cal_eps):
                    continue
                extra = {}
                if m0_arr is not None and idx < len(m0_arr) and np.isfinite(m0_arr[idx]):
                    extra['M0'] = float(m0_arr[idx])
                _append_smc_rows(
                    target_rows=osum_data, out=out_full,
                    display_name='permABC-SMC-OS', model=model, y_obs=y_obs,
                    K=K, N_particles=N_particles, n_sample=N_sample,
                    sigma2_edges=sigma2_edges, last_only=True,
                    epsilon_override=float(cal_eps),
                    extra_fields=extra if extra else None,
                )

        um_results = osum_results['under_matching']['results']
        if len(um_results['unique']) > 0:
            for out_full in um_results['full_results']:
                _append_smc_rows(
                    target_rows=osum_data, out=out_full,
                    display_name='permABC-SMC-UM', model=model, y_obs=y_obs,
                    K=K, N_particles=N_particles, n_sample=N_sample,
                    sigma2_edges=sigma2_edges, last_only=True,
                )

    # Combine all results
    all_results = []
    for i in range(len(processed_results['method'])):
        all_results.append({
            'method': processed_results['method'][i],
            'n_sim': processed_results['n_sim'][i],
            'time': processed_results['time'][i],
            'n_sim_raw': processed_results['n_sim_raw'][i],
            'time_raw': processed_results['time_raw'][i],
            'epsilon': processed_results['epsilon'][i],
            'kl_sigma2': processed_results['kl_sigma2'][i],
            'score_joint': processed_results['score_joint'][i],
            'kl_sigma2_true_vs_abc': processed_results['kl_sigma2_true_vs_abc'][i],
            'kl_mu_avg': processed_results['kl_mu_avg'][i],
            'w2_sigma2': processed_results['w2_sigma2'][i],
            'w2_mu_avg': processed_results['w2_mu_avg'][i],
            'sw2_joint': processed_results['sw2_joint'][i],
        })

    all_results.extend(smc_data)
    all_results.extend(osum_data)

    if pd is None:
        return all_results
    return pd.DataFrame(all_results)


# =============================================================================
# Plotting
# =============================================================================

def get_plot_style():
    """Get consistent plot styling for all performance plots."""
    colors = {
        'ABC-Vanilla': '#d62728',       # Red
        'permABC-Vanilla': '#2ca02c',   # Green
        'ABC-SMC': '#ff7f0e',           # Dark orange
        'ABC-PMC': '#ffbb78',           # Light orange
        'permABC-SMC': '#1f77b4',       # Blue
        'permABC-SMC-OS': '#e377c2',    # Pink
        'permABC-SMC-UM': '#9467bd'     # Purple
    }
    markers = {
        'ABC-Vanilla': 's',
        'permABC-Vanilla': 's',
        'ABC-SMC': 'o',
        'ABC-PMC': 'o',
        'permABC-SMC': 'o',
        'permABC-SMC-OS': '^',
        'permABC-SMC-UM': '^'
    }
    linestyles = {
        'ABC-Vanilla': '-',
        'permABC-Vanilla': '-',
        'ABC-SMC': '--',
        'ABC-PMC': '--',
        'permABC-SMC': '--',
        'permABC-SMC-OS': '--',
        'permABC-SMC-UM': '--'
    }
    return colors, markers, linestyles


def _create_performance_plot(df, K, K_outliers, include_osum, x_key, x_label, ax=None):
    """
    Create a performance plot with epsilon on the y-axis.

    ``x_key``: column name for the x-axis (``"n_sim"`` or ``"time"``).
    ``x_label``: axis label string.
    """
    colors, markers, linestyles = get_plot_style()

    smc_factors = {
        "permABC-SMC": 0.0001,
        "ABC-SMC": 0.0001 if x_key == "n_sim" else 0.000001,
        "ABC-PMC": 0.00001,
    }
    len_indices = {"permABC-SMC": 10, "ABC-SMC": 10, "ABC-PMC": 10}

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.get_figure()

    for method in _df_unique_methods(df):
        method_data = _df_sort_by(_df_filter_method(df, method), 'epsilon')

        if 'SMC' in method or 'PMC' in method:
            if pd is not None and hasattr(method_data, "columns"):
                method_data = method_data[method_data['n_sim'] <= 1e6]
                x_vals = method_data[x_key].to_numpy()
                eps_vals = method_data['epsilon'].to_numpy()
            else:
                method_data = [r for r in method_data if r['n_sim'] <= 1e6]
                x_vals = _df_col(method_data, x_key)
                eps_vals = _df_col(method_data, 'epsilon')

            factor = smc_factors.get(method, 2.0)
            length_indices = len_indices.get(method, 10)
            indices = select_indices(list(range(len(x_vals))), length_indices, factor=factor)

            ax.plot(x_vals[indices], eps_vals[indices],
                    label=method, color=colors.get(method, 'gray'),
                    marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                    markersize=8, linewidth=2)
        else:
            x_vals = _df_col(method_data, x_key)
            eps_vals = _df_col(method_data, 'epsilon')
            ax.plot(x_vals, eps_vals,
                    label=method, color=colors.get(method, 'gray'),
                    marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                    markersize=8, linewidth=2)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("$\\varepsilon$", fontsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()
    return fig


def create_nsim_plot(df, K, K_outliers, include_osum, ax=None):
    """Create simulation efficiency plot."""
    return _create_performance_plot(
        df, K, K_outliers, include_osum,
        x_key="n_sim",
        x_label="Number of simulations per 1000 unique particles",
        ax=ax,
    )


def create_time_plot(df, K, K_outliers, include_osum, ax=None):
    """Create time efficiency plot."""
    return _create_performance_plot(
        df, K, K_outliers, include_osum,
        x_key="time",
        x_label="Time per 1000 unique particles (seconds)",
        ax=ax,
    )


def create_combined_plot(df, K, K_outliers, include_osum):
    """Create combined simulation and time efficiency plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    create_nsim_plot(df, K, K_outliers, include_osum, ax=ax1)
    create_time_plot(df, K, K_outliers, include_osum, ax=ax2)
    plt.tight_layout()
    return fig


# =============================================================================
# Save / analyze / rerun
# =============================================================================

def save_results(df, vanilla_results, smc_results, osum_results, model, y_obs, true_theta,
                 K, K_outliers, include_osum, seed, N_particles, output_dir, plot_type='both'):
    """Save results to both pickle and CSV formats."""
    results_dir = os.path.join(output_dir, "experiments", "results", "performance_comparison")
    fig_num_str = "fig4" if not include_osum else "fig6"
    figures_dir = os.path.join(output_dir, "figures", fig_num_str)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if pd is not None and hasattr(df, "columns"):
        df['K'] = K
        df['K_outliers'] = K_outliers
        df['include_osum'] = include_osum
        df['seed'] = seed
    else:
        for r in df:
            r['K'] = K
            r['K_outliers'] = K_outliers
            r['include_osum'] = include_osum
            r['seed'] = seed

    full_data = {
        'seed': seed,
        'K': K,
        'K_outliers': K_outliers,
        'include_osum': include_osum,
        'N_particles': N_particles,
        'plot_type': plot_type,
        'summary_df': df,
        'results': {
            'vanilla': vanilla_results,
            'smc': smc_results,
            'osum': osum_results
        },
        'experiment_setup': {
            'model': model,
            'y_obs': y_obs,
            'true_theta': true_theta
        },
        'metadata': {
            'created_with': 'run_performance_comparison.py',
            'analysis_type': 'performance_comparison',
            'methods_included': _df_unique_methods(df),
            'timestamp': datetime.datetime.now().isoformat()
        }
    }

    base_filename = f"performance_K_{K}_outliers_{K_outliers}_osum_{include_osum}_seed_{seed}"
    pkl_path = os.path.join(results_dir, f"{base_filename}.pkl")
    csv_path = os.path.join(results_dir, f"{base_filename}.csv")

    # CSV first (small; survives disk-full during pickle)
    if pd is not None and hasattr(df, "to_csv"):
        df.to_csv(csv_path, index=False)
    else:
        fieldnames = sorted(list(df[0].keys())) if len(df) else []
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in df:
                w.writerow(r)
    print(f"Summary saved to CSV: {csv_path}")

    # Comprehensive pickle (skip if file already exists to avoid clobbering)
    if not os.path.exists(pkl_path):
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(full_data, f)
            print(f"Full results saved to pickle: {pkl_path}")
        except OSError as e:
            if e.errno != errno.ENOSPC:
                raise
            print(
                "Warning: no space left on device for full pickle; "
                "writing a lightweight pickle (summary_df + metadata only).",
                file=sys.stderr,
            )
            slim_data = {
                "seed": seed, "K": K, "K_outliers": K_outliers,
                "include_osum": include_osum, "N_particles": N_particles,
                "plot_type": plot_type, "summary_df": df,
                "results": None, "experiment_setup": None,
                "metadata": full_data["metadata"],
                "pickle_note": "lightweight: raw outputs omitted (ENOSPC during full pickle).",
            }
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(slim_data, f)
                print(f"Lightweight pickle saved to: {pkl_path}")
            except OSError as e2:
                print(
                    f"Warning: could not write any pickle ({e2}); CSV above is still valid.",
                    file=sys.stderr,
                )

    # Create and save plots
    fig_nsim = create_nsim_plot(df, K, K_outliers, include_osum)
    fig_nsim.savefig(os.path.join(figures_dir, f"{fig_num_str}_nsim_seed_{seed}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_nsim)

    fig_time = create_time_plot(df, K, K_outliers, include_osum)
    fig_time.savefig(os.path.join(figures_dir, f"{fig_num_str}_time_seed_{seed}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_time)

    print(f"Plots saved in: {figures_dir}")
    return pkl_path, csv_path


def analyze_performance_metrics(df):
    """Analyze and print performance metrics."""
    print("\nPerformance Analysis:")
    print("=" * 50)

    for method in _df_unique_methods(df):
        method_data = _df_filter_method(df, method)
        print(f"\n{method}:")
        if pd is not None and hasattr(method_data, "columns"):
            print(f"  Data points: {len(method_data)}")
            print(f"  Epsilon range: [{method_data['epsilon'].min():.2e}, {method_data['epsilon'].max():.2e}]")
            print(f"  Simulation efficiency range: [{method_data['n_sim'].min():.0f}, {method_data['n_sim'].max():.0f}]")
            print(f"  Time efficiency range: [{method_data['time'].min():.2f}, {method_data['time'].max():.2f}]s")
        else:
            eps = _df_col(method_data, 'epsilon')
            ns = _df_col(method_data, 'n_sim')
            tm = _df_col(method_data, 'time')
            print(f"  Data points: {len(method_data)}")
            print(f"  Epsilon range: [{np.min(eps):.2e}, {np.max(eps):.2e}]")
            print(f"  Simulation efficiency range: [{np.min(ns):.0f}, {np.max(ns):.0f}]")
            print(f"  Time efficiency range: [{np.min(tm):.2f}, {np.max(tm):.2f}]s")

    print(f"\nComparison at median epsilon levels:")
    for method in _df_unique_methods(df):
        method_data = _df_sort_by(_df_filter_method(df, method), 'epsilon')
        if len(method_data) > 0:
            median_idx = len(method_data) // 2
            if pd is not None and hasattr(method_data, "iloc"):
                median_row = method_data.iloc[median_idx]
            else:
                median_row = method_data[median_idx]
            eps = median_row['epsilon']
            ns = median_row['n_sim']
            tm = median_row['time']
            print(f"  {method}: eps={eps:.2e}, sims={ns:.0f}, time={tm:.2f}s")


def rerun_from_file(file_path, plot_type='both', osum_flag_for_plot=False, output_dir_override=None):
    """Recreate plots from existing pickle file, respecting current run flags.

    If the summary_df is missing diagnostics (score_joint, kl_sigma2) — e.g.
    because the pickle was generated before those metrics were added — the raw
    results stored in the pickle are re-processed to compute them.
    """
    print(f"Loading results from: {file_path}")

    if not file_path.endswith('.pkl'):
        print("Error: Rerun requires a .pkl file to ensure all data is available.")
        return

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    df = data['summary_df']
    K = data['K']
    K_outliers = data['K_outliers']
    seed = data['seed']
    N_particles = data.get('N_particles', 1000)

    # --- Recompute diagnostics if missing from old-format pickle ----------
    recs = df.to_dict(orient="records") if (pd is not None and hasattr(df, "columns")) else df
    has_score = any("score_joint" in r for r in recs)
    has_new_metrics = any("sw2_joint" in r for r in recs)
    _reprocessed = False
    if not has_score or not has_new_metrics:
        raw = data.get("results")
        exp = data.get("experiment_setup", {})
        model = exp.get("model")
        y_obs = exp.get("y_obs")
        if raw and model is not None and y_obs is not None:
            print("Pickle missing new diagnostics. Reprocessing...")
            include_osum = data.get("include_osum", osum_flag_for_plot)
            df = process_results(
                raw.get("vanilla", {}),
                raw.get("smc", {}),
                raw.get("osum", {}),
                model, y_obs, K, N_particles,
                include_osum=include_osum,
            )
            # Add metadata columns
            if pd is not None and hasattr(df, "columns"):
                df['K'] = K
                df['K_outliers'] = K_outliers
                df['include_osum'] = include_osum
                df['seed'] = seed
            else:
                for r in df:
                    r['K'] = K
                    r['K_outliers'] = K_outliers
                    r['include_osum'] = include_osum
                    r['seed'] = seed
            # Update the in-memory data so the rest of the function uses new df
            data['summary_df'] = df
            _reprocessed = True
            print(f"  Reprocessed: {len(df) if hasattr(df, '__len__') else '?'} records with diagnostics")
        else:
            print("WARNING: Pickle lacks diagnostics and raw data unavailable; "
                  "plots will only show methods with diagnostics.")

    if not osum_flag_for_plot:
        print("Filtering out OSUM methods for Figure 4 generation...")
        methods_to_exclude = ['permABC-SMC-OS', 'permABC-SMC-UM']
        if pd is None or not hasattr(df, "columns"):
            df_filtered = [r for r in df if r.get("method", "") not in methods_to_exclude]
        else:
            df_filtered = df[~df['method'].isin(methods_to_exclude)].copy()
    else:
        df_filtered = df

    fig_num_str = "fig4" if not osum_flag_for_plot else "fig6"
    print(f"Recreating {fig_num_str} for K={K}, K_outliers={K_outliers}, seed={seed}")

    if output_dir_override:
        figures_dir = os.path.join(output_dir_override, "figures", fig_num_str)
    else:
        figures_dir = os.path.join(os.path.dirname(file_path), "..", "..", "figures", fig_num_str)

    os.makedirs(figures_dir, exist_ok=True)

    fig_nsim = create_nsim_plot(df_filtered, K, K_outliers, osum_flag_for_plot)
    fig_nsim.savefig(os.path.join(figures_dir, f"{fig_num_str}_nsim_seed_{seed}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_nsim)

    fig_time = create_time_plot(df_filtered, K, K_outliers, osum_flag_for_plot)
    fig_time.savefig(os.path.join(figures_dir, f"{fig_num_str}_time_seed_{seed}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_time)

    # Save reprocessed pickle if diagnostics were recomputed
    if _reprocessed:
        try:
            data['summary_df'] = df
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Updated pickle with diagnostics: {file_path}")
        except Exception as e:
            print(f"WARNING: could not update pickle: {e}")

    print(f"Plots recreated in: {figures_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run performance comparison between ABC methods"
    )

    parser.add_argument('--K', type=int, default=20,
                        help='Number of components (default: 20)')
    parser.add_argument('--K_outliers', type=int, default=4,
                        help='Number of outlier components (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--N_points', type=int, default=1000000,
                        help='Number of points for vanilla methods (default: 1,000,000)')
    parser.add_argument('--N_particles', type=int, default=1000,
                        help='Number of particles for SMC methods (default: 1,000)')
    parser.add_argument('--osum', dest='include_osum', action='store_true',
                        help='Include over-sampling and under-matching methods')
    parser.add_argument('--no-osum', dest='include_osum', action='store_false',
                        help='Exclude over-sampling and under-matching methods')
    parser.add_argument('--methods', nargs='+', default=None,
                        choices=['vanilla', 'smc', 'pmc', 'osum'],
                        help='Run only selected method groups (vanilla, smc, pmc, osum). '
                             'Appends to existing CSV if present. '
                             'Example: --methods osum pmc  to run only OS/UM + PMC.')
    parser.add_argument('--plot', type=str, choices=['nsim', 'time', 'both'], default='both',
                        help='Type of plots to generate: nsim (simulations), time, or both (default: both)')
    parser.add_argument('--output-dir', type=str, default="experiments",
                        help='Output directory (default: experiments)')
    parser.add_argument('--rerun', type=str, default=None,
                        help='Path to pickle or CSV file for rerunning analysis')
    parser.add_argument('--stop-rate', type=float, default=0.0,
                        help='Stopping acceptance rate for SMC. When > 0, permABC-SMC '
                             'runs first with N_sim_max=inf to set the budget for all methods.')

    parser.set_defaults(include_osum=True)
    return parser.parse_args()


def _load_existing_csv(K, K_outliers, seed, output_dir):
    """Load existing CSV results if present, return DataFrame or list-of-dicts."""
    for osum_val in [True, False]:
        csv_path = os.path.join(
            output_dir, "experiments", "results", "performance_comparison",
            f"performance_K_{K}_outliers_{K_outliers}_osum_{osum_val}_seed_{seed}.csv"
        )
        if os.path.exists(csv_path):
            print(f"Loading existing results from: {csv_path}")
            if pd is not None:
                return pd.read_csv(csv_path), csv_path
            else:
                import csv as csv_mod
                with open(csv_path, "r") as f:
                    reader = csv_mod.DictReader(f)
                    return list(reader), csv_path
    return None, None


def _merge_results(existing_df, new_df):
    """Merge new results into existing, replacing methods that overlap."""
    if pd is not None and hasattr(new_df, "columns"):
        new_methods = set(new_df['method'].unique())
        # Remove old rows for methods we're replacing
        keep = existing_df[~existing_df['method'].isin(new_methods)]
        merged = pd.concat([keep, new_df], ignore_index=True)
        return merged
    else:
        new_methods = set(r['method'] for r in new_df)
        keep = [r for r in existing_df if r.get('method') not in new_methods]
        return keep + list(new_df)


def main():
    """Main execution function."""
    args = parse_arguments()

    if args.rerun:
        rerun_from_file(args.rerun, args.plot, args.include_osum, args.output_dir)
        return

    # Determine which method groups to run
    if args.methods is not None:
        run_vanilla = 'vanilla' in args.methods
        run_smc = 'smc' in args.methods
        run_pmc = 'pmc' in args.methods
        run_osum = 'osum' in args.methods
        include_osum = run_osum
    else:
        run_vanilla = True
        run_smc = True
        run_pmc = True
        run_osum = args.include_osum
        include_osum = args.include_osum

    print("Performance comparison between ABC methods")
    print(f"Parameters: K={args.K}, K_outliers={args.K_outliers}, seed={args.seed}")
    print(f"N_points={args.N_points:,}, N_particles={args.N_particles:,}")
    method_groups = [g for g, run in [('vanilla', run_vanilla), ('smc', run_smc), ('pmc', run_pmc), ('osum', run_osum)] if run]
    print(f"Method groups: {method_groups}")
    print(f"Plot type: {args.plot}")

    model, y_obs, true_theta, key, N_points, N_particles, stopping_rate, K = setup_experiment(
        K=args.K, K_outliers=args.K_outliers, seed=args.seed,
        N_points=args.N_points, N_particles=args.N_particles
    )

    # Override stopping_rate from CLI if provided
    if args.stop_rate > 0:
        stopping_rate = args.stop_rate

    # When stop_rate > 0: run permABC-SMC first with N_sim_max=inf to determine budget
    # This run is reused as the permABC-SMC result (no double run).
    out_perm_pilot = None
    if stopping_rate > 0 and run_smc:
        print(f"\n*** Budget discovery: running permABC-SMC with stop_rate={stopping_rate}, N_sim_max=inf ***")
        kernel = KernelTruncatedRW
        key, subkey = random.split(key)
        model.reset_weights_distance()
        out_perm_pilot = perm_abc_smc(
            key=subkey, model=model, n_particles=N_particles, epsilon_target=0, y_obs=y_obs,
            kernel=kernel, verbose=1, Final_iteration=0, update_weights_distance=False,
            stopping_accept_rate=stopping_rate, N_sim_max=np.inf
        )
        budget_comp_sims = int(np.sum(out_perm_pilot["N_sim"]))
        N_points = budget_comp_sims // K
        print(f"*** permABC-SMC budget: {budget_comp_sims:,} comp-sims → N_points={N_points:,} ***\n")

    vanilla_results = None
    smc_results = {}
    osum_results = None

    if run_vanilla:
        key, vanilla_results = run_vanilla_methods(key, model, N_points, y_obs)

    if run_smc:
        key, smc_results = run_smc_methods(
            key, model, N_particles, y_obs, stopping_rate, N_points, K,
            out_perm_pilot=out_perm_pilot,
        )

    if run_pmc:
        key, pmc_out = run_pmc_method(key, model, N_particles, y_obs, stopping_rate, N_points, K)
        if pmc_out is not None:
            smc_results['abc_pmc'] = pmc_out

    if run_osum:
        key, osum_results = run_osum_methods(key, model, N_particles, y_obs, K)

    # Build DataFrame from newly computed results
    new_df = process_results(
        vanilla_results or {'vanilla_abc': None, 'vanilla_perm': None},
        smc_results,
        osum_results,
        model, y_obs, K, N_particles,
        include_osum=run_osum,
    )

    # If running a subset, merge with existing CSV
    if args.methods is not None:
        existing_df, existing_csv_path = _load_existing_csv(
            args.K, args.K_outliers, args.seed, args.output_dir
        )
        if existing_df is not None:
            new_df = _merge_results(existing_df, new_df)
            print(f"Merged with existing results ({existing_csv_path})")
        include_osum = True  # merged df may have osum methods

    analyze_performance_metrics(new_df)

    pkl_path, csv_path = save_results(
        new_df, vanilla_results, smc_results, osum_results, model, y_obs, true_theta,
        args.K, args.K_outliers, include_osum, args.seed, N_particles, args.output_dir, args.plot
    )

    print("Performance comparison complete!")
    print(f"Full results: {pkl_path}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
