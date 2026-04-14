#!/usr/bin/env python3
"""
Final solver benchmark for permABC paper — validates claims about assignment methods.

Claims tested:
  1. Hilbert degrades with K (micro benchmark)
  2. Smart Swap (identity→swap→lsa) is efficient in SMC (macro + cascade stats)
  3. Warm-start LSA doesn't help (micro benchmark)
  4. Sinkhorn is not competitive (both levels)

Outputs:
  - micro_results.csv: per-batch assignment quality/speed
  - macro_results.pkl: full SMC results with per-iteration metrics + cascade stats
  - macro_results.csv: summary table

Usage:
  python run_solver_final_benchmark.py
  python run_solver_final_benchmark.py --level micro
  python run_solver_final_benchmark.py --level macro --K 5 10
  python run_solver_final_benchmark.py --level both --seed 42
"""

import warnings; warnings.filterwarnings('ignore')
import sys, os, time, gc, argparse, pickle
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "scripts"))

import numpy as np
import pandas as pd
from jax import random

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.utils.functions import Theta
from permabc.assignment.dispatch import (
    _build_cost_and_global, last_smart_stats,
)
from permabc.assignment.solvers.lsa import solve_lsa, solve_lsa_custom, _HAS_CUSTOM_LSA
from permabc.assignment.solvers.sinkhorn import sinkhorn_assignment
from permabc.assignment.solvers.swap import do_swap, _HAS_NUMBA
from permabc.assignment.solvers.hilbert import _HAS_CGAL
from permabc.assignment.dispatch import do_hilbert
from permabc.assignment.distances import compute_total_distance

from diagnostics import (
    build_sigma2_reference_bins,
    empirical_kl_sigma2_abc_vs_true,
    expected_neg_log_joint_true,
)


# =====================================================================
# Micro benchmark: per-batch assignment quality and speed
# =====================================================================

MICRO_METHODS = [
    "LSA (scipy)",
    "LSA custom cold",
    "LSA custom warm",
    "Hilbert",
    "Hilbert+Swap",
    "Sinkhorn",
    "Sinkhorn+Swap",
    "Swap only",
]


def run_micro_single(method, local_mats, global_d, model, zs, y_obs, K,
                     ys_lsa, zs_lsa):
    """Run one micro method, return (elapsed_ms, distance_ratio, hamming)."""
    t0 = time.perf_counter()

    if method == "LSA (scipy)":
        ys, zs_idx = solve_lsa(local_mats, parallel=True)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif method == "LSA custom cold":
        if not _HAS_CUSTOM_LSA:
            return np.nan, np.nan, np.nan
        ys, zs_idx = solve_lsa_custom(local_mats, init_col4row=None, parallel=True)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif method == "LSA custom warm":
        if not _HAS_CUSTOM_LSA:
            return np.nan, np.nan, np.nan
        ys, zs_idx = solve_lsa_custom(local_mats, init_col4row=zs_lsa, parallel=True)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif method == "Hilbert":
        y_ref = y_obs[0, :K]
        weights = np.asarray(model.weights_distance[:K])
        zs_slice = zs[:, :K]
        h_dist, ys, zs_idx = do_hilbert(zs_slice, y_ref, weights)
        dists = np.sqrt(h_dist ** 2 + global_d)

    elif method == "Hilbert+Swap":
        y_ref = y_obs[0, :K]
        weights = np.asarray(model.weights_distance[:K])
        zs_slice = zs[:, :K]
        _, ys, zs_idx = do_hilbert(zs_slice, y_ref, weights)
        ys, zs_idx = do_swap(local_mats, ys, zs_idx)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif method == "Sinkhorn":
        ys, zs_idx = sinkhorn_assignment(local_mats, backend="auto")
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif method == "Sinkhorn+Swap":
        ys, zs_idx = sinkhorn_assignment(local_mats, backend="auto")
        ys, zs_idx = do_swap(local_mats, ys, zs_idx)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif method == "Swap only":
        N = local_mats.shape[0]
        ys = np.tile(np.arange(K, dtype=np.int32), (N, 1))
        zs_idx = np.tile(np.arange(K, dtype=np.int32), (N, 1))
        ys, zs_idx = do_swap(local_mats, ys, zs_idx)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)
    else:
        raise ValueError(method)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    d_lsa = compute_total_distance(zs_lsa, ys_lsa, local_mats, global_d)
    ratio = float(np.mean(dists / np.where(d_lsa > 0, d_lsa, 1e-30)))
    hamming = float(np.mean(zs_idx != zs_lsa))

    return elapsed_ms, ratio, hamming


def _warmup_jit(K=5, n_particles=20, n_obs=2):
    """Warmup JIT-compiled functions (Numba/JAX) with a tiny problem."""
    print("  Warming up JIT-compiled solvers...", flush=True)
    model = GaussianWithNoSummaryStats(K=K, n_obs=n_obs)
    key = random.PRNGKey(999)
    k1, k2, k3 = random.split(key, 3)
    theta = model.prior_generator(k1, n_particles)
    zs = np.asarray(model.data_generator(k2, theta))
    y_obs = np.asarray(model.data_generator(k3, model.prior_generator(random.split(key)[0], 1)))
    if y_obs.ndim == 2:
        y_obs = y_obs[None, ...]
    local_mats = np.asarray(model.distance_matrices_loc(zs, y_obs))
    global_d = np.asarray(model.distance_global(zs, y_obs))
    local_mats = np.where(np.isinf(local_mats), 1e12, local_mats)
    ys_lsa, zs_lsa = solve_lsa(local_mats, parallel=False)

    for method in MICRO_METHODS:
        try:
            run_micro_single(method, local_mats, global_d, model, zs,
                             y_obs, K, ys_lsa, zs_lsa)
        except Exception:
            pass
    print("  Warmup done.", flush=True)


def run_micro(K_values, n_obs_values, n_particles, n_repeats, seed, out_dir):
    """Run micro benchmark for all K and n_obs combinations."""
    _warmup_jit()

    rows = []
    key = random.PRNGKey(seed)
    total = len(K_values) * len(n_obs_values) * n_repeats * len(MICRO_METHODS)
    done = 0
    t_global = time.perf_counter()

    for K in K_values:
        for n_obs in n_obs_values:
            model = GaussianWithNoSummaryStats(K=K, n_obs=n_obs)

            for rep in range(n_repeats):
                key, k1, k2, k3 = random.split(key, 4)
                true_theta = model.prior_generator(k1, n_particles)
                zs = np.asarray(model.data_generator(k2, true_theta))
                y_obs_single = np.asarray(model.data_generator(
                    k3, model.prior_generator(random.split(key)[0], 1)))
                if y_obs_single.ndim == 2:
                    y_obs_single = y_obs_single[None, ...]

                local_mats = np.asarray(model.distance_matrices_loc(zs, y_obs_single))
                global_d = np.asarray(model.distance_global(zs, y_obs_single))
                local_mats = np.where(np.isinf(local_mats), 1e12, local_mats)

                # Reference LSA
                t0 = time.perf_counter()
                ys_lsa, zs_lsa = solve_lsa(local_mats, parallel=True)
                lsa_ms = (time.perf_counter() - t0) * 1000

                for method in MICRO_METHODS:
                    elapsed_ms, ratio, hamming = run_micro_single(
                        method, local_mats, global_d, model, zs,
                        y_obs_single, K, ys_lsa, zs_lsa,
                    )
                    if not np.isnan(elapsed_ms):
                        rows.append({
                            "K": K, "n_obs": n_obs, "seed": seed, "repeat": rep,
                            "method": method,
                            "time_ms": elapsed_ms,
                            "lsa_time_ms": lsa_ms,
                            "speedup": lsa_ms / max(elapsed_ms, 1e-6),
                            "distance_ratio": ratio,
                            "hamming": hamming,
                        })
                    done += 1

                elapsed_total = time.perf_counter() - t_global
                if done > 0:
                    eta = elapsed_total / done * (total - done)
                    print(f"  [{done:4d}/{total}] K={K:3d} n_obs={n_obs:3d} rep={rep}  "
                          f"elapsed={elapsed_total:.0f}s  ETA={eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "micro_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMicro: {len(rows)} rows saved to {csv_path}")

    # Summary by K and method (median over n_obs and repeats)
    summary = df.groupby(["K", "method"]).agg(
        time_ms_median=("time_ms", "median"),
        ratio_median=("distance_ratio", "median"),
        ratio_max=("distance_ratio", "max"),
        hamming_median=("hamming", "median"),
        speedup_median=("speedup", "median"),
    ).reset_index()
    summary_path = os.path.join(out_dir, "micro_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    print("\n" + summary.to_string(index=False))
    return df


# =====================================================================
# Macro benchmark: full SMC with cascade stats
# =====================================================================

MACRO_CONFIGS = [
    ("LSA",            ["identity", "lsa"]),
    ("Smart Swap",     ["identity", "swap", "lsa"]),
    ("Hilbert",        ["identity", "hilbert"]),
    ("Hilbert+Swap",   ["identity", "hilbert", "swap"]),
    ("Smart Hilbert",  ["identity", "hilbert", "lsa"]),
    ("Smart H+S",      ["identity", "hilbert", "swap", "lsa"]),
    ("Sinkhorn",       ["identity", "sinkhorn"]),
    ("Sinkhorn+Swap",  ["identity", "sinkhorn", "swap"]),
    ("Swap only",      ["identity", "swap"]),
]


def setup_experiment(K, seed, n_obs=10):
    """Create model, observed data, and true parameters."""
    sigma0 = 10
    alpha, beta = 5, 5
    key = random.PRNGKey(seed)
    key, k1 = random.split(key)
    model = GaussianWithNoSummaryStats(K=K, n_obs=n_obs, sigma_0=sigma0,
                                        alpha=alpha, beta=beta)
    true_theta = model.prior_generator(k1, 1)
    true_theta = Theta(loc=true_theta.loc, glob=np.array([1.0])[None, :])
    key, k2 = random.split(key)
    y_obs = model.data_generator(k2, true_theta)
    return model, y_obs, true_theta, key


def extract_per_iteration_metrics(out, model, y_obs, K, N_particles, n_sample=1000):
    """Extract per-iteration metrics + cascade stats from SMC output."""
    sigma2_edges = build_sigma2_reference_bins(model, y_obs, nbins=80)
    rows = []
    n_sim_arr = np.asarray(out["N_sim"], dtype=float)
    eps_arr = np.asarray(out["Eps_values"], dtype=float)
    time_arr = np.asarray(out["Time"], dtype=float)
    unique_arr = np.asarray(out["unique_part"], dtype=float)

    if n_sim_arr.size < 2:
        return rows

    n_sim_cum = np.cumsum(n_sim_arr[1:])
    time_cum = np.cumsum(time_arr[1:])
    epsilons = eps_arr[1:]
    unique = unique_arr[1:]

    thetas_list = out["Thetas"]
    weights_list = out["Weights"]
    perm_list = out.get("Zs_index", None)
    cascade_stats_list = out.get("Smart_stats", None)

    n_steps = min(
        len(n_sim_cum), len(epsilons), len(time_cum), len(unique),
        max(0, len(thetas_list) - 1),
    )
    for i in range(n_steps):
        thetas_i = thetas_list[i + 1] if i + 1 < len(thetas_list) else None
        weights_i = weights_list[i + 1] if i + 1 < len(weights_list) else None
        perm_i = perm_list[i + 1] if perm_list and i + 1 < len(perm_list) else None
        if thetas_i is None:
            continue

        denom = K * max(float(unique[i]), 1e-12) * max(float(N_particles), 1.0)
        n_sim_norm = float(n_sim_cum[i]) / denom * n_sample
        time_norm = float(time_cum[i]) / denom * n_sample

        kl_s2 = empirical_kl_sigma2_abc_vs_true(
            model, y_obs, thetas_i, weights=weights_i, edges=sigma2_edges)
        score_joint = expected_neg_log_joint_true(
            model, y_obs, thetas_i, weights=weights_i, perm=perm_i)

        row = {
            "iteration": i + 1,
            "n_sim": n_sim_norm,
            "time": time_norm,
            "n_sim_raw": float(n_sim_cum[i]),
            "time_raw": float(time_cum[i]),
            "epsilon": float(epsilons[i]),
            "kl_sigma2": kl_s2,
            "score_joint": score_joint,
            "unique_particles": float(unique[i]),
        }

        # Add cascade stats if available
        if cascade_stats_list and i < len(cascade_stats_list):
            cs = cascade_stats_list[i]
            if cs:
                row["cascade_n_total"] = cs.get("n_total", 0)
                row["cascade_rate_identity"] = cs.get("rate_identity", 0)
                row["cascade_n_accepted_identity"] = cs.get("n_accepted_identity", 0)
                row["cascade_rate_swap"] = cs.get("rate_swap", 0)
                row["cascade_n_accepted_swap"] = cs.get("n_accepted_swap", 0)
                row["cascade_n_lsa"] = cs.get("n_lsa", 0)
                row["cascade_rate_lsa"] = cs.get("rate_lsa", 0)
                row["cascade_rate_hilbert"] = cs.get("rate_hilbert", 0)
                row["cascade_n_accepted_hilbert"] = cs.get("n_accepted_hilbert", 0)
                row["cascade_rate_sinkhorn"] = cs.get("rate_sinkhorn", 0)
                row["cascade_n_accepted_sinkhorn"] = cs.get("n_accepted_sinkhorn", 0)

        rows.append(row)
    return rows


def run_macro(K_values, n_particles, seed, n_obs, N_sim_max_factor, out_dir):
    """Run macro benchmark: full SMC comparison across K and methods."""
    all_records = []

    for K in K_values:
        print(f"\n{'='*60}")
        print(f"  MACRO BENCHMARK: K = {K}")
        print(f"{'='*60}")

        N_sim_max = K * N_sim_max_factor
        model, y_obs, true_theta, key = setup_experiment(K, seed, n_obs=n_obs)

        for name, cascade in MACRO_CONFIGS:
            print(f"\n  --- {name}  (cascade={cascade}) ---")
            key, subkey = random.split(key)
            model.reset_weights_distance()

            t0 = time.time()
            try:
                out = perm_abc_smc(
                    key=subkey,
                    model=model,
                    n_particles=n_particles,
                    epsilon_target=0,
                    y_obs=y_obs,
                    kernel=KernelTruncatedRW,
                    verbose=1,
                    Final_iteration=0,
                    update_weights_distance=False,
                    stopping_accept_rate=0.0,
                    N_sim_max=N_sim_max,
                    try_identity="identity" in cascade,
                    try_hilbert="hilbert" in cascade,
                    try_sinkhorn="sinkhorn" in cascade,
                    try_swaps="swap" in cascade,
                    try_lsa="lsa" in cascade,
                )
                wall = time.time() - t0
                rows = extract_per_iteration_metrics(
                    out, model, y_obs, K, n_particles)
                print(f"  {name}: {len(rows)} iterations, wall = {wall:.1f}s")

                for r in rows:
                    r["method"] = name
                    r["K"] = K
                all_records.extend(rows)

            except Exception as e:
                wall = time.time() - t0
                print(f"  {name}: FAILED after {wall:.1f}s — {e}")

            gc.collect()

    # Save pickle
    pkl_path = os.path.join(out_dir, "macro_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "records": all_records,
            "K_values": K_values,
            "seed": seed,
            "n_particles": n_particles,
            "n_obs": n_obs,
        }, f)
    print(f"\nMacro pickle saved: {pkl_path}")

    # Save CSV
    if all_records:
        df = pd.DataFrame(all_records)
        csv_path = os.path.join(out_dir, "macro_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Macro CSV saved: {csv_path}")

        # Print cascade stats summary for Smart Swap
        ss = df[df["method"] == "Smart Swap"]
        if not ss.empty and "cascade_rate_identity" in ss.columns:
            print("\n--- Smart Swap cascade stats ---")
            for K in K_values:
                ss_k = ss[ss["K"] == K]
                if ss_k.empty:
                    continue
                id_rate = ss_k["cascade_rate_identity"].mean()
                sw_rate = ss_k["cascade_rate_swap"].mean()
                lsa_rate = ss_k["cascade_rate_lsa"].mean()
                print(f"  K={K}: identity={id_rate:.1%}, swap={sw_rate:.1%}, lsa={lsa_rate:.1%}")

    return all_records


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Final solver benchmark for permABC paper")
    parser.add_argument("--level", choices=["micro", "macro", "both"],
                        default="both")
    parser.add_argument("--K", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--n-obs", nargs="+", type=int, default=[2, 10, 50])
    parser.add_argument("--n-particles", type=int, default=1000)
    parser.add_argument("--n-repeats-micro", type=int, default=5)
    parser.add_argument("--n-particles-micro", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsim-factor", type=int, default=1_000_000,
                        help="N_sim_max = K * nsim_factor")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or str(
        _PROJECT_ROOT / "experiments" / "results" / "solver_final_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Numba: {_HAS_NUMBA}  |  CGAL: {_HAS_CGAL}  |  Custom LSA: {_HAS_CUSTOM_LSA}")
    print(f"K: {args.K}  |  n_obs: {args.n_obs}  |  seed={args.seed}")
    print(f"Output: {out_dir}\n")

    if args.level in ("micro", "both"):
        print("=" * 60)
        print("MICRO BENCHMARK")
        print("=" * 60)
        run_micro(args.K, args.n_obs, args.n_particles_micro,
                  args.n_repeats_micro, args.seed, out_dir)

    if args.level in ("macro", "both"):
        print("\n" + "=" * 60)
        print("MACRO BENCHMARK (full SMC)")
        print("=" * 60)
        run_macro(args.K, args.n_particles, args.seed,
                  n_obs=10, N_sim_max_factor=args.nsim_factor,
                  out_dir=out_dir)


if __name__ == "__main__":
    main()
