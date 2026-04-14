"""
Unified benchmark for assignment methods in permABC-SMC.

Two levels:
  --level micro  : per-iteration assignment quality and speed (no full SMC)
  --level macro  : full SMC runs comparing posterior quality
  --level both   : both

Usage:
  python run_assignment_benchmark.py --level micro --K 5 10 20 --seed 0
  python run_assignment_benchmark.py --level macro --K 5 10 --seed 42
  python run_assignment_benchmark.py --level both --K 5 --quick
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, gc, argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from jax import random

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.assignment.dispatch import (
    optimal_index_distance, _build_cost_and_global,
    _apply_lsa, _apply_sinkhorn, _apply_hilbert, _apply_swap,
    do_swap, do_hilbert,
)
from permabc.assignment.solvers.lsa import solve_lsa, solve_lsa_custom, _HAS_CUSTOM_LSA
from permabc.assignment.solvers.sinkhorn import sinkhorn_assignment, _HAS_NUMBA as _SINK_HAS_NUMBA
from permabc.assignment.solvers.swap import _HAS_NUMBA
from permabc.assignment.solvers.hilbert import _HAS_CGAL
from permabc.assignment.distances import compute_total_distance


# =====================================================================
# Method definitions
# =====================================================================

MICRO_METHODS = [
    {"name": "LSA (scipy)",      "fn": "lsa"},
    {"name": "LSA custom cold",  "fn": "lsa_custom_cold"},
    {"name": "LSA custom warm",  "fn": "lsa_custom_warm"},
    {"name": "Hilbert",          "fn": "hilbert"},
    {"name": "Hilbert+Swap",     "fn": "hilbert_swap"},
    {"name": "Sinkhorn (numpy)", "fn": "sinkhorn_numpy"},
    {"name": "Sinkhorn (numba)", "fn": "sinkhorn_numba"},
    {"name": "Sinkhorn+Swap",    "fn": "sinkhorn_swap"},
    {"name": "Swap only",        "fn": "swap_only"},
]

MACRO_METHODS = [
    {"cascade": ["identity", "lsa"],                        "label": "LSA"},
    {"cascade": ["identity", "lsa", "swap"],                "label": "LSA+Swap"},
    {"cascade": ["identity", "hilbert"],                    "label": "Hilbert"},
    {"cascade": ["identity", "hilbert", "swap"],            "label": "Hilbert+Swap"},
    {"cascade": ["identity", "sinkhorn"],                   "label": "Sinkhorn"},
    {"cascade": ["identity", "sinkhorn", "swap"],           "label": "Sinkhorn+Swap"},
    {"cascade": ["identity", "swap"],                       "label": "Swap only"},
    {"cascade": ["identity", "swap", "lsa"],                "label": "Smart Swap"},
    {"cascade": ["identity", "hilbert", "lsa"],             "label": "Smart Hilbert"},
    {"cascade": ["identity", "hilbert", "swap", "lsa"],     "label": "Smart H+S"},
]


# =====================================================================
# MICRO benchmark: per-batch assignment quality and speed
# =====================================================================

def _solve_micro(fn_name, local_mats, global_d, model, zs, y_obs, M, K,
                 ys_lsa, zs_lsa):
    """Run one micro method, return (dists, ys, zs, elapsed)."""
    t0 = time.perf_counter()

    if fn_name == "lsa":
        ys, zs_idx = solve_lsa(local_mats, parallel=True)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif fn_name == "lsa_custom_cold":
        if not _HAS_CUSTOM_LSA:
            return None, None, None, np.nan, np.nan, np.nan
        ys, zs_idx = solve_lsa_custom(local_mats, init_col4row=None, parallel=True)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif fn_name == "lsa_custom_warm":
        if not _HAS_CUSTOM_LSA:
            return None, None, None, np.nan, np.nan, np.nan
        # Use LSA solution as warm-start hint (simulates warm-start from previous iter)
        ys, zs_idx = solve_lsa_custom(local_mats, init_col4row=zs_lsa, parallel=True)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif fn_name == "hilbert":
        y_ref = y_obs[0, :K]
        weights = np.asarray(model.weights_distance[:K])
        zs_slice = zs[:, :K]
        h_dist, ys, zs_idx = do_hilbert(zs_slice, y_ref, weights)
        dists = np.sqrt(h_dist ** 2 + global_d)

    elif fn_name == "hilbert_swap":
        y_ref = y_obs[0, :K]
        weights = np.asarray(model.weights_distance[:K])
        zs_slice = zs[:, :K]
        _, ys, zs_idx = do_hilbert(zs_slice, y_ref, weights)
        ys, zs_idx = do_swap(local_mats, ys, zs_idx)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif fn_name == "sinkhorn_numpy":
        ys, zs_idx = sinkhorn_assignment(local_mats, backend="numpy")
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif fn_name == "sinkhorn_numba":
        ys, zs_idx = sinkhorn_assignment(local_mats, backend="numba")
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif fn_name == "sinkhorn_swap":
        ys, zs_idx = sinkhorn_assignment(local_mats, backend="numba")
        ys, zs_idx = do_swap(local_mats, ys, zs_idx)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)

    elif fn_name == "swap_only":
        N = local_mats.shape[0]
        ys = np.tile(np.arange(K, dtype=np.int32), (N, 1))
        zs_idx = np.tile(np.arange(K, dtype=np.int32), (N, 1))
        ys, zs_idx = do_swap(local_mats, ys, zs_idx)
        dists = compute_total_distance(zs_idx, ys, local_mats, global_d)
    else:
        raise ValueError(fn_name)

    elapsed = time.perf_counter() - t0

    # Quality metrics vs LSA (scipy)
    d_lsa = compute_total_distance(zs_lsa, ys_lsa, local_mats, global_d)
    ratio = np.mean(dists / np.where(d_lsa > 0, d_lsa, 1e-30))

    # Hamming distance to LSA assignment
    hamming = np.mean(zs_idx != zs_lsa)

    return dists, ys, zs_idx, elapsed, float(ratio), float(hamming)


def run_micro(K_values, n_obs_values, n_particles, n_repeats, seed, out_dir):
    """Run micro benchmark: measure assignment quality and speed per batch."""
    rows = []
    key = random.PRNGKey(seed)
    total = len(K_values) * len(n_obs_values) * n_repeats * len(MICRO_METHODS)
    done = 0
    t_global = time.perf_counter()

    for K in K_values:
        for n_obs in n_obs_values:
            model = GaussianWithNoSummaryStats(K=K, n_obs=n_obs)

            for rep in range(n_repeats):
                key, k1, k2 = random.split(key, 3)
                true_theta = model.prior_generator(k1, n_particles)
                zs = np.asarray(model.data_generator(k2, true_theta))
                y_obs_single = np.asarray(model.data_generator(
                    random.split(key)[0],
                    model.prior_generator(random.split(key)[1], 1)
                ))
                if y_obs_single.ndim == 2:
                    y_obs_single = y_obs_single[None, ...]

                # Build cost matrices (shared across methods)
                local_mats = np.asarray(model.distance_matrices_loc(zs, y_obs_single))
                global_d = np.asarray(model.distance_global(zs, y_obs_single))
                local_mats = np.where(np.isinf(local_mats), 1e12, local_mats)

                # Reference: LSA (always first, used for ratios)
                t0 = time.perf_counter()
                ys_lsa, zs_lsa = solve_lsa(local_mats, parallel=True)
                lsa_time = time.perf_counter() - t0
                d_lsa = compute_total_distance(zs_lsa, ys_lsa, local_mats, global_d)

                for m in MICRO_METHODS:
                    _, _, _, elapsed, ratio, hamming = _solve_micro(
                        m["fn"], local_mats, global_d, model, zs,
                        y_obs_single, K, K, ys_lsa, zs_lsa
                    )
                    if np.isnan(elapsed):
                        done += 1
                        continue
                    rows.append({
                        "K": K, "n_obs": n_obs, "seed": seed, "repeat": rep,
                        "method": m["name"],
                        "time_s": elapsed,
                        "lsa_time_s": lsa_time,
                        "speedup": lsa_time / max(elapsed, 1e-10),
                        "distance_ratio": ratio,
                        "hamming": hamming,
                        "n_particles": n_particles,
                    })
                    done += 1

                elapsed_total = time.perf_counter() - t_global
                if done > 0:
                    eta = elapsed_total / done * (total - done)
                    print(f"  [{done:4d}/{total}] K={K:3d} n_obs={n_obs:3d} rep={rep}  "
                          f"elapsed={elapsed_total:.0f}s  ETA={eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "micro_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMicro benchmark done — {len(rows)} rows saved to {csv_path}")

    # Summary
    summary = df.groupby(["K", "method"]).agg(
        time_median=("time_s", "median"),
        speedup_median=("speedup", "median"),
        ratio_median=("distance_ratio", "median"),
        ratio_max=("distance_ratio", "max"),
        hamming_median=("hamming", "median"),
    ).reset_index()
    summary_path = os.path.join(out_dir, "micro_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    return df


# =====================================================================
# MACRO benchmark: full SMC posterior quality
# =====================================================================

def posterior_mse(result, true_loc, true_glob):
    theta = result['Thetas'][-1]
    w = np.asarray(result['Weights'][-1])
    w = w / w.sum()
    loc = np.asarray(theta.loc[:, :, 0])
    glob_val = np.asarray(theta.glob[:, 0])
    post_loc = np.average(np.sort(loc, axis=1), weights=w, axis=0)
    post_glob = np.average(glob_val, weights=w)
    true_sorted = np.sort(np.asarray(true_loc).flatten())
    return float(np.mean((post_loc - true_sorted)**2)), float((post_glob - float(true_glob))**2)


def run_macro(K_values, n_obs, n_particles, alpha, n_iter_max, n_sim_max,
              n_repeats, seed, out_dir):
    """Run macro benchmark: full SMC with different assignment methods."""
    rows = []
    key = random.PRNGKey(seed)
    total = len(K_values) * n_repeats * len(MACRO_METHODS)
    done = 0
    t_global = time.perf_counter()

    for K in K_values:
        for rep in range(n_repeats):
            model = GaussianWithNoSummaryStats(K=K, n_obs=n_obs)
            key, k1, k2 = random.split(key, 3)
            true_theta = model.prior_generator(k1, 1)
            y_obs = np.asarray(model.data_generator(k2, true_theta))
            true_loc = np.asarray(true_theta.loc[0, :, 0])
            true_glob = np.asarray(true_theta.glob[0, 0])

            for m in MACRO_METHODS:
                key, k_run = random.split(key)
                try:
                    t0 = time.perf_counter()
                    res = perm_abc_smc(
                        key=k_run, model=model, n_particles=n_particles,
                        epsilon_target=0.0, y_obs=y_obs,
                        kernel=KernelTruncatedRW, alpha_epsilon=alpha,
                        verbose=0, N_iteration_max=n_iter_max,
                        N_sim_max=n_sim_max,
                        try_identity="identity" in m['cascade'],
                        try_hilbert="hilbert" in m['cascade'],
                        try_sinkhorn="sinkhorn" in m['cascade'],
                        try_swaps="swap" in m['cascade'],
                        try_lsa="lsa" in m['cascade'],
                    )
                    elapsed = time.perf_counter() - t0
                    mse_loc, mse_glob = posterior_mse(res, true_loc, true_glob)
                    row = {
                        'K': K, 'n_obs': n_obs, 'seed': seed, 'repeat': rep,
                        'method': m['label'],
                        'time': elapsed,
                        'mse_loc': mse_loc,
                        'mse_glob': mse_glob,
                        'n_iters': len(res['Eps_values']) - 1,
                        'final_eps': float(res['Eps_values'][-1]),
                        'total_nsim': int(np.sum(res['N_sim'])),
                        'n_particles': n_particles,
                        'alpha': alpha,
                    }
                except Exception as e:
                    row = {
                        'K': K, 'n_obs': n_obs, 'seed': seed, 'repeat': rep,
                        'method': m['label'],
                        'time': np.nan, 'mse_loc': np.nan, 'mse_glob': np.nan,
                        'n_iters': np.nan, 'final_eps': np.nan,
                        'total_nsim': np.nan, 'n_particles': n_particles,
                        'alpha': alpha,
                    }
                    print(f"  ERROR K={K} rep={rep} {m['label']}: {e}")

                rows.append(row)
                done += 1
                elapsed_total = time.perf_counter() - t_global
                eta = elapsed_total / done * (total - done)
                print(f"  [{done:4d}/{total}] K={K:3d} rep={rep} {m['label']:20s}  "
                      f"t={row.get('time',0):.1f}s  elapsed={elapsed_total:.0f}s  "
                      f"ETA={eta:.0f}s", flush=True)

            gc.collect()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "macro_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMacro benchmark done — {len(rows)} rows saved to {csv_path}")

    # Summary
    summary = df.groupby(["K", "method"]).agg(
        time_median=("time", "median"),
        mse_loc_median=("mse_loc", "median"),
        mse_glob_median=("mse_glob", "median"),
        n_iters_median=("n_iters", "median"),
        final_eps_median=("final_eps", "median"),
        nsim_median=("total_nsim", "median"),
    ).reset_index()
    # Add speedup and relative MSE vs LSA
    lsa_time = summary.loc[summary['method'] == 'LSA'].set_index('K')['time_median']
    lsa_mse = summary.loc[summary['method'] == 'LSA'].set_index('K')['mse_loc_median']
    summary['speedup'] = summary.apply(
        lambda r: lsa_time.get(r['K'], np.nan) / max(r['time_median'], 1e-10), axis=1)
    summary['rel_mse'] = summary.apply(
        lambda r: r['mse_loc_median'] / max(lsa_mse.get(r['K'], np.nan), 1e-10), axis=1)

    summary_path = os.path.join(out_dir, "macro_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    return df


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Assignment benchmark for permABC")
    parser.add_argument("--level", choices=["micro", "macro", "both"], default="both")
    parser.add_argument("--K", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--n-obs", nargs="+", type=int, default=[2, 5, 10, 25, 50, 100])
    parser.add_argument("--n-particles", type=int, default=500)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with small grid for testing")
    args = parser.parse_args()

    if args.quick:
        args.K = [5, 10]
        args.n_obs = [2, 10]
        args.n_particles = 200
        args.n_repeats = 1

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'results', 'assignment_benchmark_final')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Numba: {_HAS_NUMBA}  |  CGAL: {_HAS_CGAL}  |  Custom LSA: {_HAS_CUSTOM_LSA}  |  Sinkhorn Numba: {_SINK_HAS_NUMBA}")
    print(f"K: {args.K}  |  n_obs: {args.n_obs}  |  N={args.n_particles}  "
          f"|  repeats={args.n_repeats}  |  seed={args.seed}")
    print(f"Output: {out_dir}\n")

    if args.level in ("micro", "both"):
        print("=" * 60)
        print("MICRO BENCHMARK: per-batch assignment quality and speed")
        print("=" * 60)
        run_micro(args.K, args.n_obs, args.n_particles,
                  args.n_repeats, args.seed, out_dir)

    if args.level in ("macro", "both"):
        print("\n" + "=" * 60)
        print("MACRO BENCHMARK: full SMC posterior quality")
        print("=" * 60)
        run_macro(args.K, n_obs=10, n_particles=args.n_particles,
                  alpha=args.alpha, n_iter_max=30, n_sim_max=200_000,
                  n_repeats=args.n_repeats, seed=args.seed, out_dir=out_dir)


if __name__ == "__main__":
    main()
