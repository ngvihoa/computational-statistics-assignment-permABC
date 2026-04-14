"""
SMC-context assignment benchmark.

Runs full perm_abc_smc with different assignment methods and compares
posterior quality at each epsilon level to the LSA reference.

This tests methods in the REAL SMC context where sigma_t is available
from the previous iteration (not just cold random cost matrices).

Usage:
  python run_smc_assignment_benchmark.py --K 5 10 20 --seed 42
  python run_smc_assignment_benchmark.py --K 5 --n-repeats 5 --seed 0
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, gc, argparse
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from jax import random

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from experiments.scripts.diagnostics import (
    empirical_kl_sigma2, build_sigma2_reference_bins
)


# =====================================================================
# Method configurations
# =====================================================================

METHODS = [
    {"cascade": ["identity", "lsa"],                        "label": "LSA (reference)"},
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
# Run one SMC and extract per-iteration diagnostics
# =====================================================================

def run_smc_with_diagnostics(key, model, y_obs, n_particles, alpha, n_iter_max,
                             n_sim_max, cascade, kl_edges, verbose=0):
    """
    Run perm_abc_smc and extract per-iteration diagnostics.

    Returns a list of dicts, one per iteration, with:
      - epsilon, time, n_sim, acc_rate
      - kl_sigma2 (vs true posterior)
      - mse_loc, mse_glob (vs true params)
      - posterior samples (for comparison)
    """
    t0 = time.perf_counter()
    res = perm_abc_smc(
        key=key, model=model, n_particles=n_particles,
        epsilon_target=0.0, y_obs=y_obs,
        kernel=KernelTruncatedRW, alpha_epsilon=alpha,
        verbose=verbose, N_iteration_max=n_iter_max,
        N_sim_max=n_sim_max,
        try_identity="identity" in cascade,
        try_hilbert="hilbert" in cascade,
        try_sinkhorn="sinkhorn" in cascade,
        try_swaps="swap" in cascade,
        try_lsa="lsa" in cascade,
    )
    total_time = time.perf_counter() - t0

    # Extract per-iteration data
    n_iters = len(res['Eps_values'])
    smart_stats_list = res.get('Smart_stats', [{}] * n_iters)
    rows = []
    for t in range(n_iters):
        theta_t = res['Thetas'][t]
        w_t = np.asarray(res['Weights'][t])
        w_t = w_t / w_t.sum()
        eps_t = float(res['Eps_values'][t])

        # KL divergence on sigma2
        kl = empirical_kl_sigma2(model, y_obs, theta_t, weights=w_t,
                                 edges=kl_edges, direction="q_vs_p")

        # MSE
        loc = np.asarray(theta_t.loc[:, :, 0])
        glob_val = np.asarray(theta_t.glob[:, 0])
        post_loc = np.average(np.sort(loc, axis=1), weights=w_t, axis=0)
        post_glob = np.average(glob_val, weights=w_t)

        row = {
            "iter": t,
            "epsilon": eps_t,
            "kl_sigma2": kl,
            "post_glob_mean": post_glob,
            "post_loc_mean": float(np.mean(post_loc)),
        }

        # Cascade acceptance stats
        ss = smart_stats_list[t] if t < len(smart_stats_list) else {}
        row["rate_sigma_t"] = ss.get("rate_sigma_t", np.nan)
        row["n_rejected"] = ss.get("n_rejected", np.nan)
        row["rate_hilbert"] = ss.get("rate_hilbert", np.nan)
        row["rate_swap"] = ss.get("rate_swap", np.nan)
        row["rate_lsa"] = ss.get("rate_lsa", np.nan)
        row["n_accepted_sigma_t"] = ss.get("n_accepted_sigma_t", np.nan)

        rows.append(row)

    return rows, total_time, res


# =====================================================================
# Main benchmark
# =====================================================================

def run_benchmark(K_values, n_obs, n_particles, alpha, n_iter_max, n_sim_max,
                  n_repeats, seed, out_dir, methods=None):
    """Run full SMC benchmark comparing methods at each epsilon."""
    if methods is None:
        methods = METHODS

    all_rows = []
    key = random.PRNGKey(seed)
    total = len(K_values) * n_repeats * len(methods)
    done = 0
    t_global = time.perf_counter()

    for K in K_values:
        for rep in range(n_repeats):
            model = GaussianWithNoSummaryStats(K=K, n_obs=n_obs)
            key, k1, k2, k3 = random.split(key, 4)

            # Generate ground truth
            true_theta = model.prior_generator(k1, 1)
            y_obs = np.asarray(model.data_generator(k2, true_theta))
            true_loc = np.asarray(true_theta.loc[0, :, 0])
            true_glob = float(true_theta.glob[0, 0])

            # Build shared KL bins (same for all methods in this (K, rep) pair)
            kl_edges = build_sigma2_reference_bins(model, y_obs, nbins=80)

            for m in methods:
                key, k_run = random.split(key)
                label = m["label"]

                try:
                    iter_rows, total_time, res = run_smc_with_diagnostics(
                        key=k_run, model=model, y_obs=y_obs,
                        n_particles=n_particles, alpha=alpha,
                        n_iter_max=n_iter_max, n_sim_max=n_sim_max,
                        cascade=m["cascade"],
                        kl_edges=kl_edges, verbose=0,
                    )

                    # Compute final MSE vs true params
                    theta_final = res['Thetas'][-1]
                    w_final = np.asarray(res['Weights'][-1])
                    w_final = w_final / w_final.sum()
                    loc_f = np.asarray(theta_final.loc[:, :, 0])
                    glob_f = np.asarray(theta_final.glob[:, 0])
                    post_loc_f = np.average(np.sort(loc_f, axis=1), weights=w_final, axis=0)
                    post_glob_f = np.average(glob_f, weights=w_final)
                    mse_loc = float(np.mean((post_loc_f - np.sort(true_loc))**2))
                    mse_glob = float((post_glob_f - true_glob)**2)

                    for row in iter_rows:
                        row.update({
                            "K": K, "n_obs": n_obs, "seed": seed, "repeat": rep,
                            "method": label,
                            "total_time": total_time,
                            "n_iters_total": len(iter_rows),
                            "final_mse_loc": mse_loc,
                            "final_mse_glob": mse_glob,
                            "true_glob": true_glob,
                        })
                        all_rows.append(row)

                except Exception as e:
                    print(f"  ERROR K={K} rep={rep} {label}: {e}")
                    import traceback; traceback.print_exc()

                done += 1
                elapsed = time.perf_counter() - t_global
                eta = elapsed / done * (total - done)
                n_its = len(iter_rows) if 'iter_rows' in dir() else '?'
                print(f"  [{done:3d}/{total}] K={K:3d} rep={rep} {label:20s}  "
                      f"t={total_time:.1f}s  iters={n_its}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

            gc.collect()

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "smc_context_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark done — {len(all_rows)} rows saved to {csv_path}")

    # Summary: final iteration only
    df_final = df.loc[df.groupby(["K", "method", "repeat"])["iter"].idxmax()]
    summary = df_final.groupby(["K", "method"]).agg(
        epsilon_median=("epsilon", "median"),
        kl_median=("kl_sigma2", "median"),
        n_iters_median=("n_iters_total", "median"),
        time_median=("total_time", "median"),
        mse_loc_median=("final_mse_loc", "median"),
        mse_glob_median=("final_mse_glob", "median"),
    ).reset_index()
    summary_path = os.path.join(out_dir, "smc_context_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    return df


# =====================================================================
# Plotting
# =====================================================================

def generate_plots(df, out_dir):
    """Generate comparison plots from the benchmark results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ks = sorted(df["K"].unique())
    methods_plot = [m["label"] for m in METHODS]
    colors = {m: f"C{i}" for i, m in enumerate(methods_plot)}

    # ---- Figure 1: KL vs epsilon for each K ----
    fig, axes = plt.subplots(1, len(Ks), figsize=(6 * len(Ks), 5), sharey=False)
    if len(Ks) == 1:
        axes = [axes]

    for ax, K in zip(axes, Ks):
        sub = df[df["K"] == K]
        for m in methods_plot:
            ms = sub[sub["method"] == m]
            if ms.empty:
                continue
            # Median KL over repeats at each iteration
            med = ms.groupby("iter").agg(
                eps=("epsilon", "median"),
                kl=("kl_sigma2", "median"),
            ).reset_index()
            # Remove inf epsilon
            med = med[med["eps"] < 1e10]
            if med.empty:
                continue
            lw = 2.5 if "LSA (ref" in m else 1.2
            ls = "-" if "LSA (ref" in m else "--"
            ax.plot(med["eps"], med["kl"], label=m, color=colors.get(m, "gray"),
                    linewidth=lw, linestyle=ls, alpha=0.85)
        ax.set_xlabel(r"$\varepsilon$", fontsize=12)
        ax.set_ylabel(r"$\mathrm{KL}(q_{\mathrm{ABC}} \| p_{\mathrm{true}})$", fontsize=12)
        ax.set_title(f"K = {K}", fontsize=13)
        ax.invert_xaxis()
        ax.set_yscale("log")

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Posterior quality (KL) vs epsilon — SMC context", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_kl_vs_epsilon.pdf"), bbox_inches="tight")
    print("Saved fig_kl_vs_epsilon.pdf")

    # ---- Figure 2: KL at final iteration (boxplot) ----
    df_final = df.loc[df.groupby(["K", "method", "repeat"])["iter"].idxmax()]

    fig, axes = plt.subplots(1, len(Ks), figsize=(6 * len(Ks), 5))
    if len(Ks) == 1:
        axes = [axes]

    for ax, K in zip(axes, Ks):
        sub = df_final[df_final["K"] == K]
        methods_present = [m for m in methods_plot if m in sub["method"].values]
        data = [sub[sub["method"] == m]["kl_sigma2"].values for m in methods_present]
        labels = [m.replace(" (reference)", "\n(ref)") for m in methods_present]
        bp = ax.boxplot(data, labels=labels, vert=True, patch_artist=True)
        for patch, m in zip(bp["boxes"], methods_present):
            patch.set_facecolor(colors.get(m, "gray"))
            patch.set_alpha(0.6)
        ax.set_ylabel(r"$\mathrm{KL}$", fontsize=12)
        ax.set_title(f"K = {K} — Final KL", fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_kl_final_boxplot.pdf"), bbox_inches="tight")
    print("Saved fig_kl_final_boxplot.pdf")

    # ---- Figure 3: Time vs final KL (Pareto) ----
    fig, axes = plt.subplots(1, len(Ks), figsize=(6 * len(Ks), 5))
    if len(Ks) == 1:
        axes = [axes]

    for ax, K in zip(axes, Ks):
        sub = df_final[df_final["K"] == K]
        for m in methods_plot:
            ms = sub[sub["method"] == m]
            if ms.empty:
                continue
            ax.scatter(ms["total_time"], ms["kl_sigma2"],
                       label=m, color=colors.get(m, "gray"), s=50, alpha=0.8)
        ax.set_xlabel("Total SMC time (s)", fontsize=12)
        ax.set_ylabel(r"Final $\mathrm{KL}$", fontsize=12)
        ax.set_title(f"K = {K} — Speed vs Quality", fontsize=13)
        ax.set_yscale("log")

    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_pareto_time_kl.pdf"), bbox_inches="tight")
    print("Saved fig_pareto_time_kl.pdf")

    # ---- Figure 4: Number of iterations per method ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    df_niters = df_final.groupby(["K", "method"])["n_iters_total"].median().reset_index()
    for m in methods_plot:
        ms = df_niters[df_niters["method"] == m]
        if ms.empty:
            continue
        ax.plot(ms["K"], ms["n_iters_total"], marker="o", label=m,
                color=colors.get(m, "gray"), linewidth=1.5)
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Number of SMC iterations", fontsize=12)
    ax.set_title("SMC iterations to convergence", fontsize=13)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_n_iters.pdf"), bbox_inches="tight")
    print("Saved fig_n_iters.pdf")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="SMC-context assignment benchmark")
    parser.add_argument("--K", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--n-particles", type=int, default=500)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--n-iter-max", type=int, default=30)
    parser.add_argument("--n-sim-max", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", type=str, default=None,
                        help="Path to existing CSV — skip runs, only generate plots")
    args = parser.parse_args()

    if args.quick:
        args.K = [5]
        args.n_particles = 200
        args.n_repeats = 1
        args.n_iter_max = 15
        args.n_sim_max = 50_000

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'results', 'smc_context_benchmark')
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        df = pd.read_csv(args.plot_only)
        generate_plots(df, out_dir)
        return

    print(f"K: {args.K}  |  n_obs: {args.n_obs}  |  N={args.n_particles}  "
          f"|  repeats={args.n_repeats}  |  seed={args.seed}")
    print(f"Max iters: {args.n_iter_max}  |  Max sims: {args.n_sim_max}")
    print(f"Output: {out_dir}\n")

    df = run_benchmark(
        K_values=args.K, n_obs=args.n_obs,
        n_particles=args.n_particles, alpha=args.alpha,
        n_iter_max=args.n_iter_max, n_sim_max=args.n_sim_max,
        n_repeats=args.n_repeats, seed=args.seed, out_dir=out_dir,
    )

    print("\nGenerating plots...")
    generate_plots(df, out_dir)
    print("Done!")


if __name__ == "__main__":
    main()
