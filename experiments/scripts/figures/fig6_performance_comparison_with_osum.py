#!/usr/bin/env python3
"""
Figure 6: Performance comparison with over-sampling and under-matching methods.

This script generates performance comparison plots between ABC algorithms
including over-sampling and under-matching methods.

Usage:
    python fig6_performance_comparison_with_osum.py
    python fig6_performance_comparison_with_osum.py --K 20 --K_outliers 4 --seed 42
    python fig6_performance_comparison_with_osum.py --rerun experiments/results/performance_K_20_outliers_4_osum_True_seed_42.pkl
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import (
    METHOD_COLORS, METHOD_MARKERS, METHOD_LINESTYLES, METHOD_ORDER,
    setup_matplotlib, save_figure,
    records_from_pkl, detect_joint_key, joint_ylabel, metric_ylabel,
    extract_series, plot_method_panel, find_project_root,
)

setup_matplotlib()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate Figure 6: Performance comparison with OSUM methods"
    )
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--K_outliers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--N_points', type=int, default=1000000)
    parser.add_argument('--N_particles', type=int, default=1000)
    parser.add_argument('--plot', type=str, choices=['nsim', 'time', 'both'], default='both')
    parser.add_argument('--output-dir', type=str, default="experiments")
    parser.add_argument('--rerun', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Figure 6: Performance comparison with OSUM methods")
    print(f"Parameters: K={args.K}, K_outliers={args.K_outliers}, seed={args.seed}")

    project_root = find_project_root()
    main_script_path = project_root / 'experiments' / 'scripts' / 'run_performance_comparison.py'
    results_dir = project_root / 'experiments' / 'results' / 'performance_comparison'

    # ── Check for existing results to reuse ────────────────────────────────
    rerun_path = None
    if args.rerun:
        rerun_path = args.rerun
        print(f"User specified --rerun. Using file: {rerun_path}")
    else:
        results_filename = f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_True_seed_{args.seed}.pkl"
        results_filepath = results_dir / results_filename
        if results_filepath.exists():
            rerun_path = str(results_filepath)
            print(f"Found existing results file. Re-running in plot-only mode from: {rerun_path}")

    # ── Build and execute the command ──────────────────────────────────────
    if rerun_path:
        cmd_args = [
            sys.executable, str(main_script_path),
            '--rerun', rerun_path,
            '--output-dir', str(project_root),
            '--plot', args.plot,
            '--osum',
        ]
    else:
        print("No results file found. Running full experiment...")
        cmd_args = [
            sys.executable, str(main_script_path),
            '--K', str(args.K), '--K_outliers', str(args.K_outliers),
            '--seed', str(args.seed), '--N_points', str(args.N_points),
            '--N_particles', str(args.N_particles),
            '--plot', args.plot,
            '--output-dir', str(project_root),
            '--osum',
        ]

    print(f"\nExecuting: {' '.join(cmd_args)}\n")
    try:
        env = os.environ.copy()
        env_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(project_root) + (":" + env_pythonpath if env_pythonpath else "")
        subprocess.run(cmd_args, check=True, cwd=str(project_root), env=env)
        print("\nFigure 6 generation complete!")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Script failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)

    # ── Additional KL plots ───────────────────────────────────────────────
    try:
        if rerun_path:
            pkl_path = rerun_path
        else:
            pkl_path = str(
                results_dir / f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_True_seed_{args.seed}.pkl"
            )
        print(f"\nLoading results for KL plots: {pkl_path}")

        records, data = records_from_pkl(pkl_path)
        if not records:
            raise ValueError("Empty summary_df; cannot plot KL.")

        # ── Run ABC-Gibbs if not already in records ───────────────────────
        has_gibbs = any(r.get("method") == "ABC-Gibbs" for r in records)
        if not has_gibbs:
            try:
                scripts_dir = Path(__file__).resolve().parent.parent
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                from abc_gibbs_gaussian import run_gibbs_for_benchmark
                from jax import random as jrandom

                exp_setup = data.get("experiment_setup", {})
                model_obj = exp_setup.get("model") or data.get("model")
                y_obs_obj = exp_setup.get("y_obs") or data.get("y_obs")

                if model_obj is not None and y_obs_obj is not None:
                    K_val = data.get("K", args.K)
                    seed_val = data.get("seed", args.seed)
                    perm_nsims = [r.get("n_sim_raw") or r.get("n_sim", 0) for r in records
                                  if r.get("method") == "permABC-SMC"]
                    perm_nsims = [x for x in perm_nsims if x and float(x) > 0]
                    budget = int(max(perm_nsims)) if perm_nsims else 500_000
                    print(f"Running ABC-Gibbs (budget={budget})...")
                    gibbs_key = jrandom.PRNGKey(seed_val + 999)
                    gibbs_records = run_gibbs_for_benchmark(
                        gibbs_key, model_obj, y_obs_obj, K_val, budget,
                    )
                    records.extend(gibbs_records)
                    print(f"  Added {len(gibbs_records)} ABC-Gibbs checkpoints")
                else:
                    print("WARNING: model/y_obs not in pickle, cannot run ABC-Gibbs")
            except Exception as e:
                print(f"WARNING: ABC-Gibbs failed: {e}")

        jk = detect_joint_key(records)
        jl = joint_ylabel(jk)
        methods = [m for m in METHOD_ORDER if any(r.get("method") == m for r in records)]

        figures_dir = project_root / "experiments" / "figures" / "fig6"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # -- All diagnostic metrics in a combined grid -------------------------
        x_axes_grid = [
            ("n_sim", "Nsim (per 1000 unique)"),
            ("time",  "Time (seconds)"),
        ]
        y_metrics_grid = [
            ("kl_sigma2", metric_ylabel("kl_sigma2"), True),
            (jk,          jl,                         jk != "score_joint"),
            ("kl_mu_avg", metric_ylabel("kl_mu_avg"), True),
            ("w2_sigma2", metric_ylabel("w2_sigma2"), True),
            ("w2_mu_avg", metric_ylabel("w2_mu_avg"), True),
            ("sw2_joint", metric_ylabel("sw2_joint"), True),
        ]
        n_ymet = len(y_metrics_grid)
        fig, axes = plt.subplots(n_ymet, 2, figsize=(14, 5 * n_ymet))
        for row_i, (yk, yl, ly) in enumerate(y_metrics_grid):
            for col_i, (xk, xl) in enumerate(x_axes_grid):
                plot_method_panel(axes[row_i, col_i], records, methods,
                                  xk, yk, xl, yl, log_y=ly)
                if row_i == 0 and col_i == 0:
                    axes[row_i, col_i].legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, figures_dir / f"fig6_all_metrics_K_{args.K}_outliers_{args.K_outliers}_seed_{args.seed}.pdf")
        print("All-metrics grid saved")

        # -- Individual panels -------------------------------------------------
        individual_panels = [
            ("n_sim",   "epsilon",   "Nsim (per 1000 unique)", r"$\varepsilon$",  "epsilon_vs_nsim"),
            ("time",    "epsilon",   "Time (seconds)",          r"$\varepsilon$",  "epsilon_vs_time"),
            ("epsilon", jk,          r"$\varepsilon$",          jl,
             f"{'score_joint' if jk == 'score_joint' else 'kl_joint'}_vs_epsilon"),
            ("epsilon", "kl_sigma2", r"$\varepsilon$", metric_ylabel("kl_sigma2"), "kl_sigma2_vs_epsilon"),
            ("epsilon", "kl_mu_avg", r"$\varepsilon$", metric_ylabel("kl_mu_avg"), "kl_mu_avg_vs_epsilon"),
            ("epsilon", "w2_sigma2", r"$\varepsilon$", metric_ylabel("w2_sigma2"), "w2_sigma2_vs_epsilon"),
            ("epsilon", "w2_mu_avg", r"$\varepsilon$", metric_ylabel("w2_mu_avg"), "w2_mu_avg_vs_epsilon"),
            ("epsilon", "sw2_joint", r"$\varepsilon$", metric_ylabel("sw2_joint"), "sw2_joint_vs_epsilon"),
            ("n_sim",   "time",      "Nsim (per 1000 unique)", "Time (seconds)",  "time_vs_nsim"),
        ]
        for xk, yk, xl, yl, tag in individual_panels:
            log_y = (yk not in ("score_joint",))
            fig_s, ax_s = plt.subplots(figsize=(7.5, 5.5))
            plot_method_panel(ax_s, records, methods, xk, yk, xl, yl, log_y=log_y)
            ax_s.legend(fontsize=8)
            fig_s.tight_layout()
            save_figure(fig_s, figures_dir / f"fig6_{tag}_K_{args.K}_outliers_{args.K_outliers}_seed_{args.seed}.pdf")
            print(f"{tag} saved")

    except Exception as e:
        print(f"Warning: KL plot generation failed: {e}")

if __name__ == "__main__":
    main()
