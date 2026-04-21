#!/usr/bin/env python3
"""
Figure 4bis: Efficiency comparison of ABC methods (without OSUM).

Generates a multi-row panel figure comparing ABC-Vanilla, permABC-Vanilla,
ABC-SMC, ABC-PMC, and permABC-SMC on the Gaussian toy model.

  Rows: score_joint, KL_sigma2, KL_mu_avg, W2_sigma2, W2_mu_avg
  Cols: N_sim, Time, epsilon

Also generates individual standalone PDFs for each panel.

Usage:
    python fig4bis_performance_comparison_without_osum.py
    python fig4bis_performance_comparison_without_osum.py --rerun path/to/results.pkl
    python fig4bis_performance_comparison_without_osum.py --K 20 --seed 42
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
    METHODS_EXCLUDE_NO_OSUM, setup_matplotlib, save_figure,
    records_from_pkl, detect_joint_key, joint_ylabel, metric_ylabel,
    plot_method_panel, find_project_root,
)
from diagnostics import sample_true_posterior, sliced_w2_joint
from permabc.utils.functions import Theta

setup_matplotlib()


def compute_sw2_floor(model, y_obs, n_particles, n_projections=200, seed=0):
    """Compute SW2(true, true) — the MC noise floor for N_particles."""
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1000)
    mu1, s2_1 = sample_true_posterior(model, y_obs, n_particles, rng=rng1)
    mu2, s2_2 = sample_true_posterior(model, y_obs, n_particles, rng=rng2)

    ref_joint = np.column_stack([mu2, s2_2])
    abc_joint = np.column_stack([mu1, s2_1])
    dim = abc_joint.shape[1]

    rng_proj = np.random.default_rng(seed + 1)
    directions = rng_proj.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    sw2_sq = 0.0
    for omega in directions:
        a_sorted = np.sort(abc_joint @ omega)
        b_sorted = np.sort(ref_joint @ omega)
        sw2_sq += np.mean((a_sorted - b_sorted) ** 2)
    sw2_sq /= n_projections
    return float(np.sqrt(sw2_sq))


def add_sw2_floor_line(ax, sw2_floor):
    """Add a horizontal dashed line for SW2(true, true)."""
    ax.axhline(sw2_floor, color="gray", linestyle=":", linewidth=1.2,
               label=r"$\mathrm{SW}_2$(true, true)", zorder=0)


# -- Main ---------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Figure 4bis: Efficiency (no OSUM)")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--K_outliers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N_points", type=int, default=1_000_000)
    parser.add_argument("--N_particles", type=int, default=1000)
    parser.add_argument("--rerun", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_root = find_project_root()

    results_dir = project_root / "experiments" / "results" / "performance_comparison"
    figures_dir = project_root / "experiments" / "figures" / "fig4bis"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -- Locate or generate pickle ---------------------------------------------
    pkl_path = args.rerun
    if pkl_path is None:
        candidates = [
            results_dir / f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_False_seed_{args.seed}.pkl",
            results_dir / f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_True_seed_{args.seed}.pkl",
        ]
        for c in candidates:
            if c.exists():
                pkl_path = str(c)
                print(f"Found existing results: {pkl_path}")
                break

    if pkl_path is None:
        print("No results found. Running experiment via run_performance_comparison.py ...")
        main_script = project_root / "experiments" / "scripts" / "run_performance_comparison.py"
        env = os.environ.copy()
        pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(project_root) + (":" + pypath if pypath else "")
        cmd = [
            sys.executable, str(main_script),
            "--K", str(args.K), "--K_outliers", str(args.K_outliers),
            "--seed", str(args.seed), "--N_points", str(args.N_points),
            "--N_particles", str(args.N_particles),
            "--no-osum", "--output-dir", str(project_root),
        ]
        subprocess.run(cmd, check=True, cwd=str(project_root), env=env)
        pkl_path = str(results_dir / f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_False_seed_{args.seed}.pkl")

    # -- Load data -------------------------------------------------------------
    records, meta = records_from_pkl(pkl_path)
    records = [r for r in records if r.get("method") not in METHODS_EXCLUDE_NO_OSUM]
    K = meta.get("K", args.K)
    seed = meta.get("seed", args.seed)

    # -- Run ABC-Gibbs if not already in records -------------------------------
    has_gibbs = any(r.get("method") == "ABC-Gibbs" for r in records)
    if not has_gibbs:
        try:
            from abc_gibbs_gaussian import run_gibbs_for_benchmark
            from jax import random as jrandom

            exp_setup = meta.get("experiment_setup", {})
            model_obj = exp_setup.get("model") or meta.get("model")
            y_obs_obj = exp_setup.get("y_obs") or meta.get("y_obs")
            if model_obj is not None and y_obs_obj is not None:
                perm_nsims = [r.get("n_sim_raw") or r.get("n_sim", 0) for r in records
                              if r.get("method") == "permABC-SMC"]
                perm_nsims = [x for x in perm_nsims if x and float(x) > 0]
                budget = int(max(perm_nsims)) if perm_nsims else 500_000
                print(f"Running ABC-Gibbs (budget={budget})...")
                gibbs_key = jrandom.PRNGKey(seed + 999)
                gibbs_records = run_gibbs_for_benchmark(
                    gibbs_key, model_obj, y_obs_obj, K, budget,
                )
                records.extend(gibbs_records)
                print(f"  Added {len(gibbs_records)} ABC-Gibbs checkpoints")
            else:
                print("WARNING: model/y_obs not in pickle, cannot run ABC-Gibbs")
        except Exception as e:
            print(f"WARNING: ABC-Gibbs failed: {e}")

    methods = [m for m in METHOD_ORDER if any(r.get("method") == m for r in records)]
    jk = detect_joint_key(records)
    jl = joint_ylabel(jk)
    log_y_joint = jk != "score_joint"

    # -- SW2 floor (MC noise) -------------------------------------------------
    sw2_floor = None
    exp_setup = meta.get("experiment_setup", {})
    model_obj = exp_setup.get("model") if exp_setup.get("model") is not None else meta.get("model")
    y_obs_obj = exp_setup.get("y_obs") if exp_setup.get("y_obs") is not None else meta.get("y_obs")
    N_particles = meta.get("N_particles", args.N_particles)
    if model_obj is not None and y_obs_obj is not None:
        print(f"Computing SW2(true, true) floor for N={N_particles}...")
        sw2_floor = compute_sw2_floor(model_obj, y_obs_obj, N_particles)
        print(f"  SW2 floor = {sw2_floor:.4f}")

    # -- Metric definitions ----------------------------------------------------
    x_axes = [
        ("n_sim",   r"$N_{\mathrm{sim}}$ (per 1000 unique)"),
        ("time",    "Time (s, per 1000 unique)"),
        ("epsilon", r"$\varepsilon$"),
    ]
    y_metrics = [
        (jk,          jl,                        log_y_joint),
        ("kl_sigma2", metric_ylabel("kl_sigma2"), True),
        ("kl_mu_avg", metric_ylabel("kl_mu_avg"), True),
        ("w2_sigma2", metric_ylabel("w2_sigma2"), True),
        ("w2_mu_avg", metric_ylabel("w2_mu_avg"), True),
        ("sw2_joint", metric_ylabel("sw2_joint"), True),
    ]

    # flat list: (xk, yk, xl, yl, log_y)
    panels = [(xk, yk, xl, yl, ly)
              for yk, yl, ly in y_metrics
              for xk, xl in x_axes]

    # -- Combined figure (n_metrics x 3) ---------------------------------------
    n_rows = len(y_metrics)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    for idx, (xk, yk, xl, yl, ly) in enumerate(panels):
        row, col = divmod(idx, 3)
        plot_method_panel(axes[row, col], records, methods, xk, yk, xl, yl, log_y=ly)
        if yk == "sw2_joint" and sw2_floor is not None:
            add_sw2_floor_line(axes[row, col], sw2_floor)
        if row == 0 and col == 0:
            axes[row, col].legend(fontsize=8, loc="best")

    fig.suptitle(f"Figure 4bis  --  K = {K},  seed = {seed}", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, figures_dir / f"fig4bis_K_{K}_seed_{seed}.pdf")
    print(f"Saved combined figure")

    # -- Individual standalone panels ------------------------------------------
    x_short = {"n_sim": "nsim", "time": "time", "epsilon": "epsilon"}
    for xk, yk, xl, yl, ly in panels:
        tag = f"{yk}_vs_{x_short[xk]}"
        fig_s, ax_s = plt.subplots(figsize=(7, 5))
        plot_method_panel(ax_s, records, methods, xk, yk, xl, yl, log_y=ly)
        if yk == "sw2_joint" and sw2_floor is not None:
            add_sw2_floor_line(ax_s, sw2_floor)
        ax_s.legend(fontsize=9)
        fig_s.tight_layout()
        save_figure(fig_s, figures_dir / f"fig4bis_{tag}_K_{K}_seed_{seed}.pdf")
    print(f"Saved individual panels in {figures_dir}")

    # -- Raw cost axes (nsim_raw, time_raw) ------------------------------------
    raw_x = [
        ("n_sim_raw", r"$N_{\mathrm{sim}}$ (total)"),
        ("time_raw",  "Time (s, total)"),
    ]
    raw_panels = [(xk, yk, xl, yl, ly)
                  for yk, yl, ly in y_metrics
                  for xk, xl in raw_x]
    raw_panels += [
        ("n_sim_raw", "epsilon", r"$N_{\mathrm{sim}}$ (total)", r"$\varepsilon$", True),
        ("time_raw",  "epsilon", "Time (s, total)",              r"$\varepsilon$", True),
    ]
    has_raw = any("n_sim_raw" in r for r in records)
    if has_raw:
        n_raw_rows = (len(raw_panels) + 2) // 3
        fig_r, axes_r = plt.subplots(n_raw_rows, 3, figsize=(18, 5 * n_raw_rows))
        if n_raw_rows == 1:
            axes_r = axes_r[None, :]
        for idx, (xk, yk, xl, yl, ly) in enumerate(raw_panels):
            row, col = divmod(idx, 3)
            plot_method_panel(axes_r[row, col], records, methods, xk, yk, xl, yl, log_y=ly)
            if yk == "sw2_joint" and sw2_floor is not None:
                add_sw2_floor_line(axes_r[row, col], sw2_floor)
            if row == 0 and col == 0:
                axes_r[row, col].legend(fontsize=8, loc="best")
        for idx in range(len(raw_panels), n_raw_rows * 3):
            row, col = divmod(idx, 3)
            axes_r[row, col].set_visible(False)
        fig_r.suptitle(f"Figure 4bis (raw cost)  --  K = {K},  seed = {seed}", fontsize=14, y=1.01)
        fig_r.tight_layout()
        save_figure(fig_r, figures_dir / f"fig4bis_raw_K_{K}_seed_{seed}.pdf")
        print(f"Saved raw-cost figure")

    print("Done.")


if __name__ == "__main__":
    main()
