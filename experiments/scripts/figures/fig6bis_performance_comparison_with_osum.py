#!/usr/bin/env python3
"""
Figure 6bis: Efficiency comparison of ABC methods (with OSUM).

Generates a multi-row panel figure comparing ABC-Vanilla, permABC-Vanilla,
ABC-SMC, ABC-PMC, permABC-SMC, permABC-SMC-OS, and permABC-SMC-UM
on the Gaussian toy model.

  Rows: score_joint, KL_sigma2, KL_mu_avg, W2_sigma2, W2_mu_avg
  Cols: N_sim, Time, epsilon

Also generates individual standalone PDFs for each panel.

Usage:
    python fig6bis_performance_comparison_with_osum.py
    python fig6bis_performance_comparison_with_osum.py --rerun path/to/results.pkl
    python fig6bis_performance_comparison_with_osum.py --K 20 --K_outliers 4 --seed 42
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import (
    METHOD_COLORS, METHOD_MARKERS, METHOD_LINESTYLES, METHOD_ORDER,
    setup_matplotlib, save_figure,
    records_from_pkl, detect_joint_key, joint_ylabel, metric_ylabel,
    plot_method_panel, find_project_root,
)

setup_matplotlib()


# -- Main ---------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Figure 6bis: Efficiency (with OSUM)")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--K_outliers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N_points", type=int, default=1_000_000)
    parser.add_argument("--N_particles", type=int, default=1000)
    parser.add_argument("--rerun", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_root = find_project_root()

    results_dir = project_root / "experiments" / "results" / "performance_comparison"
    figures_dir = project_root / "experiments" / "figures" / "fig6bis"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -- Locate or generate pickle ---------------------------------------------
    pkl_path = args.rerun
    if pkl_path is None:
        cand = results_dir / f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_True_seed_{args.seed}.pkl"
        if cand.exists():
            pkl_path = str(cand)
            print(f"Found existing results: {pkl_path}")

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
            "--osum", "--output-dir", str(project_root),
        ]
        subprocess.run(cmd, check=True, cwd=str(project_root), env=env)
        pkl_path = str(results_dir / f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_True_seed_{args.seed}.pkl")

    # -- Load data -------------------------------------------------------------
    records, meta = records_from_pkl(pkl_path)
    K = meta.get("K", args.K)
    seed = meta.get("seed", args.seed)
    K_outliers = meta.get("K_outliers", args.K_outliers)

    methods = [m for m in METHOD_ORDER if any(r.get("method") == m for r in records)]
    jk = detect_joint_key(records)
    jl = joint_ylabel(jk)
    log_y_joint = jk != "score_joint"

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
        if row == 0 and col == 0:
            axes[row, col].legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Figure 6bis  --  K = {K},  K_outliers = {K_outliers},  seed = {seed}",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, figures_dir / f"fig6bis_K_{K}_outliers_{K_outliers}_seed_{seed}.pdf")
    print(f"Saved combined figure")

    # -- Individual standalone panels ------------------------------------------
    x_short = {"n_sim": "nsim", "time": "time", "epsilon": "epsilon"}
    for xk, yk, xl, yl, ly in panels:
        tag = f"{yk}_vs_{x_short[xk]}"
        fig_s, ax_s = plt.subplots(figsize=(7, 5))
        plot_method_panel(ax_s, records, methods, xk, yk, xl, yl, log_y=ly)
        ax_s.legend(fontsize=9)
        fig_s.tight_layout()
        save_figure(fig_s, figures_dir / f"fig6bis_{tag}_K_{K}_outliers_{K_outliers}_seed_{seed}.pdf")
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
            if row == 0 and col == 0:
                axes_r[row, col].legend(fontsize=8, loc="best")
        for idx in range(len(raw_panels), n_raw_rows * 3):
            row, col = divmod(idx, 3)
            axes_r[row, col].set_visible(False)
        fig_r.suptitle(
            f"Figure 6bis (raw cost)  --  K = {K},  K_outliers = {K_outliers},  seed = {seed}",
            fontsize=14, y=1.01,
        )
        fig_r.tight_layout()
        save_figure(fig_r, figures_dir / f"fig6bis_raw_K_{K}_outliers_{K_outliers}_seed_{seed}.pdf")
        print(f"Saved raw-cost figure")

    print("Done.")


if __name__ == "__main__":
    main()
