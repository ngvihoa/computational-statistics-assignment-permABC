#!/usr/bin/env python3
"""
Figure: SIR sensitivity analysis across sigma values and methods.

Produces:
  - KDE of R0 posterior for each (sigma, method)
  - KDE of gamma posteriors for each (sigma, method)
  - Posterior predictive check (overlay simulated trajectories on observed data)

Usage:
    python fig_sir_sensitivity.py --seed 42 --scale departmental
    python fig_sir_sensitivity.py --seed 42 --scale departmental --sigmas 0.01 0.05 0.10
"""

import sys
import argparse
import pickle
import glob as globmod
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()
RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "sir_real_world_inference"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "fig_sir_sensitivity"


# ── Method definitions ──────────────────────────────────────────────────────

METHOD_STYLES = {
    "perm_smc":         ("permABC-SMC",              "#1f77b4", "-"),
    "abc_smc":          ("ABC-SMC",                  "#8c564b", "--"),
    "abc_smc_g3":       ("ABC-SMC (Gibbs 3b)",       "#d62728", "-."),
    "abc_smc_g5":       ("ABC-SMC (Gibbs 5b)",       "#d62728", ":"),
    "abc_smc_g10":      ("ABC-SMC (Gibbs 10b)",      "#d62728", "--"),
    "abc_smc_g15":      ("ABC-SMC (Gibbs 15b)",      "#d62728", "-"),
    "perm_smc_g3":      ("permABC-SMC (Gibbs 3b)",   "#2ca02c", "-."),
    "perm_smc_g5":      ("permABC-SMC (Gibbs 5b)",   "#2ca02c", ":"),
    "perm_smc_g10":     ("permABC-SMC (Gibbs 10b)",  "#2ca02c", "--"),
    "perm_smc_g15":     ("permABC-SMC (Gibbs 15b)",  "#2ca02c", "-"),
    "abc_gibbs_true":   ("ABC-Gibbs",                "#ff7f0e", "-"),
    "perm_smc_os":      ("permABC-SMC-OS",           "#e377c2", "-"),
    "perm_smc_um":      ("permABC-SMC-UM",           "#9467bd", "-"),
}


def discover_results(scale, seed):
    """Scan results dir and return {(method_tag, sigma): data_dict}."""
    pattern = str(RESULTS_DIR / f"inference_sir_{scale}_*_sigma_*_seed_{seed}.pkl")
    files = globmod.glob(pattern)
    results = {}
    for fpath in sorted(files):
        fname = Path(fpath).stem
        # Parse: inference_sir_{scale}_{method_tag}_sigma_{sigma_str}_seed_{seed}
        parts = fname.split(f"inference_sir_{scale}_")[1]
        # e.g. "perm_smc_sigma_0p050_seed_42"
        sigma_idx = parts.index("_sigma_")
        method_tag = parts[:sigma_idx]
        rest = parts[sigma_idx + len("_sigma_"):]
        sigma_str = rest.split("_seed_")[0]
        sigma_val = float(sigma_str.replace("p", "."))

        with open(fpath, "rb") as f:
            data = pickle.load(f)
        results[(method_tag, sigma_val)] = data
        style = METHOD_STYLES.get(method_tag, (method_tag, "gray", "-"))
        print(f"  Loaded {style[0]} sigma={sigma_val:.3f}: "
              f"{data.get('n_iterations', '?')} iters, N_sim={data.get('total_n_sim', '?'):,}")
    return results


def plot_r0_by_sigma(results, scale, seed):
    """One subplot per sigma, KDE of R0 for each method."""
    sigmas = sorted(set(s for _, s in results.keys()))
    n_sigma = len(sigmas)
    fig, axes = plt.subplots(1, max(n_sigma, 1), figsize=(5 * max(n_sigma, 1), 4.5), squeeze=False)
    axes = axes[0]

    for i, sigma in enumerate(sigmas):
        ax = axes[i]
        for (mtag, s), data in results.items():
            if s != sigma:
                continue
            style = METHOD_STYLES.get(mtag, (mtag, "gray", "-"))
            r0 = np.asarray(data['Thetas_final'].glob[:, 0])
            sns.kdeplot(r0, ax=ax, color=style[1], linestyle=style[2],
                        label=style[0], linewidth=2)
        ax.set_title(rf"$\sigma = {sigma:.3f}$")
        ax.set_xlabel(r"$R_0$")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.suptitle(rf"Posterior $R_0$ — {scale} (seed {seed})", fontsize=13)
    fig.tight_layout()
    return fig


def plot_gamma_by_sigma(results, scale, seed):
    """One subplot per sigma, KDE of gamma (all K components overlaid) per method."""
    sigmas = sorted(set(s for _, s in results.keys()))
    n_sigma = len(sigmas)
    fig, axes = plt.subplots(1, max(n_sigma, 1), figsize=(5 * max(n_sigma, 1), 4.5), squeeze=False)
    axes = axes[0]

    for i, sigma in enumerate(sigmas):
        ax = axes[i]
        for (mtag, s), data in results.items():
            if s != sigma:
                continue
            style = METHOD_STYLES.get(mtag, (mtag, "gray", "-"))
            gamma = np.asarray(data['Thetas_final'].loc[:, :, 2])
            K = gamma.shape[1]
            # Plot mean across components
            gamma_flat = gamma.flatten()
            sns.kdeplot(gamma_flat, ax=ax, color=style[1], linestyle=style[2],
                        label=style[0], linewidth=2)
        ax.set_title(rf"$\sigma = {sigma:.3f}$")
        ax.set_xlabel(r"$\gamma$")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.suptitle(rf"Posterior $\gamma$ (pooled) — {scale} (seed {seed})", fontsize=13)
    fig.tight_layout()
    return fig


def plot_summary_table(results, scale, seed):
    """Print a summary table of all runs."""
    print(f"\n{'Method':<30s} {'sigma':>6s} {'Iters':>6s} {'N_sim':>12s} {'eps_final':>12s} {'Time(s)':>8s}")
    print("-" * 80)
    for (mtag, sigma), data in sorted(results.items(), key=lambda x: (x[0][1], x[0][0])):
        style = METHOD_STYLES.get(mtag, (mtag, "gray", "-"))
        print(f"{style[0]:<30s} {sigma:>6.3f} {data.get('n_iterations', '?'):>6} "
              f"{data.get('total_n_sim', 0):>12,} {data.get('final_epsilon', float('inf')):>12.1f} "
              f"{data.get('time_final', 0):>8.1f}")


def main():
    parser = argparse.ArgumentParser(description="SIR sensitivity analysis figure")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', type=str, default='departmental',
                        choices=['national', 'regional', 'departmental'])
    args = parser.parse_args()

    print(f"SIR sensitivity analysis ({args.scale}, seed={args.seed})")
    results = discover_results(args.scale, args.seed)

    if not results:
        print("\nNo results found. Run run_sir_real_world.py with --sigma first.")
        sys.exit(1)

    plot_summary_table(results, args.scale, args.seed)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_r0 = plot_r0_by_sigma(results, args.scale, args.seed)
    r0_path = FIGURES_DIR / f"sir_sensitivity_R0_{args.scale}_seed_{args.seed}.pdf"
    save_figure(fig_r0, r0_path)
    print(f"Saved: {r0_path}")

    fig_gamma = plot_gamma_by_sigma(results, args.scale, args.seed)
    gamma_path = FIGURES_DIR / f"sir_sensitivity_gamma_{args.scale}_seed_{args.seed}.pdf"
    save_figure(fig_gamma, gamma_path)
    print(f"Saved: {gamma_path}")


if __name__ == "__main__":
    main()
