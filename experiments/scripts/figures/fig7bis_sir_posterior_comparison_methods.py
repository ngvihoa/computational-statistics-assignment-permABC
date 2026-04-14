#!/usr/bin/env python3
"""
Figure 7bis: Posterior comparison across methods on SIR real-world data.

KDE plots comparing R0 and gamma (nu) parameters for permABC-SMC, ABC-Gibbs,
permABC-SMC-OS, and permABC-SMC-UM on the same COVID-19 dataset.

Usage:
    python fig7bis_sir_posterior_comparison_methods.py
    python fig7bis_sir_posterior_comparison_methods.py --seed 42 --scale regional
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()

# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "sir_real_world_inference"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "fig7bis"


def _fig_path(scale, seed):
    return FIGURES_DIR / f"fig7bis_sir_methods_{scale}_seed_{seed}.pdf"


# ── Method definitions ───────────────────────────────────────────────────────

METHODS = [
    ("permABC-SMC",              "perm_smc",         "#1f77b4"),
    ("ABC-SMC (Gibbs 3b)",       "abc_smc_g3",       "#d62728"),
    ("permABC-SMC (Gibbs 3b)",   "perm_smc_g3",      "#2ca02c"),
    ("ABC-Gibbs",                "abc_gibbs_true",    "#ff7f0e"),
    ("ABC-SMC",                  "abc_smc",           "#8c564b"),
    ("permABC-SMC-OS",           "perm_smc_os",       "#e377c2"),
    ("permABC-SMC-UM",           "perm_smc_um",       "#9467bd"),
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_results(seed, scale):
    loaded = {}
    for display_name, file_tag, color in METHODS:
        path = RESULTS_DIR / f"inference_sir_{scale}_{file_tag}_seed_{seed}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            loaded[display_name] = data
            n_part = data['Thetas_final'].glob.shape[0] if hasattr(data['Thetas_final'], 'glob') else '?'
            print(f"  Loaded {display_name}: {data.get('n_iterations', '?')} iters, "
                  f"N_sim={data.get('total_n_sim', '?'):,}, "
                  f"n_particles={n_part}")
        else:
            print(f"  Not found: {display_name} ({path.name})")
    return loaded


# ── Plotting ─────────────────────────────────────────────────────────────────

def create_posterior_comparison(all_data, scale):
    method_colors = {name: color for name, _, color in METHODS}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── R0 (global parameter, index 0) ──────────────────────────────────
    for method_name, data in all_data.items():
        r0_samples = np.asarray(data['Thetas_final'].glob[:, 0])
        color = method_colors.get(method_name, 'gray')
        sns.kdeplot(r0_samples, ax=axes[0], color=color,
                    label=method_name, linewidth=2)

    axes[0].set_xlabel(r"$R_0$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"Distribution of $R_0$")
    axes[0].legend(fontsize=9)

    # ── Gamma / nu (local parameter, index 2) ───────────────────────────
    for method_name, data in all_data.items():
        gamma_samples = np.asarray(data['Thetas_final'].loc[:, :, 2])
        color = method_colors.get(method_name, 'gray')
        K = gamma_samples.shape[1]
        for k in range(K):
            alpha = max(0.15, 1.0 / K)
            label = method_name if k == 0 else None
            sns.kdeplot(gamma_samples[:, k], ax=axes[1],
                        color=color, alpha=alpha, label=label)

    axes[1].set_xlabel(r"$\nu$")
    axes[1].set_ylabel("Density")
    axes[1].set_title(r"Distribution of $\nu$ (all components)")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"SIR posterior comparison — {scale} scale", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Figure 7bis: SIR posterior comparison across methods"
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', type=str, default='regional',
                        choices=['national', 'regional', 'departmental'])
    args = parser.parse_args()

    print(f"Figure 7bis: SIR posterior comparison ({args.scale}, seed={args.seed})")

    all_data = load_all_results(args.seed, args.scale)
    if not all_data:
        print("\nERROR: No inference results found. Run run_sir_real_world.py first.")
        sys.exit(1)

    fig = create_posterior_comparison(all_data, args.scale)
    fig_path = _fig_path(args.scale, args.seed)
    save_figure(fig, fig_path)
    print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
