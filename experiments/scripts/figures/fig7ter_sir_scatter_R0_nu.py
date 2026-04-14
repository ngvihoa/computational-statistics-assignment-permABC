#!/usr/bin/env python3
"""
Figure 7ter: Scatter R0 vs nu_k for SIR real-world posterior comparison.

Shows joint exploration quality: ABC-Gibbs with poor mixing will show
correlated streaks or clumps, while SMC methods should fill the space.

For each method, plots R0 vs gamma_k for a few representative regions k,
plus a marginal R0 vs mean(gamma) panel.

Usage:
    python fig7ter_sir_scatter_R0_nu.py
    python fig7ter_sir_scatter_R0_nu.py --seed 42 --scale regional --n_regions 4
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "sir_real_world_inference"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "fig7ter"

# ── Method definitions (must match run_sir_real_world.py tags) ──────────────

METHODS = [
    ("permABC-SMC",              "perm_smc",       "#1f77b4"),
    ("ABC-SMC (Gibbs 3b)",       "abc_smc_g3",     "#d62728"),
    ("permABC-SMC (Gibbs 3b)",   "perm_smc_g3",    "#2ca02c"),
    ("ABC-Gibbs",                "abc_gibbs_true",  "#ff7f0e"),
    ("permABC-SMC-OS",           "perm_smc_os",     "#e377c2"),
    ("permABC-SMC-UM",           "perm_smc_um",     "#9467bd"),
]


def load_all_results(seed, scale):
    loaded = {}
    for display_name, file_tag, color in METHODS:
        path = RESULTS_DIR / f"inference_sir_{scale}_{file_tag}_seed_{seed}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            loaded[display_name] = data
            print(f"  Loaded {display_name}")
        else:
            print(f"  Not found: {display_name} ({path.name})")
    return loaded


def pick_representative_regions(all_data, n_regions):
    """Pick n_regions spread across the K regions (first, middle, last-ish)."""
    for data in all_data.values():
        K = np.asarray(data['Thetas_final'].loc).shape[1]
        break
    if K <= n_regions:
        return list(range(K))
    indices = np.linspace(0, K - 1, n_regions, dtype=int)
    return list(indices)


def create_scatter_grid(all_data, scale, region_indices):
    """One row per method, columns = selected regions + mean(gamma)."""
    method_colors = {name: color for name, _, color in METHODS}
    methods_present = [m for m in all_data]
    n_methods = len(methods_present)
    n_cols = len(region_indices) + 1  # +1 for R0 vs mean(gamma)

    fig, axes = plt.subplots(n_methods, n_cols,
                             figsize=(4 * n_cols, 3.2 * n_methods),
                             squeeze=False)

    for row, method_name in enumerate(methods_present):
        data = all_data[method_name]
        r0 = np.asarray(data['Thetas_final'].glob[:, 0])
        gamma_all = np.asarray(data['Thetas_final'].loc[:, :, 2])  # (N, K)
        color = method_colors.get(method_name, 'gray')
        n_pts = len(r0)
        alpha = max(0.05, min(0.5, 200.0 / n_pts))
        s = max(2, min(10, 500.0 / n_pts))

        # Scatter R0 vs gamma_k for selected regions
        for col, k in enumerate(region_indices):
            ax = axes[row, col]
            ax.scatter(r0, gamma_all[:, k], c=color, alpha=alpha, s=s,
                       edgecolors='none', rasterized=True)
            if row == 0:
                ax.set_title(f"Region {k}", fontsize=10)
            if col == 0:
                ax.set_ylabel(method_name, fontsize=9)
            if row == n_methods - 1:
                ax.set_xlabel(r"$R_0$")

        # Last column: R0 vs mean(gamma)
        ax = axes[row, -1]
        gamma_mean = np.mean(gamma_all, axis=1)
        ax.scatter(r0, gamma_mean, c=color, alpha=alpha, s=s,
                   edgecolors='none', rasterized=True)
        if row == 0:
            ax.set_title(r"$\bar{\nu}$ (mean)", fontsize=10)
        if row == n_methods - 1:
            ax.set_xlabel(r"$R_0$")

    fig.suptitle(f"R0 vs nu — {scale} scale  ({n_pts} particles)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def create_overlay_scatter(all_data, scale, region_indices):
    """All methods overlaid on same axes — one panel per region + mean."""
    method_colors = {name: color for name, _, color in METHODS}
    n_cols = len(region_indices) + 1

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5),
                             squeeze=False)

    for method_name, data in all_data.items():
        r0 = np.asarray(data['Thetas_final'].glob[:, 0])
        gamma_all = np.asarray(data['Thetas_final'].loc[:, :, 2])
        color = method_colors.get(method_name, 'gray')
        n_pts = len(r0)
        alpha = max(0.1, min(0.4, 200.0 / n_pts))
        s = max(3, min(12, 500.0 / n_pts))

        for col, k in enumerate(region_indices):
            axes[0, col].scatter(r0, gamma_all[:, k], c=color, alpha=alpha,
                                 s=s, label=method_name if col == 0 else None,
                                 edgecolors='none', rasterized=True)
            axes[0, col].set_title(f"Region {k}", fontsize=10)
            axes[0, col].set_xlabel(r"$R_0$")
            if col == 0:
                axes[0, col].set_ylabel(r"$\nu_k$")

        gamma_mean = np.mean(gamma_all, axis=1)
        axes[0, -1].scatter(r0, gamma_mean, c=color, alpha=alpha, s=s,
                            edgecolors='none', rasterized=True)

    axes[0, -1].set_title(r"$\bar{\nu}$ (mean)", fontsize=10)
    axes[0, -1].set_xlabel(r"$R_0$")
    axes[0, 0].legend(fontsize=8, markerscale=3, loc='upper right')

    fig.suptitle(f"R0 vs nu overlay — {scale} scale",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def create_chain_trace(all_data, scale):
    """For ABC-Gibbs: trace plot of R0 and mean(gamma) to show mixing."""
    gibbs_methods = [m for m in all_data if "Gibbs" in m and "SMC" not in m]
    if not gibbs_methods:
        return None

    n_gibbs = len(gibbs_methods)
    fig, axes = plt.subplots(n_gibbs, 2, figsize=(12, 3.5 * n_gibbs),
                             squeeze=False)

    for row, method_name in enumerate(gibbs_methods):
        data = all_data[method_name]
        r0 = np.asarray(data['Thetas_final'].glob[:, 0])
        gamma_mean = np.mean(np.asarray(data['Thetas_final'].loc[:, :, 2]), axis=1)

        axes[row, 0].plot(r0, lw=0.5, color='#ff7f0e')
        axes[row, 0].set_ylabel(r"$R_0$")
        axes[row, 0].set_title(f"{method_name} — R0 trace")

        axes[row, 1].plot(gamma_mean, lw=0.5, color='#ff7f0e')
        axes[row, 1].set_ylabel(r"$\bar{\nu}$")
        axes[row, 1].set_title(f"{method_name} — mean(nu) trace")

    for ax in axes[-1]:
        ax.set_xlabel("Iteration (post burn-in)")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Figure 7ter: Scatter R0 vs nu for SIR methods"
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', type=str, default='regional',
                        choices=['national', 'regional', 'departmental'])
    parser.add_argument('--n_regions', type=int, default=4,
                        help='Number of representative regions to plot')
    args = parser.parse_args()

    print(f"Figure 7ter: R0 vs nu scatter ({args.scale}, seed={args.seed})")

    all_data = load_all_results(args.seed, args.scale)
    if not all_data:
        print("\nERROR: No inference results found.")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    region_indices = pick_representative_regions(all_data, args.n_regions)
    print(f"  Regions selected: {region_indices}")

    # Grid: one row per method
    fig_grid = create_scatter_grid(all_data, args.scale, region_indices)
    p = FIGURES_DIR / f"fig7ter_grid_{args.scale}_seed_{args.seed}.pdf"
    save_figure(fig_grid, p)
    print(f"  Saved: {p}")

    # Overlay: all methods on same axes
    fig_overlay = create_overlay_scatter(all_data, args.scale, region_indices)
    p = FIGURES_DIR / f"fig7ter_overlay_{args.scale}_seed_{args.seed}.pdf"
    save_figure(fig_overlay, p)
    print(f"  Saved: {p}")

    # Trace plot for Gibbs methods
    fig_trace = create_chain_trace(all_data, args.scale)
    if fig_trace:
        p = FIGURES_DIR / f"fig7ter_trace_{args.scale}_seed_{args.seed}.pdf"
        save_figure(fig_trace, p)
        print(f"  Saved: {p}")


if __name__ == "__main__":
    main()
