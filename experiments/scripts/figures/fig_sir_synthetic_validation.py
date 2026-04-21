#!/usr/bin/env python3
"""
Figure: SIR synthetic data validation.

Produces a multi-panel figure for the supplement showing:
  (a) Posterior of R0 with true value marked (one panel per sigma)
  (b) Posteriors of selected local gamma_k with true values
  (c) Posterior predictive check: simulated trajectories vs synthetic data

Usage:
    python fig_sir_synthetic_validation.py --seed 42
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_result(results_dir, K, sigma, seed):
    sigma_str = f"{sigma:.3f}".replace('.', 'p')
    fname = os.path.join(results_dir, f"sir_synthetic_K{K}_sigma_{sigma_str}_seed_{seed}.pkl")
    if not os.path.exists(fname):
        print(f"  WARNING: {fname} not found, skipping sigma={sigma}")
        return None
    with open(fname, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--sigmas', nargs='+', type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument('--results-dir', type=str,
                        default='experiments/experiments/results/sir_synthetic')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/figures/fig_sir_synthetic')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    results = {}
    for sigma in args.sigmas:
        r = load_result(args.results_dir, args.K, sigma, args.seed)
        if r is not None:
            results[sigma] = r

    if not results:
        print("No results found. Run run_sir_synthetic_validation.py first.")
        sys.exit(1)

    n_sigma = len(results)
    sigma_list = sorted(results.keys())

    # --- Figure layout ---
    # Row 1: R0 posteriors (one per sigma)
    # Row 2: gamma_k posteriors for k=1..4 (sigma=0.05 only or middle sigma)
    # Row 3: Posterior predictive check (sigma=0.05 only or middle sigma)
    fig, axes = plt.subplots(3, max(n_sigma, 4), figsize=(4 * max(n_sigma, 4), 10))

    # Ensure axes is 2D
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    # ── Row 1: R0 posterior for each sigma ──
    for i, sigma in enumerate(sigma_list):
        ax = axes[0, i]
        data = results[sigma]
        true_R0 = float(data['true_theta'].glob[0, 0])
        R0_samples = np.array(data['Thetas_final'].glob[:, 0])
        weights = np.array(data['Weights_final'])
        weights = weights / weights.sum()

        ax.hist(R0_samples, bins=50, weights=weights, density=True,
                alpha=0.7, color='steelblue', edgecolor='white', linewidth=0.3)
        ax.axvline(true_R0, color='red', linewidth=2, linestyle='--', label=f'True $R_0$={true_R0:.2f}')
        ax.set_title(f'$\\sigma={sigma}$', fontsize=12)
        ax.set_xlabel('$R_0$')
        if i == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    # Hide unused axes in row 1
    for i in range(n_sigma, axes.shape[1]):
        axes[0, i].set_visible(False)

    # ── Row 2: gamma_k posteriors (use middle sigma) ──
    ref_sigma = 0.05 if 0.05 in results else sigma_list[len(sigma_list) // 2]
    data = results[ref_sigma]
    true_gamma = np.array(data['true_theta'].loc[0, :, 2])
    gamma_samples = np.array(data['Thetas_final'].loc[:, :, 2])
    weights = np.array(data['Weights_final'])
    weights = weights / weights.sum()
    n_show = min(4, data['K'])

    for k in range(n_show):
        ax = axes[1, k]
        ax.hist(gamma_samples[:, k], bins=50, weights=weights, density=True,
                alpha=0.7, color='darkorange', edgecolor='white', linewidth=0.3)
        ax.axvline(true_gamma[k], color='red', linewidth=2, linestyle='--',
                   label=f'True $\\gamma_{{{k+1}}}$={true_gamma[k]:.3f}')
        ax.set_title(f'$\\gamma_{{{k+1}}}$ ($\\sigma={ref_sigma}$)', fontsize=11)
        ax.set_xlabel(f'$\\gamma_{{{k+1}}}$')
        if k == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    for k in range(n_show, axes.shape[1]):
        axes[1, k].set_visible(False)

    # ── Row 3: Posterior predictive check (use middle sigma) ──
    y_obs = np.array(data['y_obs']).squeeze()  # (K, n_obs)
    y_pred = np.array(data['y_pred'])           # (n_pp, K, n_obs)
    n_obs = y_obs.shape[-1]
    days = np.arange(n_obs)
    n_show_k = min(4, data['K'])

    for k in range(n_show_k):
        ax = axes[2, k]

        # Posterior predictive envelope
        pp_k = y_pred[:, k, :]  # (n_pp, n_obs)
        q05 = np.percentile(pp_k, 5, axis=0)
        q25 = np.percentile(pp_k, 25, axis=0)
        q75 = np.percentile(pp_k, 75, axis=0)
        q95 = np.percentile(pp_k, 95, axis=0)
        median = np.median(pp_k, axis=0)

        ax.fill_between(days, q05, q95, alpha=0.15, color='steelblue')
        ax.fill_between(days, q25, q75, alpha=0.3, color='steelblue')
        ax.plot(days, median, color='steelblue', linewidth=1, label='Posterior pred.')

        # Observed synthetic data
        ax.plot(days, y_obs[k], color='black', linewidth=1.5, linestyle='-', label='Observed')

        ax.set_title(f'Dept. {k+1} ($\\sigma={ref_sigma}$)', fontsize=11)
        ax.set_xlabel('Day')
        if k == 0:
            ax.set_ylabel('Infected (per 100k)')
        if k == 0:
            ax.legend(fontsize=8)

    for k in range(n_show_k, axes.shape[1]):
        axes[2, k].set_visible(False)

    fig.tight_layout(h_pad=2.5)
    outpath = os.path.join(args.output_dir,
                           f"fig_sir_synthetic_validation_seed_{args.seed}.pdf")
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    print(f"Saved: {outpath}")

    # Also save to paper/fig/ for LaTeX inclusion
    paper_fig = os.path.join('paper', 'fig',
                             f"fig_sir_synthetic_validation_seed_{args.seed}.pdf")
    os.makedirs(os.path.dirname(paper_fig), exist_ok=True)
    fig.savefig(paper_fig, bbox_inches='tight', dpi=150)
    print(f"Saved: {paper_fig}")

    plt.close(fig)


if __name__ == '__main__':
    main()
