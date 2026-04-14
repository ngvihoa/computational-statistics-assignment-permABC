#!/usr/bin/env python3
"""
Exploratory script: verify that all posterior quality metrics decrease
as the ABC threshold epsilon decreases, using vanilla rejection ABC.

Model: GaussianWithNoSummaryStats (K components, shared sigma², independent mu_k).

Usage:
    python explore_metrics_vs_epsilon.py
    python explore_metrics_vs_epsilon.py --K 5 --seed 42 --Nsim 200000
"""

import sys
import argparse
import time
import pickle
from pathlib import Path

import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.assignment.dispatch import optimal_index_distance
from permabc.utils.functions import Theta

from diagnostics import (
    build_sigma2_reference_bins,
    empirical_kl_sigma2,
    empirical_kl_mu_avg,
    empirical_w2_sigma2,
    empirical_w2_mu_avg,
    empirical_kl_joint,
    expected_neg_log_joint_true,
    sliced_w2_joint,
)

import matplotlib.pyplot as plt


# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "explore_metrics"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "explore_metrics"


# ── Experiment ───────────────────────────────────────────────────────────────

def setup_experiment(K=5, seed=42):
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    n_obs = 10
    sigma0 = 3
    alpha, beta = 2., 2.

    model = GaussianWithNoSummaryStats(
        K=K, n_obs=n_obs, sigma_0=sigma0, alpha=alpha, beta=beta
    )
    true_theta = model.prior_generator(subkey, 1)
    # Fix true params for reproducibility
    glob = np.array(true_theta.glob)
    loc = np.array(true_theta.loc)
    glob[0, 0] = 1.0
    loc[0, 0, 0] = 0.0
    true_theta = Theta(glob=glob, loc=loc)

    key, subkey = random.split(key)
    y_obs = model.data_generator(subkey, true_theta)
    return model, y_obs, true_theta, key


def run_vanilla_abc(key, model, y_obs, Nsim=200000):
    """Generate prior samples + simulations, compute vanilla distances."""
    print(f"Generating {Nsim:,} prior samples...")
    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, Nsim)
    key, subkey = random.split(key)
    zs = model.data_generator(subkey, thetas)

    print("Computing vanilla ABC distances...")
    dists = model.distance(zs, y_obs)

    print("Computing permABC distances...")
    dists_perm, _, zs_index, _ = optimal_index_distance(
        model=model, zs=zs, y_obs=y_obs, epsilon=0, verbose=2
    )
    thetas_perm = thetas.apply_permutation(zs_index)

    return thetas, thetas_perm, dists, dists_perm


def compute_all_metrics(model, y_obs, thetas, weights=None, perm=None,
                        sigma2_edges=None, label=""):
    """Compute all available metrics for a set of ABC particles."""
    metrics = {}

    # Marginal sigma²
    metrics['kl_sigma2'] = empirical_kl_sigma2(
        model, y_obs, thetas, weights=weights, edges=sigma2_edges, direction="q_vs_p"
    )
    metrics['w2_sigma2'] = empirical_w2_sigma2(model, y_obs, thetas, weights=weights)

    # Marginal mu (averaged over K)
    metrics['kl_mu_avg'] = empirical_kl_mu_avg(
        model, y_obs, thetas, weights=weights, perm=perm
    )
    metrics['w2_mu_avg'] = empirical_w2_mu_avg(
        model, y_obs, thetas, weights=weights, perm=perm
    )

    # Joint
    metrics['kl_joint_q_vs_p'] = empirical_kl_joint(
        model, y_obs, thetas, weights=weights, perm=perm, direction="q_vs_p"
    )
    metrics['kl_joint_p_vs_q'] = empirical_kl_joint(
        model, y_obs, thetas, weights=weights, perm=perm, direction="p_vs_q"
    )
    metrics['score_joint'] = expected_neg_log_joint_true(
        model, y_obs, thetas, weights=weights, perm=perm
    )
    metrics['sw2_joint'] = sliced_w2_joint(
        model, y_obs, thetas, weights=weights, perm=perm
    )

    if label:
        print(f"  [{label}]")
    for k, v in metrics.items():
        print(f"    {k:20s} = {v:.6f}")

    return metrics


def run_metrics_vs_epsilon(model, y_obs, thetas, thetas_perm, dists, dists_perm,
                           n_quantiles=10):
    """Compute metrics for both vanilla and permABC at various epsilon quantiles."""
    # Fixed bins for consistent KL comparison
    sigma2_edges = build_sigma2_reference_bins(model, y_obs)

    # Epsilon quantiles (from generous to strict)
    quantiles = np.linspace(0.5, 0.01, n_quantiles)
    epsilons_vanilla = np.quantile(dists, quantiles)
    epsilons_perm = np.quantile(dists_perm, quantiles)

    results = {
        'quantiles': quantiles,
        'epsilons_vanilla': epsilons_vanilla,
        'epsilons_perm': epsilons_perm,
        'n_accepted_vanilla': [],
        'n_accepted_perm': [],
        'metrics_vanilla': [],
        'metrics_perm': [],
    }

    for i, q in enumerate(quantiles):
        eps_v = epsilons_vanilla[i]
        eps_p = epsilons_perm[i]

        # Vanilla ABC
        mask_v = dists <= eps_v
        n_v = int(np.sum(mask_v))
        thetas_v = thetas[mask_v]

        # permABC
        mask_p = dists_perm <= eps_p
        n_p = int(np.sum(mask_p))
        thetas_p = thetas_perm[mask_p]

        results['n_accepted_vanilla'].append(n_v)
        results['n_accepted_perm'].append(n_p)

        print(f"\n--- Quantile {q:.2f} (vanilla eps={eps_v:.4f}, perm eps={eps_p:.4f}) ---")
        print(f"    Accepted: vanilla={n_v}, perm={n_p}")

        m_v = compute_all_metrics(
            model, y_obs, thetas_v, sigma2_edges=sigma2_edges, label="vanilla"
        )
        m_p = compute_all_metrics(
            model, y_obs, thetas_p, sigma2_edges=sigma2_edges, label="permABC"
        )

        results['metrics_vanilla'].append(m_v)
        results['metrics_perm'].append(m_p)

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    'kl_sigma2': r'KL$(\hat{q} \| p)$ on $\sigma^2$',
    'w2_sigma2': r'$W_2$ on $\sigma^2$',
    'kl_mu_avg': r'KL$(\hat{q} \| p)$ on $\mu_k$ (avg)',
    'w2_mu_avg': r'$W_2$ on $\mu_k$ (avg)',
    'kl_joint_q_vs_p': r'KL$(\hat{q} \| p)$ joint',
    'kl_joint_p_vs_q': r'KL$(p \| \hat{q})$ joint',
    'score_joint': r'$-\mathbb{E}_q[\log p^*]$ joint',
    'sw2_joint': r'Sliced $W_2$ joint',
}


def create_metrics_plot(results):
    metric_names = list(METRIC_LABELS.keys())
    n_metrics = len(metric_names)
    ncols = 2
    nrows = (n_metrics + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    quantiles = results['quantiles']

    for i, name in enumerate(metric_names):
        ax = axes[i]
        vals_v = [m[name] for m in results['metrics_vanilla']]
        vals_p = [m[name] for m in results['metrics_perm']]

        ax.plot(quantiles, vals_v, 'o-', color='#d62728', label='vanilla ABC', markersize=4)
        ax.plot(quantiles, vals_p, 's-', color='#1f77b4', label='permABC', markersize=4)
        ax.set_xlabel('Acceptance quantile')
        ax.set_ylabel(METRIC_LABELS[name])
        ax.set_title(METRIC_LABELS[name])
        ax.invert_xaxis()  # stricter epsilon to the right
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore metrics vs epsilon for vanilla ABC and permABC"
    )
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--Nsim', type=int, default=200000)
    parser.add_argument('--n-quantiles', type=int, default=10)
    args = parser.parse_args()

    print(f"Explore metrics vs epsilon: K={args.K}, seed={args.seed}, Nsim={args.Nsim:,}")

    model, y_obs, true_theta, key = setup_experiment(K=args.K, seed=args.seed)
    thetas, thetas_perm, dists, dists_perm = run_vanilla_abc(
        key, model, y_obs, Nsim=args.Nsim
    )

    results = run_metrics_vs_epsilon(
        model, y_obs, thetas, thetas_perm, dists, dists_perm,
        n_quantiles=args.n_quantiles,
    )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = RESULTS_DIR / f"explore_metrics_K_{args.K}_seed_{args.seed}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            'results': results,
            'K': args.K,
            'seed': args.seed,
            'Nsim': args.Nsim,
        }, f)
    print(f"\nResults saved to: {pkl_path}")

    fig = create_metrics_plot(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"explore_metrics_K_{args.K}_seed_{args.seed}.pdf"
    save_figure(fig, fig_path)
    print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
