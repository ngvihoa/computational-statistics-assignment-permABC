#!/usr/bin/env python3
"""
Figure 3: Posterior comparison between permABC and vanilla ABC.

Scatter plots showing how permABC resolves label-switching on a 2D uniform model.

Usage:
    python fig3_posterior_comparison_perm_vanilla.py
    python fig3_posterior_comparison_perm_vanilla.py --seed 42 --nsim 500000
    python fig3_posterior_comparison_perm_vanilla.py --rerun path/to/results.pkl
    python fig3_posterior_comparison_perm_vanilla.py --force
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from jax import random

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from permabc.assignment.dispatch import optimal_index_distance
from permabc.models.uniform_known import Uniform_known


# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "posterior_comparison"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "fig3"


def _pkl_path(seed):
    return RESULTS_DIR / f"fig3_uniform_seed_{seed}.pkl"


def _fig_path(seed):
    return FIGURES_DIR / f"fig3_uniform_seed_{seed}.pdf"


# ── Experiment ───────────────────────────────────────────────────────────────

def setup_experiment(seed=42):
    model = Uniform_known(K=2)
    y_obs = np.array([0., 0.5])[None, :, None]
    key = random.PRNGKey(seed)
    epsilon_star = model.distance(y_obs, y_obs[:, ::-1])[0] / 2
    print(f"Epsilon star: {epsilon_star}")
    return model, y_obs, key, epsilon_star


def run_abc_comparison(model, y_obs, key, Nsim=1000000):
    print("Running ABC and permABC simulations...")
    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, Nsim)
    key, subkey = random.split(key)
    zs = model.data_generator(subkey, thetas)

    dists = model.distance(zs, y_obs)
    dists_perm, _, zs_index, _ = optimal_index_distance(
        zs=zs, y_obs=y_obs, model=model, epsilon=0, verbose=2
    )
    thetas_perm = thetas.apply_permutation(zs_index)
    return thetas, thetas_perm, dists, dists_perm


def analyze_performance(dists, dists_perm, epsilon_star):
    results = {
        'epsilon_levels': [],
        'abc_counts': [],
        'perm_counts': [],
        'improvement_ratios': [],
    }
    epsilons = [np.inf, epsilon_star + 1, epsilon_star, epsilon_star * 0.5]
    for eps in epsilons:
        abc_count = np.sum(dists <= eps)
        perm_count = np.sum(dists_perm <= eps)
        ratio = perm_count / abc_count if abc_count > 0 else (np.inf if perm_count > 0 else 1.0)
        results['epsilon_levels'].append(eps)
        results['abc_counts'].append(abc_count)
        results['perm_counts'].append(perm_count)
        results['improvement_ratios'].append(ratio)
        if eps == np.inf:
            print(f"All particles: ABC={abc_count}, permABC={perm_count}")
        else:
            print(f"eps={eps:.3f}: ABC={abc_count}, permABC={perm_count} (ratio: {ratio:.2f})")

    print(f"\nDistance stats:")
    print(f"  ABC    — mean: {np.mean(dists):.4f}, median: {np.median(dists):.4f}")
    print(f"  permABC — mean: {np.mean(dists_perm):.4f}, median: {np.median(dists_perm):.4f}")
    print(f"  Improvement ratio: {np.mean(dists) / np.mean(dists_perm):.2f}")
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def create_comparison_plot(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star):
    epsilons = [np.inf, epsilon_star + 1, epsilon_star]
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))

    N_plot = 10000
    s = 1
    alpha_fig = 0.25
    y = y_obs[0, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i, epsilon in enumerate(epsilons):
        abc_idx = np.where(dists <= epsilon)[0]
        perm_idx = np.where(dists_perm <= epsilon)[0]
        if len(abc_idx) > N_plot:
            abc_idx = np.random.choice(abc_idx, N_plot, replace=False)
        if len(perm_idx) > N_plot:
            perm_idx = np.random.choice(perm_idx, N_plot, replace=False)

        mus_abc = thetas.loc[abc_idx].squeeze()
        mus_perm = thetas_perm.loc[perm_idx].squeeze()

        if len(mus_abc) > 0:
            axes[0].scatter(mus_abc[:, 0], mus_abc[:, 1],
                            color=colors[i], alpha=alpha_fig, s=10)
        if len(mus_perm) > 0:
            axes[1].scatter(mus_perm[:, 0], mus_perm[:, 1],
                            color=colors[i], alpha=alpha_fig, s=10)

    prior_region = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]])
    posterior_region = np.array([
        [y[0] - s, y[1] - s], [y[0] + s, y[1] - s],
        [y[0] + s, y[1] + s], [y[0] - s, y[1] + s],
        [y[0] - s, y[1] - s],
    ])

    for ax, title in zip(axes, ["ABC", "permABC"]):
        ax.scatter(y[0], y[1], c='black', marker='x', s=100, label='y')
        ax.plot(posterior_region[:, 0], posterior_region[:, 1],
                color='black', linestyle='--', linewidth=2, label='True posterior')
        ax.plot(prior_region[:, 0], prior_region[:, 1],
                color='grey', linestyle='--', linewidth=1, label='Prior')
        ax.set_title(title)
        ax.set_xlabel(r'$\mu_1$')
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.grid(False)

    axes[0].set_ylabel(r'$\mu_2$')
    axes[0].legend(loc='upper right', fontsize=10)

    fig.tight_layout()
    return fig


# ── Save / Load ──────────────────────────────────────────────────────────────

def save_results(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star,
                 performance_results, seed):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_data = {
        'seed': seed,
        'epsilon_star': epsilon_star,
        'n_particles': len(dists),
        'performance_results': performance_results,
        'thetas': thetas,
        'thetas_perm': thetas_perm,
        'dists': dists,
        'dists_perm': dists_perm,
        'y_obs': y_obs,
    }
    pkl = _pkl_path(seed)
    with open(pkl, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Results saved to: {pkl}")

    fig = create_comparison_plot(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star)
    fig_path = _fig_path(seed)
    save_figure(fig, fig_path)
    print(f"Figure saved to: {fig_path}")


def replot_from_pickle(pkl_path):
    print(f"Loading results from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    seed = data['seed']
    print(f"  seed={seed}, epsilon_star={data['epsilon_star']:.4f}")

    fig = create_comparison_plot(
        data['thetas'], data['thetas_perm'],
        data['dists'], data['dists_perm'],
        data['y_obs'], data['epsilon_star'],
    )
    fig_path = _fig_path(seed)
    save_figure(fig, fig_path)
    print(f"Figure saved to: {fig_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Figure 3: Posterior comparison between ABC and permABC"
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nsim', type=int, default=1000000)
    parser.add_argument('--rerun', type=str, default=None,
                        help='Path to existing .pkl — replot without re-simulation')
    parser.add_argument('--force', action='store_true',
                        help='Force re-simulation even if cached results exist')
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Figure 3: Posterior comparison between ABC and permABC")
    print(f"Parameters: seed={args.seed}, nsim={args.nsim:,}")

    if args.rerun:
        replot_from_pickle(args.rerun)
        return

    pkl = _pkl_path(args.seed)
    if pkl.exists() and not args.force:
        print(f"Found cached results: {pkl}")
        replot_from_pickle(pkl)
        return

    model, y_obs, key, epsilon_star = setup_experiment(seed=args.seed)
    thetas, thetas_perm, dists, dists_perm = run_abc_comparison(
        model, y_obs, key, Nsim=args.nsim
    )
    performance_results = analyze_performance(dists, dists_perm, epsilon_star)
    save_results(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star,
                 performance_results, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
