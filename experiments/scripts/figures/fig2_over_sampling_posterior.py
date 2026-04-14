#!/usr/bin/env python3
"""
Figure 2: Over-sampling posterior comparison.

Shows how the posterior changes as M (number of over-sampled components) increases
from K to 10K on a Gaussian model.

Usage:
    python fig2_over_sampling_posterior.py
    python fig2_over_sampling_posterior.py --K 10 --seed 42
    python fig2_over_sampling_posterior.py --rerun path/to/results.pkl
    python fig2_over_sampling_posterior.py --force   # Force new simulation
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
from scipy.stats import norm, invgamma

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from permabc.assignment.dispatch import optimal_index_distance
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta


# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "over_sampling"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "fig2"


def _pkl_path(K, seed):
    return RESULTS_DIR / f"fig2_over_sampling_K_{K}_seed_{seed}.pkl"


def _fig_path(K, seed):
    return FIGURES_DIR / f"fig2_over_sampling_K_{K}_seed_{seed}.pdf"


# ── Experiment ───────────────────────────────────────────────────────────────

def setup_experiment(K=10, seed=42):
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    n = 10
    sigma0 = 3
    alpha, beta = 1.01, 1.

    model = GaussianWithNoSummaryStats(K=K, n_obs=n, sigma_0=sigma0, alpha=alpha, beta=beta)
    true_theta = model.prior_generator(subkey, 1)

    glob = np.array(true_theta.glob)
    loc = np.array(true_theta.loc)
    glob[0, 0] = 1.
    loc[0, 0, 0] = 0.
    true_theta = Theta(glob=glob, loc=loc)

    print(f"True sigma2: {true_theta.glob[0, 0]}")

    key, subkey = random.split(key)
    y_obs = model.data_generator(subkey, true_theta)

    return model, y_obs, true_theta, key


def run_over_sampling_posterior(key, model, y_obs, K):
    print("Running over-sampling posterior analysis...")

    N_epsilon = 100000
    M0s = np.array([K, 1.5*K, 2*K, 5*K, 10*K, 25*K], dtype=int)
    alpha_epsilon = 0.05

    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, N_epsilon, np.max(M0s))
    key, subkey = random.split(key)
    zs = model.data_generator(subkey, thetas)

    dists_perm, _, _, _ = optimal_index_distance(model, zs[:, :K], y_obs, M=K)
    epsilon = np.quantile(dists_perm, alpha_epsilon)
    print(f"Epsilon threshold: {epsilon}")

    results = {
        'M0_values': [],
        'glob_posteriors': [],
        'loc_posteriors': [],
        'acceptance_rates': [],
    }

    for M0 in M0s:
        print(f"Processing M0 = {M0}")
        dists_perm, ys_index, zs_index, _ = optimal_index_distance(
            model, zs[:, :M0], y_obs, M=M0
        )
        thetas_perm = thetas.apply_permutation(zs_index)
        accepted = dists_perm < epsilon
        thetas_accepted = thetas_perm[accepted]

        acceptance_rate = np.sum(accepted) / N_epsilon
        print(f"  Acceptance rate: {acceptance_rate:.2%}")

        results['M0_values'].append(M0)
        results['glob_posteriors'].append(thetas_accepted.glob[:, 0])
        results['loc_posteriors'].append(thetas_accepted.loc[:, 0, 0])
        results['acceptance_rates'].append(acceptance_rate)

    return results, epsilon


# ── True conjugate posterior ──────────────────────────────────────────────────

def _log_marginal_likelihood_sigma2(sigma2_grid, y_obs, model):
    """Log p(y | sigma^2) after integrating out all mu_k analytically.

    Model: mu_k ~ N(0, sigma_0^2),  X_{k,i} | mu_k, sigma^2 ~ N(mu_k, sigma^2).
    Marginal per component: y_k ~ N(0, sigma^2 I + sigma_0^2 11^T).
    """
    K = model.K
    n = model.n_obs
    sigma_0_sq = model.sigma_0 ** 2
    y = np.asarray(y_obs[0])  # (K, n_obs)

    log_lik = np.zeros_like(sigma2_grid)
    for k in range(K):
        SS_k = np.sum(y[k] ** 2)
        S_k = np.sum(y[k])
        # det and quadratic form via Woodbury / matrix determinant lemma
        log_lik += (
            -n / 2 * np.log(sigma2_grid)
            - 0.5 * np.log(1 + n * sigma_0_sq / sigma2_grid)
            - 1 / (2 * sigma2_grid) * (
                SS_k - sigma_0_sq * S_k ** 2 / (sigma2_grid + n * sigma_0_sq)
            )
        )
    return log_lik


def true_posterior_sigma2(model, y_obs, sigma2_grid):
    """Marginal posterior density of sigma^2 on a grid (normalised)."""
    alpha, beta = model.alpha, model.beta
    log_prior = -(alpha + 1) * np.log(sigma2_grid) - beta / sigma2_grid
    log_post = log_prior + _log_marginal_likelihood_sigma2(sigma2_grid, y_obs, model)
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= np.trapezoid(post, sigma2_grid)
    return post


def true_posterior_mu1(model, y_obs, mu1_grid, n_sigma2=1000):
    """Marginal posterior density of mu_1 by integrating over sigma^2."""
    n = model.n_obs
    sigma_0_sq = model.sigma_0 ** 2
    y = np.asarray(y_obs[0])
    S_1 = np.sum(y[0])

    # sigma^2 grid — wide enough to cover the posterior mass
    s2_grid = np.linspace(0.01, 30, n_sigma2)
    post_s2 = true_posterior_sigma2(model, y_obs, s2_grid)
    ds2 = s2_grid[1] - s2_grid[0]

    post_mu1 = np.zeros_like(mu1_grid)
    for i, s2 in enumerate(s2_grid):
        v1 = 1.0 / (n / s2 + 1.0 / sigma_0_sq)
        m1 = v1 * S_1 / s2
        post_mu1 += post_s2[i] * norm.pdf(mu1_grid, loc=m1, scale=np.sqrt(v1))
    post_mu1 *= ds2
    # renormalise for numerical safety
    post_mu1 /= np.trapezoid(post_mu1, mu1_grid)
    return post_mu1


# ── Plotting ─────────────────────────────────────────────────────────────────

def create_posterior_plot(results, true_theta, model, y_obs=None):
    M0s = results['M0_values']
    colors = plt.cm.viridis(np.linspace(0, 1, len(M0s)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ── Global parameter (sigma^2) ───────────────────────────────────────
    for i, M0 in enumerate(M0s):
        glob_samples = results['glob_posteriors'][i]
        if len(glob_samples) > 0:
            sns.kdeplot(glob_samples, ax=axes[0], color=colors[i], label=f"M = {M0}")

    prior_glob = invgamma(a=model.alpha, scale=model.beta)
    a_glob, b_glob = prior_glob.ppf(0.00), prior_glob.ppf(0.9)
    x_glob = np.linspace(max(a_glob, 1e-3), b_glob, 1000)
    axes[0].plot(x_glob, prior_glob.pdf(x_glob),
                 linestyle="--", color="grey", label="Prior")

    # True conjugate posterior
    if y_obs is not None:
        post_s2 = true_posterior_sigma2(model, y_obs, x_glob)
        # axes[0].plot(x_glob, post_s2,
        #              linestyle="-", color="black", linewidth=2, label="True posterior")

    # axes[0].axvline(true_theta.glob[0, 0], color='red', linestyle='-',
    #                 linewidth=1.5, alpha=0.7, label=f"True: {true_theta.glob[0, 0]:.1f}")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Density")
    axes[0].set_xlabel(r"$\sigma^2$")
    axes[0].set_title("Global parameter")
    axes[0].set_xlim(0, b_glob)

    # ── Local parameter (mu_1) ───────────────────────────────────────────
    for i, M0 in enumerate(M0s):
        loc_samples = results['loc_posteriors'][i]
        if len(loc_samples) > 0:
            sns.kdeplot(loc_samples, ax=axes[1], color=colors[i], label=f"M = {M0}")

    prior_loc = norm(loc=0, scale=model.sigma_0)
    a_loc, b_loc = prior_loc.interval(0.999)
    x_loc = np.linspace(a_loc, b_loc, 1000)
    axes[1].plot(x_loc, prior_loc.pdf(x_loc),
                 linestyle="--", color="grey", label="Prior")

    # True conjugate posterior
    if y_obs is not None:
        post_mu1 = true_posterior_mu1(model, y_obs, x_loc)
        # axes[1].plot(x_loc, post_mu1,
                    #  linestyle="-", color="black", linewidth=2, label="True posterior")

    # axes[1].axvline(true_theta.loc[0, 0, 0], color='red', linestyle='-',
    #                 linewidth=1.5, alpha=0.7, label=f"True: {true_theta.loc[0, 0, 0]:.1f}")
    # axes[1].set_ylabel("")
    axes[1].set_xlabel(r"$\mu_1$")
    axes[1].set_title("Local parameter")
    axes[1].set_xlim(-10, 10)

    fig.tight_layout()
    return fig


# ── Save / Load ──────────────────────────────────────────────────────────────

def save_results(results, true_theta, epsilon, model, y_obs, K, seed):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_data = {
        'results': results,
        'true_theta': true_theta,
        'epsilon': epsilon,
        'model': model,
        'y_obs': y_obs,
        'K': K,
        'seed': seed,
    }
    pkl = _pkl_path(K, seed)
    with open(pkl, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Results saved to: {pkl}")

    fig = create_posterior_plot(results, true_theta, model, y_obs=y_obs)
    fig_path = _fig_path(K, seed)
    save_figure(fig, fig_path)
    print(f"Figure saved to: {fig_path}")


def replot_from_pickle(pkl_path):
    print(f"Loading results from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    results = data['results']
    true_theta = data['true_theta']
    K = data['K']
    seed = data['seed']

    # Rebuild model if not pickled (legacy pkl)
    model = data.get('model')
    if model is None:
        model = GaussianWithNoSummaryStats(K=K, n_obs=100, sigma_0=10, alpha=2, beta=2)
    y_obs = data.get('y_obs')

    print(f"K={K}, seed={seed}, epsilon={data['epsilon']}")
    print(f"M0 values: {results['M0_values']}")
    print(f"Acceptance rates: {results['acceptance_rates']}")

    fig = create_posterior_plot(results, true_theta, model, y_obs=y_obs)
    fig_path = _fig_path(K, seed)
    save_figure(fig, fig_path)
    print(f"Figure saved to: {fig_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Figure 2: Over-sampling posterior comparison"
    )
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rerun', type=str, default=None,
                        help='Path to existing .pkl — replot without re-simulation')
    parser.add_argument('--force', action='store_true',
                        help='Force re-simulation even if cached results exist')
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Figure 2: Over-sampling posterior comparison")
    print(f"Parameters: K={args.K}, seed={args.seed}")

    # Replot from given pickle
    if args.rerun:
        replot_from_pickle(args.rerun)
        return

    # Check cache (skip if --force)
    pkl = _pkl_path(args.K, args.seed)
    if pkl.exists() and not args.force:
        print(f"Found cached results: {pkl}")
        replot_from_pickle(pkl)
        return

    # Run simulation
    model, y_obs, true_theta, key = setup_experiment(K=args.K, seed=args.seed)
    results, epsilon = run_over_sampling_posterior(key, model, y_obs, args.K)
    save_results(results, true_theta, epsilon, model, y_obs, args.K, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
