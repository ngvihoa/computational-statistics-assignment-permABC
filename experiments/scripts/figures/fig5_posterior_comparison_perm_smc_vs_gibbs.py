#!/usr/bin/env python3
"""
Figure 5: Posterior comparison between permABC-SMC and ABC-Gibbs sampler.

Scatter plots on a Gaussian model with correlated parameters.

Usage:
    python fig5_posterior_comparison_perm_smc_vs_gibbs.py
    python fig5_posterior_comparison_perm_smc_vs_gibbs.py --N 500 --K 10 --seed 42
    python fig5_posterior_comparison_perm_smc_vs_gibbs.py --rerun path/to/results.pkl
    python fig5_posterior_comparison_perm_smc_vs_gibbs.py --force
"""

import sys
import argparse
import pickle
from pathlib import Path

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random, vmap, jit
import jax.numpy as jnp
from tqdm import tqdm

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.models.Gaussian_with_correlated_params import GaussianWithCorrelatedParams
from permabc.utils.functions import Theta


# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "posterior_comparison"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "fig5"


def _pkl_path(K, seed):
    return RESULTS_DIR / f"fig5_perm_vs_gibbs_K_{K}_seed_{seed}.pkl"


def _fig_path(K, seed):
    return FIGURES_DIR / f"fig5_perm_vs_gibbs_K_{K}_seed_{seed}.pdf"


# ── Experiment ───────────────────────────────────────────────────────────────

def setup_experiment(N=1000, K=20, seed=42):
    n_obs = 20
    sigma_mu, sigma_alpha = 10., 10.

    if seed == 42:
        seed = 0
    key = random.PRNGKey(seed)
    key, key_theta, key_yobs = random.split(key, 3)

    model = GaussianWithCorrelatedParams(K=K, n_obs=n_obs, sigma_mu=sigma_mu, sigma_alpha=sigma_alpha)
    true_theta = model.prior_generator(key_theta, 1)
    true_theta = Theta(
        loc=true_theta.loc.at[0, 0, 0].set(0.),
        glob=true_theta.glob.at[0, 0].set(0.)
    )
    y_obs = model.data_generator(key_yobs, true_theta)
    # Centre first component observations at 0 so posterior is visually centred
    y_obs = np.array(y_obs)
    y_obs[0, 0, :] -= np.mean(y_obs[0, 0, :])
    return model, y_obs, true_theta, key, N, K


def run_perm_abc_smc(key, model, N, y_obs):
    print("Running permABC-SMC...")
    key, subkey = random.split(key)
    out = perm_abc_smc(
        key=subkey, model=model, n_particles=N, epsilon_target=0, y_obs=y_obs,
        kernel=KernelTruncatedRW, verbose=0, update_weights_distance=False,
        Final_iteration=0,
    )
    mus = out["Thetas"][-1].loc.reshape(-1, model.K)
    betas = out["Thetas"][-1].glob.squeeze()
    n_sim = np.sum(out["N_sim"])
    return mus, betas, n_sim


def setup_gibbs_functions(model, K):
    @jit
    def distance_one_silo(x_k, y_k):
        return jnp.sum((x_k - y_k) ** 2)

    @jit
    def distance_all_silo(x, y):
        return vmap(distance_one_silo, in_axes=(0, 0))(x, y)

    @jit
    def distance_xs(xs, y):
        return vmap(distance_all_silo, in_axes=(0, None))(xs, y)

    @jit
    def distance_sum_silo(x, y):
        return jnp.mean(distance_all_silo(x, y))

    @jit
    def distance_sum(xs, y):
        return vmap(distance_sum_silo, in_axes=(0, None))(xs, y)

    def ABCmus(key, M, y_obs, alpha):
        key, key_mus, key_data = random.split(key, 3)
        mus = random.normal(key_mus, shape=(M, K)) * model.sigma_mu
        thetas = Theta(loc=mus[:, :, None], glob=np.repeat([alpha], M)[:, None])
        xs = model.data_generator(key_data, thetas)
        dists = distance_xs(xs, y_obs)
        index_min = jnp.argmin(dists, axis=0)
        Eps_betas = jnp.array([dists[index_min[i], i] for i in range(K)])
        mus_min = np.array([mus[index_min[i], i] for i in range(K)])
        return mus_min, Eps_betas

    def ABCalpha(key, M, y_obs, mus):
        key, key_alpha, key_data = random.split(key, 3)
        alphas = random.normal(key_alpha, shape=(M, 1)) * model.sigma_alpha
        thetas = Theta(loc=np.repeat([mus], M, axis=0)[:, :, None], glob=alphas)
        xs = model.data_generator(key_data, thetas)
        dists = distance_sum(xs, y_obs)
        index_min = jnp.argmin(dists)
        Eps_alpha = dists[index_min]
        alpha_min = alphas[index_min]
        return alpha_min[0], Eps_alpha

    return ABCmus, ABCalpha


def run_gibbs_sampler(key, model, K, y_obs, T, M_mu, M_alpha):
    print("Running ABC-Gibbs sampler...")
    ABCmus, ABCalpha = setup_gibbs_functions(model, K)

    mus = np.zeros((T + 1, K))
    alphas = np.zeros(T + 1)
    Eps_mu = np.zeros((T, K))
    Eps_alpha = np.zeros(T)

    key, key_alpha, key_mu = random.split(key, 3)
    mus[0] = random.normal(key_mu, shape=(K,)) * model.sigma_mu
    alphas[0] = random.normal(key_alpha) * model.sigma_alpha

    for t in tqdm(range(T), desc="Gibbs sampling"):
        key, key_mus = random.split(key)
        mus[t + 1], Eps_mu[t] = ABCmus(key_mus, M_mu, y_obs, alphas[t])
        key, key_alpha = random.split(key)
        alphas[t + 1], Eps_alpha[t] = ABCalpha(key_alpha, M_alpha, y_obs, mus[t + 1])

    return mus, alphas, Eps_mu, Eps_alpha


def get_true_posterior_marginal(model, K, y_obs, k=0):
    """Exact 2D Gaussian posterior of (mu_k, alpha) given y.

    Model: mu_k ~ N(0, sigma_mu^2), alpha ~ N(0, sigma_alpha^2),
           X_{k,i} ~ N(mu_k + alpha, 1).
    Everything is linear-Gaussian so the joint posterior of
    (mu_1,...,mu_K, alpha) is multivariate Gaussian.

    Returns (mean_2d, cov_2d) for the (mu_k, alpha) marginal.
    """
    n = model.n_obs
    sigma_mu = model.sigma_mu
    sigma_alpha = model.sigma_alpha
    y = np.asarray(y_obs[0])  # (K, n_obs)
    S = np.sum(y, axis=1)     # (K,)

    # Prior precision (K+1 x K+1): [mu_1,...,mu_K, alpha]
    diag_prior = np.concatenate([np.full(K, 1.0 / sigma_mu ** 2),
                                  [1.0 / sigma_alpha ** 2]])
    Lambda_prior = np.diag(diag_prior)

    # Likelihood precision
    Lambda_lik = np.zeros((K + 1, K + 1))
    for j in range(K):
        Lambda_lik[j, j] = n
        Lambda_lik[j, K] = n
        Lambda_lik[K, j] = n
    Lambda_lik[K, K] = n * K

    # Likelihood information vector
    eta_lik = np.zeros(K + 1)
    eta_lik[:K] = S
    eta_lik[K] = np.sum(S)

    # Posterior
    Lambda_post = Lambda_prior + Lambda_lik
    Sigma_post = np.linalg.inv(Lambda_post)
    mu_post = Sigma_post @ eta_lik

    # Extract (mu_k, alpha) marginal
    idx = [k, K]
    return mu_post[idx], Sigma_post[np.ix_(idx, idx)]


def run_comparison(model, y_obs, key, N, K):
    print("Running comparison between permABC-SMC and ABC-Gibbs...")

    t0 = time.time()
    mus_perm_smc, betas_perm_smc, n_sim_perm_smc = run_perm_abc_smc(key, model, N, y_obs)
    time_perm = time.time() - t0
    print(f"permABC-SMC time: {time_perm:.1f}s")

    M = n_sim_perm_smc // (2 * K * N)
    print(f"Using M = {M} for Gibbs sampler")

    key, subkey = random.split(key)
    t0 = time.time()
    mus_gibbs, alphas_gibbs, Eps_mus_gibbs, Eps_alphas_gibbs = run_gibbs_sampler(
        subkey, model, K, y_obs[0], N, M, M
    )
    time_gibbs = time.time() - t0
    print(f"ABC-Gibbs time: {time_gibbs:.1f}s")

    post_mean, post_cov = get_true_posterior_marginal(model, K, y_obs, k=0)

    return {
        'mus_perm_smc': mus_perm_smc,
        'betas_perm_smc': betas_perm_smc,
        'mus_gibbs': mus_gibbs,
        'alphas_gibbs': alphas_gibbs,
        'true_post_mean': post_mean,
        'true_post_cov': post_cov,
        'n_sim_perm_smc': n_sim_perm_smc,
        'M': M,
        'time_perm': time_perm,
        'time_gibbs': time_gibbs,
    }


def analyze_performance(results):
    print("\nPerformance Analysis:")
    print(f"  permABC-SMC simulations: {results['n_sim_perm_smc']}")
    print(f"  Gibbs M parameter: {results['M']}")
    mu_perm_mean = np.atleast_1d(np.mean(results['mus_perm_smc'], axis=0))
    mu_gibbs_mean = np.atleast_1d(np.mean(results['mus_gibbs'], axis=0))
    beta_perm_mean = np.mean(results['betas_perm_smc'])
    alpha_gibbs_mean = np.mean(results['alphas_gibbs'])
    print(f"  permABC-SMC mu mean: {mu_perm_mean[:3]} (first 3)")
    print(f"  Gibbs mu mean: {mu_gibbs_mean[:3]} (first 3)")
    print(f"  permABC-SMC beta mean: {beta_perm_mean:.4f}")
    print(f"  Gibbs alpha mean: {alpha_gibbs_mean:.4f}")
    return {
        'mu_perm_mean': mu_perm_mean,
        'mu_gibbs_mean': mu_gibbs_mean,
        'beta_perm_mean': beta_perm_mean,
        'alpha_gibbs_mean': alpha_gibbs_mean,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_gaussian_contours(ax, mean, cov, levels=None, **kwargs):
    """Draw contour lines of a 2D Gaussian on *ax*."""
    from scipy.stats import multivariate_normal
    if levels is None:
        levels = [0.5, 0.9, 0.99]

    # Mahalanobis radii for the given probability levels
    from scipy.stats import chi2
    r2 = chi2.ppf(levels, df=2)

    # Grid around the mean
    std_x = np.sqrt(cov[0, 0])
    std_y = np.sqrt(cov[1, 1])
    margin = 4
    x = np.linspace(mean[0] - margin * std_x, mean[0] + margin * std_x, 200)
    y = np.linspace(mean[1] - margin * std_y, mean[1] + margin * std_y, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)

    # Convert probability levels to density thresholds
    peak = rv.pdf(mean)
    thresholds = [peak * np.exp(-0.5 * r) for r in r2]
    thresholds.sort()

    ax.contour(X, Y, Z, levels=thresholds, **kwargs)


def create_comparison_plot(results, k=0, show_time=False):
    alpha_fig = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    # Centre axes on posterior mean
    post_mean = results.get('true_post_mean')
    post_cov = results.get('true_post_cov')
    if post_mean is not None and post_cov is not None:
        cx, cy = post_mean
        half = max(4 * np.sqrt(post_cov[0, 0]), 4 * np.sqrt(post_cov[1, 1]), 8)
    else:
        cx, cy, half = 0, 0, 10

    axes[0].set_aspect('equal')
    axes[0].scatter(results['mus_gibbs'][:, k], results['alphas_gibbs'],
                    label="ABC-Gibbs", color="#d62728", alpha=alpha_fig)
    axes[0].set_title("ABC-Gibbs")
    axes[0].set_xlabel(r"$\mu_1$")
    axes[0].set_ylabel(r"$\alpha$")
    axes[0].set_xlim(cx - half, cx + half)
    axes[0].set_ylim(cy - half, cy + half)

    axes[1].set_aspect('equal')
    axes[1].scatter(results['mus_perm_smc'][:, k], results['betas_perm_smc'],
                    label="permABC-SMC", color="#1f77b4", alpha=alpha_fig)
    axes[1].set_title("permABC-SMC")
    axes[1].set_xlabel(r"$\mu_1$")
    axes[1].set_ylabel(r"$\alpha$")
    axes[1].set_xlim(cx - half, cx + half)
    axes[1].set_ylim(cy - half, cy + half)

    # True posterior contours (analytic Gaussian)
    post_mean = results.get('true_post_mean')
    post_cov = results.get('true_post_cov')
    if post_mean is not None and post_cov is not None:
        for ax in axes:
            _plot_gaussian_contours(ax, post_mean, post_cov,
                                    colors="black", linewidths=1.5)

    # Show execution time
    if show_time:
        t_gibbs = results.get('time_gibbs')
        t_perm = results.get('time_perm')
        if t_gibbs is not None:
            axes[0].text(0.05, 0.95, f"t = {t_gibbs:.1f}s",
                         transform=axes[0].transAxes, fontsize=11,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        if t_perm is not None:
            axes[1].text(0.05, 0.95, f"t = {t_perm:.1f}s",
                         transform=axes[1].transAxes, fontsize=11,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    return fig


# ── Save / Load ──────────────────────────────────────────────────────────────

def save_results(results, model, y_obs, true_theta, seed, N, K, show_time=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_data = {
        'seed': seed,
        'N': N,
        'K': K,
        'model_params': {
            'sigma_mu': model.sigma_mu,
            'sigma_alpha': model.sigma_alpha,
            'n_obs': model.n_obs,
        },
        'true_theta': true_theta,
        'y_obs': y_obs,
        'results': results,
        'performance_analysis': analyze_performance(results),
    }
    pkl = _pkl_path(K, seed)
    with open(pkl, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Results saved to: {pkl}")

    fig = create_comparison_plot(results, show_time=show_time)
    fig_path = _fig_path(K, seed)
    save_figure(fig, fig_path)
    print(f"Figure saved to: {fig_path}")


def replot_from_pickle(pkl_path, show_time=False):
    print(f"Loading results from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    K = data['K']
    seed = data['seed']
    print(f"  K={K}, seed={seed}, N={data['N']}")

    fig = create_comparison_plot(data['results'], show_time=show_time)
    fig_path = _fig_path(K, seed)
    save_figure(fig, fig_path)
    print(f"Figure saved to: {fig_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Figure 5: Posterior comparison permABC-SMC vs ABC-Gibbs"
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--rerun', type=str, default=None,
                        help='Path to existing .pkl — replot without re-simulation')
    parser.add_argument('--force', action='store_true',
                        help='Force re-simulation even if cached results exist')
    parser.add_argument('--show-time', action='store_true',
                        help='Display execution time on each panel')
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Figure 5: Posterior comparison permABC-SMC vs ABC-Gibbs")
    print(f"Parameters: seed={args.seed}, N={args.N}, K={args.K}")

    if args.rerun:
        replot_from_pickle(args.rerun, show_time=args.show_time)
        return

    pkl = _pkl_path(args.K, args.seed)
    if pkl.exists() and not args.force:
        print(f"Found cached results: {pkl}")
        replot_from_pickle(pkl, show_time=args.show_time)
        return

    model, y_obs, true_theta, key, N, K = setup_experiment(N=args.N, K=args.K, seed=args.seed)
    results = run_comparison(model, y_obs, key, N, K)
    save_results(results, model, y_obs, true_theta, args.seed, N, K, show_time=args.show_time)
    print("Done.")


if __name__ == "__main__":
    main()
