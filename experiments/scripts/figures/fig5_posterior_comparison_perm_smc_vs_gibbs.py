#!/usr/bin/env python3
"""
Figure 5: Posterior comparison between permABC-SMC and ABC-Gibbs sampler.

This script generates a comparison plot showing the performance of permABC-SMC
versus a custom ABC-Gibbs sampler on a Gaussian model with correlated parameters.

Usage:
    python fig5_posterior_comparison_perm_smc_vs_gibbs.py
    python fig5_posterior_comparison_perm_smc_vs_gibbs.py --N 500 --K 10 --seed 42
    python fig5_posterior_comparison_perm_smc_vs_gibbs.py --rerun
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import argparse
from jax import random, vmap, jit
import jax.numpy as jnp
from tqdm import tqdm


from permabc.algorithms.smc import perm_abc_smc
from permabc.core.kernels import KernelTruncatedRW
from permabc.models.Gaussian_with_correlated_params import GaussianWithCorrelatedParams
from permabc.utils.functions import Theta


def setup_experiment(N=1000, K=15, seed=42):
    """Setup experimental parameters and generate synthetic data."""
    # Parameters
    n_obs = 20
    sigma_mu, sigma_alpha = 10., 10.
    
    # Initialize model and generate true data
    if seed ==42:
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
    
    return model, y_obs, true_theta, key, N, K


def run_perm_abc_smc(key, model, N, y_obs):
    """Run permABC-SMC algorithm."""
    print("Running permABC-SMC...")
    key, subkey = random.split(key)
    kernel = KernelTruncatedRW
    
    out_perm_smc = perm_abc_smc(
        key=subkey, model=model, n_particles=N, epsilon_target=0, y_obs=y_obs, 
        kernel=kernel, verbose=0, update_weights_distance=False, Final_iteration=0
    )
    
    mus_perm_smc = out_perm_smc["Thetas"][-1].loc.squeeze()
    betas_perm_smc = out_perm_smc["Thetas"][-1].glob.squeeze()
    n_sim_perm_smc = np.sum(out_perm_smc["N_sim"])
    
    return mus_perm_smc, betas_perm_smc, n_sim_perm_smc


def setup_gibbs_functions(model, K):
    """Setup JIT-compiled functions for Gibbs sampler."""
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
        """ABC step for mu parameters."""
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
        """ABC step for alpha parameter."""
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
    """Run custom ABC-Gibbs sampler."""
    print("Running ABC-Gibbs sampler...")
    
    ABCmus, ABCalpha = setup_gibbs_functions(model, K)
    
    mus = np.zeros((T + 1, K))
    alphas = np.zeros(T + 1)
    Eps_mu = np.zeros((T, K))
    Eps_alpha = np.zeros(T)
    
    key, key_alpha, key_mu = random.split(key, 3)
    
    # Initialize
    mus[0] = random.normal(key_mu, shape=(K,)) * model.sigma_mu
    alphas[0] = random.normal(key_alpha) * model.sigma_alpha
    
    # Gibbs iterations
    for t in tqdm(range(T), desc="Gibbs sampling"):
        key, key_mus = random.split(key)
        mus[t + 1], Eps_mu[t] = ABCmus(key_mus, M_mu, y_obs, alphas[t])
        key, key_alpha = random.split(key)
        alphas[t + 1], Eps_alpha[t] = ABCalpha(key_alpha, M_alpha, y_obs, mus[t + 1])
    
    return mus, alphas, Eps_mu, Eps_alpha


def get_true_posterior(model, K, y_obs, N_pymc=10000):
    """Get true posterior using PyMC (if available)."""
    try:
        import pymc as pm
        
        # Define the model
        with pm.Model() as mod:
            sigma_x = 1.0
            sigma_mu, sigma_alpha = model.sigma_mu, model.sigma_alpha
            
            mu = pm.Normal('mu', mu=0, sigma=sigma_mu, shape=(K, 1))
            alpha = pm.Normal('alpha', mu=0, sigma=sigma_alpha, shape=(1, 1))
            
            # Note: This is a simplified version - you'll need to adapt for your specific model
            x = pm.Normal('x', mu=mu + alpha, sigma=sigma_x, observed=y_obs[0])
            
            # Inference
            trace = pm.sample(N_pymc, return_inferencedata=True)
        
        true_post_mu = np.array(trace.posterior.mu[:, :, :, 0]).reshape(4 * N_pymc, K)
        true_post_alpha = np.array(trace.posterior.alpha).reshape(-1)
        
        return true_post_mu, true_post_alpha
    
    except ImportError:
        print("PyMC not available, skipping true posterior computation")
        return None, None


def run_comparison(model, y_obs, key, N, K):
    """Run both permABC-SMC and ABC-Gibbs for comparison."""
    print("Running comparison between permABC-SMC and ABC-Gibbs...")
    
    # Run permABC-SMC
    mus_perm_smc, betas_perm_smc, n_sim_perm_smc = run_perm_abc_smc(key, model, N, y_obs)
    
    # Setup Gibbs parameters
    M = n_sim_perm_smc // (2 * K * N)
    print(f"Using M = {M} for Gibbs sampler")
    
    # Run Gibbs sampler
    key, subkey = random.split(key)
    mus_gibbs, alphas_gibbs, Eps_mus_gibbs, Eps_alphas_gibbs = run_gibbs_sampler(
        subkey, model, K, y_obs[0], N, M, M
    )
    
    # Get true posterior (optional)
    true_post_mu, true_post_alpha = get_true_posterior(model, K, y_obs, N_pymc=N)
    
    return {
        'mus_perm_smc': mus_perm_smc,
        'betas_perm_smc': betas_perm_smc,
        'mus_gibbs': mus_gibbs,
        'alphas_gibbs': alphas_gibbs,
        'true_post_mu': true_post_mu,
        'true_post_alpha': true_post_alpha,
        'n_sim_perm_smc': n_sim_perm_smc,
        'M': M
    }


def analyze_performance(results):
    """Analyze and compare performance metrics."""
    print("\nPerformance Analysis:")
    print(f"permABC-SMC simulations: {results['n_sim_perm_smc']}")
    print(f"Gibbs M parameter: {results['M']}")
    
    # Compare posterior statistics
    mu_perm_mean = np.mean(results['mus_perm_smc'], axis=0)
    mu_gibbs_mean = np.mean(results['mus_gibbs'], axis=0)
    
    beta_perm_mean = np.mean(results['betas_perm_smc'])
    alpha_gibbs_mean = np.mean(results['alphas_gibbs'])
    
    print(f"permABC-SMC μ mean: {mu_perm_mean[:3]} (showing first 3)")
    print(f"Gibbs μ mean: {mu_gibbs_mean[:3]} (showing first 3)")
    print(f"permABC-SMC β mean: {beta_perm_mean:.4f}")
    print(f"Gibbs α mean: {alpha_gibbs_mean:.4f}")
    
    # Calculate coverage statistics if true posterior available
    if results['true_post_mu'] is not None:
        print("True posterior comparison available")
    else:
        print("True posterior comparison not available")
    
    return {
        'mu_perm_mean': mu_perm_mean,
        'mu_gibbs_mean': mu_gibbs_mean,
        'beta_perm_mean': beta_perm_mean,
        'alpha_gibbs_mean': alpha_gibbs_mean
    }


def create_comparison_plot(results, k=0):
    """Create comparison plot between permABC-SMC and ABC-Gibbs."""
    # Plot parameters
    title_size = 15
    fontsize = 12
    alpha_fig = 0.25
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    
    # ABC-Gibbs plot
    axes[0].set_aspect('equal')
    axes[0].scatter(x=results['mus_gibbs'][:, k], y=results['alphas_gibbs'], 
                   label="ABC-Gibbs", color="red", alpha=alpha_fig)
    axes[0].set_title("ABC-Gibbs", fontsize=title_size)
    axes[0].set_xlabel("$\\mu_1$", fontsize=fontsize)
    axes[0].set_ylabel("$\\beta$", fontsize=fontsize)
    axes[0].set_xlim(-10, 10)
    axes[0].set_ylim(-10, 10)
    
    # permABC-SMC plot
    axes[1].set_aspect('equal')
    axes[1].scatter(x=results['mus_perm_smc'][:, k], y=results['betas_perm_smc'], 
                   label="permABC-SMC", color="blue", alpha=alpha_fig)
    axes[1].set_title("permABC-SMC", fontsize=title_size)
    axes[1].set_xlabel("$\\mu_1$", fontsize=fontsize)
    axes[1].set_ylabel("$\\beta$", fontsize=fontsize)
    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(-10, 10)
    
    # Add true posterior if available
    if results['true_post_mu'] is not None and results['true_post_alpha'] is not None:
        sns.kdeplot(x=results['true_post_mu'][:, k], y=results['true_post_alpha'], 
                   label="True posterior", color="black", alpha=1, ax=axes[0])
        sns.kdeplot(x=results['true_post_mu'][:, k], y=results['true_post_alpha'], 
                   label="True posterior", color="black", alpha=1, ax=axes[1])
        # axes[0].legend(loc='upper right', fontsize=10)
        # axes[1].legend(loc='upper right', fontsize=10)
    
    return fig


def save_results(results, model, y_obs, true_theta, seed, N, K, output_dir):
    """Save results to pickle and create plots."""
    # Create output directories
    results_dir = output_dir
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    figures_dir = os.path.join(BASE_DIR, "figures", "fig5")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create complete data dictionary for pickle
    save_data = {
        'seed': seed,
        'N': N,
        'K': K,
        'model_params': {
            'sigma_mu': model.sigma_mu,
            'sigma_alpha': model.sigma_alpha,
            'n_obs': model.n_obs
        },
        'true_theta': true_theta,
        'y_obs': y_obs,
        'results': results,
        'performance_analysis': analyze_performance(results)
    }
    
    # Save complete data to pickle
    pkl_path = os.path.join(results_dir, f"fig5_perm_vs_gibbs_K_{K}_seed_{seed}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Results saved to: {pkl_path}")
    
    # Create and save plots
    fig = create_comparison_plot(results)
    
    # Save plots in multiple formats
    base_name = f"fig5_K_{K}_seed_{seed}"
    fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.svg"), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figures saved to: {figures_dir}/{base_name}.*")
    
    return pkl_path


def rerun_from_pickle(pkl_path):
    """Recreate analysis from existing pickle results."""
    print(f"Loading results from: {pkl_path}")
    
    # Load results
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print("Summary of results:")
    print(f"  Seed: {data['seed']}")
    print(f"  N: {data['N']}, K: {data['K']}")
    print(f"  Model parameters: {data['model_params']}")
    K = data['K']
    # Show performance analysis
    perf = data['performance_analysis']
    print(f"  permABC-SMC μ mean: {perf['mu_perm_mean'][:3]} (first 3)")
    print(f"  Gibbs μ mean: {perf['mu_gibbs_mean'][:3]} (first 3)")
    print(f"  permABC-SMC β mean: {perf['beta_perm_mean']:.4f}")
    print(f"  Gibbs α mean: {perf['alpha_gibbs_mean']:.4f}")
    
    # Recreate the exact plot from original data
    fig = create_comparison_plot(data['results'])
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    figures_dir = os.path.join(BASE_DIR, "figures", "fig5")
    seed = data['seed']
    base_name = f"fig5_K_{K}_seed_{seed}"
    fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plot recreated from pickle data.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate posterior comparison plots between permABC-SMC and ABC-Gibbs"
    )
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--N', type=int, default=1000,
                       help='Number of particles (default: 1000)')
    parser.add_argument('--K', type=int, default=10,
                       help='Number of parameters (default: 10)')
    parser.add_argument('--output-dir', type=str, default="experiments/results/fig5",
                       help='Output directory (default: experiments/results/fig5)')
    parser.add_argument('--rerun', action='store_true',
                       help='Force re-simulation even if cached results exist')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("Figure 5: Posterior comparison between permABC-SMC and ABC-Gibbs")
    print(f"Parameters: seed={args.seed}, N={args.N}, K={args.K}")
    # args.seed = int(args.seed)+1  # Increment seed for reproducibility
    # Handle rerun case - force re-simulation
    if args.rerun:
        print("Rerun mode: forcing new simulation...")
        # Setup experiment and run new simulation
        model, y_obs, true_theta, key, N, K = setup_experiment(N=args.N, K=args.K, seed=args.seed)
        results = run_comparison(model, y_obs, key, N, K)
        pickle_path = save_results(results, model, y_obs, true_theta, args.seed, N, K, args.output_dir)
        print("New simulation complete!")
        print(f"Results: {pickle_path}")
        return
    
    # Check for existing results with this seed
    pickle_path = os.path.join(args.output_dir, f"fig5_perm_vs_gibbs_K_{args.K}_seed_{args.seed}.pkl")
    
    if os.path.exists(pickle_path):
        print(f"Found existing results at {pickle_path}")
        print("Generating plot from cached data...")
        rerun_from_pickle(pickle_path)
        return
    
    # No existing results found, run new simulation
    print("No existing results found, running new simulation...")
    
    # Setup experiment
    model, y_obs, true_theta, key, N, K = setup_experiment(N=args.N, K=args.K, seed=args.seed)
    
    # Run comparison
    results = run_comparison(model, y_obs, key, N, K)
    
    # Save results and create plots
    pickle_path = save_results(results, model, y_obs, true_theta, args.seed, N, K, args.output_dir)
    
    print("Analysis complete!")
    print("The plot shows the comparison between permABC-SMC and ABC-Gibbs samplers")
    print("for posterior approximation on a Gaussian model with correlated parameters.")
    print(f"Results: {pickle_path}")


if __name__ == "__main__":
    main()