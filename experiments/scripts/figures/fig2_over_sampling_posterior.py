#!/usr/bin/env python3
"""
Figure 2: Over-sampling posterior comparison.

This script generates posterior distributions showing the effect of over-sampling
on different values of M0 for a Gaussian model.

Usage:
    python fig2_over_sampling_posterior.py
    python fig2_over_sampling_posterior.py --K 10 --seed 42
    python fig2_over_sampling_posterior.py --rerun  # Force new simulation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from jax import random
from scipy.stats import norm, invgamma
import pickle


from permabc.core.distances import optimal_index_distance
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta

def setup_experiment(K=10, K_outliers=0, seed=42):
    """Setup experimental parameters and generate synthetic data."""
    # Configuration
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    n = 100
    sigma0 = 10
    alpha, beta = 3., 5.
    
    # Initialize model
    model = GaussianWithNoSummaryStats(K=K, n_obs=n, sigma_0=sigma0, alpha=alpha, beta=beta)
    true_theta = model.prior_generator(subkey, 1)
    
    # Set specific values for clear visualization
    true_theta = Theta(
        glob=true_theta.glob.at[0, 0].set(10.),
        loc=true_theta.loc.at[0, 0, 0].set(0.)
    )
    
    print(f"True sigma2: {true_theta.glob[0, 0]}")
    
    # Generate observed data
    key, subkey = random.split(key)
    y_obs = model.data_generator(subkey, true_theta)
    
    return model, y_obs, true_theta, key


def run_over_sampling_posterior(key, model, y_obs, K):
    """Run over-sampling analysis for different M0 values."""
    print("Running over-sampling posterior analysis...")
    
    # Configuration
    N_epsilon = 20000
    M0s = np.array([K, 1.1*K, 1.2*K, 1.5*K, 2*K, 5*K, 7.5*K, 10*K], dtype=int)
    alpha_epsilon = 0.05
    
    # Generate samples for largest M0
    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, N_epsilon, np.max(M0s))
    key, subkey = random.split(key)
    zs = model.data_generator(subkey, thetas)
    
    # Calculate epsilon based on K components
    dists_perm, _, _, _ = optimal_index_distance(model, zs[:, :K], y_obs, M=K)
    epsilon = np.quantile(dists_perm, alpha_epsilon)
    print(f"Epsilon threshold: {epsilon}")
    
    results = {
        'M0_values': [],
        'glob_posteriors': [],
        'loc_posteriors': [],
        'acceptance_rates': []
    }
    
    # Process each M0 value
    for M0 in M0s:
        print(f"Processing M0 = {M0}")
        
        # Calculate distances for current M0
        dists_perm, ys_index, zs_index, _ = optimal_index_distance(
            model, zs[:, :M0], y_obs, M=M0
        )
        
        # Apply permutations and filter by epsilon
        thetas_perm = thetas.apply_permutation(zs_index)
        accepted = dists_perm < epsilon
        thetas_accepted = thetas_perm[accepted]
        
        acceptance_rate = np.sum(accepted) / N_epsilon
        print(f"  Acceptance rate: {acceptance_rate:.2%}")
        
        # Store results
        results['M0_values'].append(M0)
        results['glob_posteriors'].append(thetas_accepted.glob[:, 0])
        results['loc_posteriors'].append(thetas_accepted.loc[:, 0, 0])
        results['acceptance_rates'].append(acceptance_rate)
    
    return results, epsilon


def create_posterior_plot(results, true_theta):
    """Create posterior comparison plot."""
    M0s = results['M0_values']
    colors = plt.cm.viridis(np.linspace(0, 1, len(M0s)))
    
    # Setup plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fontsize = 12
    titlesize = 15
    
    # Plot global parameter posteriors
    for i, M0 in enumerate(M0s):
        glob_samples = results['glob_posteriors'][i]
        if len(glob_samples) > 0:
            sns.kdeplot(glob_samples, ax=axes[0], color=colors[i], 
                       label=f"M = {M0}")
    
    # Add prior for global parameter
    prior_glob = invgamma(a=3, scale=5)
    a_glob, b_glob = prior_glob.interval(0.999)
    x_glob = np.linspace(a_glob, b_glob, 1000)
    axes[0].plot(x_glob, prior_glob.pdf(x_glob), 
                linestyle="--", color="grey", label="Prior")
    # axes[0].axvline(true_theta.glob[0, 0], color='red', linestyle='-', 
    #                linewidth=2, label=f"True: {true_theta.glob[0, 0]}")
    
    # axes[0].legend()
    axes[0].set_ylabel("Density", fontsize=fontsize)
    axes[0].set_xlabel("β", fontsize=fontsize)
    axes[0].set_title("Global parameter", fontsize=titlesize)
    axes[0].set_xlim(0, 10)
    
    # Plot local parameter posteriors
    for i, M0 in enumerate(M0s):
        loc_samples = results['loc_posteriors'][i]
        if len(loc_samples) > 0:
            sns.kdeplot(loc_samples, ax=axes[1], color=colors[i], 
                       label=f"M = {M0}")
    
    # Add prior for local parameter
    prior_loc = norm(loc=0, scale=10)
    a_loc, b_loc = prior_loc.interval(0.999)
    x_loc = np.linspace(a_loc, b_loc, 1000)
    axes[1].plot(x_loc, prior_loc.pdf(x_loc), 
                linestyle="--", color="grey", label="Prior")
    # axes[1].axvline(true_theta.loc[0, 0, 0], color='red', linestyle='-', 
                #    linewidth=2, label=f"True: {true_theta.loc[0, 0, 0]}")
    
    axes[1].set_ylabel("")
    axes[1].set_xlabel("μ₁", fontsize=fontsize)
    axes[1].set_title("Local parameter", fontsize=titlesize)
    axes[1].set_xlim(-10, 10)
    
    plt.tight_layout()
    return fig


def save_results(results, true_theta, epsilon, K, seed, output_dir):
    """Save results to pickle and create plots."""
    # Create output directories
    results_dir = output_dir  # Direct storage in output_dir
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    figures_dir = os.path.join(BASE_DIR, "figures", "fig2")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Prepare data to pickle
    save_data = {
        'results': results,
        'true_theta': true_theta,
        'epsilon': epsilon,
        'K': K,
        'seed': seed
    }
    
    # Save to pickle - directly in results_dir
    pickle_path = os.path.join(results_dir, f"fig2_over_sampling_K_{K}_seed_{seed}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Results saved to: {pickle_path}")
    
    # Create and save plots
    fig = create_posterior_plot(results, true_theta)
    
    # Save plots
    base_name = f"fig2_seed_{seed}"
    fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figures saved to: {figures_dir}/{base_name}.*")
    
    return pickle_path


def rerun_from_pickle(pickle_path):
    """Recreate plots from existing pickle results."""
    print(f"Recreating plots from: {pickle_path}")
    
    # Load results
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    results = data['results']
    true_theta = data['true_theta']
    epsilon = data['epsilon']
    K = data['K']
    seed = data['seed']
    
    print("Loaded data:")
    print(f"K={K}, seed={seed}, epsilon={epsilon}")
    print("M0 values:", results['M0_values'])
    print("Acceptance rates:", results['acceptance_rates'])
    
    # Create and show plot
    fig = create_posterior_plot(results, true_theta)
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    figures_dir = os.path.join(BASE_DIR, "figures", "fig2")

    base_name = f"fig2_seed_{seed}"
    fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("Figures saved to:",  os.path.join(figures_dir, f"{base_name}.*"))



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate over-sampling posterior comparison plots"
    )
    
    parser.add_argument('--K', type=int, default=10,
                       help='Number of components (default: 10)')
    parser.add_argument('--K_outliers', type=int, default=0,
                       help='Number of outlier components (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default="experiments/results/fig2",
                       help='Output directory (default: experiments/results/fig2)')
    parser.add_argument('--rerun', action='store_true',
                       help='Force re-simulation even if cached results exist')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("Figure 2: Over-sampling posterior comparison")
    print(f"Parameters: K={args.K}, K_outliers={args.K_outliers}, seed={args.seed}")
    
    # Handle rerun case - force re-simulation
    if args.rerun:
        print("Rerun mode: forcing new simulation...")
        # Setup experiment and run new simulation
        model, y_obs, true_theta, key = setup_experiment(
            K=args.K, K_outliers=args.K_outliers, seed=args.seed
        )
        results, epsilon = run_over_sampling_posterior(key, model, y_obs, args.K)
        pickle_path = save_results(results, true_theta, epsilon, args.K, args.seed, args.output_dir)
        print("New simulation complete!")
        print(f"Results: {pickle_path}")
        return
    
    # Check for existing results with this K and seed
    pickle_path = os.path.join(args.output_dir, f"fig2_over_sampling_K_{args.K}_seed_{args.seed}.pkl")
    
    if os.path.exists(pickle_path):
        print(f"Found existing results at {pickle_path}")
        print("Generating plot from cached data...")
        rerun_from_pickle(pickle_path)
        return
    
    # No existing results found, run new simulation
    print("No existing results found, running new simulation...")
    
    # Setup experiment
    model, y_obs, true_theta, key = setup_experiment(
        K=args.K, K_outliers=args.K_outliers, seed=args.seed
    )
    
    # Run analysis
    results, epsilon = run_over_sampling_posterior(key, model, y_obs, args.K)
    
    # Save results and create plots
    pickle_path = save_results(results, true_theta, epsilon, args.K, args.seed, args.output_dir)
    
    print("Analysis complete!")
    print(f"Results: {pickle_path}")


if __name__ == "__main__":
    main()