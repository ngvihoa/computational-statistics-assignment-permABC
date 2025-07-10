#!/usr/bin/env python3
"""
Figure 3: Posterior comparison between permABC and vanilla ABC.

This script generates scatter plots comparing posterior approximations between 
standard ABC and permutation-enhanced ABC on a 2D uniform model.

Usage:
    python fig3_posterior_comparison_perm_vanilla.py
    python fig3_posterior_comparison_perm_vanilla.py --seed 42 --nsim 500000
    python fig3_posterior_comparison_perm_vanilla.py --rerun  # Force new simulation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from jax import random


from permabc.core.distances import optimal_index_distance
from permabc.models.uniform_known import Uniform_known
import pickle


def setup_experiment(seed=42):
    """Setup experimental parameters and generate synthetic data."""
    # Initialize model
    model = Uniform_known(K=2)
    
    # Set strategic observed data to show permutation benefits
    y_obs = np.array([0., 0.5])[None, :, None]
    
    # Setup random key
    key = random.PRNGKey(seed)
    
    # Compute theoretical epsilon threshold
    # Compute theoretical epsilon threshold
    epsilon_star = model.distance(y_obs, y_obs[:, ::-1])[0] / 2
    print(f"Epsilon star: {epsilon_star}")
    return model, y_obs, key, epsilon_star


def run_abc_comparison(model, y_obs, key, Nsim=1000000):
    """Run both standard ABC and permABC for comparison."""
    print("Running ABC and permABC simulations...")
    
    # Generate particles
    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, Nsim)
    key, subkey = random.split(key)
    zs = model.data_generator(subkey, thetas)
    
    # Standard ABC distances
    dists = model.distance(zs, y_obs)
    
    # permABC distances and permutations
    dists_perm, _, zs_index, _ = optimal_index_distance(
        zs=zs, y_obs=y_obs, model=model, epsilon=0, verbose=2
    )
    
    # Apply optimal permutations
    thetas_perm = thetas.apply_permutation(zs_index)
    
    return thetas, thetas_perm, dists, dists_perm


def analyze_performance(dists, dists_perm, epsilon_star):
    """Analyze and compare performance metrics."""
    results = {
        'epsilon_levels': [],
        'abc_counts': [],
        'perm_counts': [],
        'improvement_ratios': []
    }
    
    # Compare distances at different epsilon levels
    epsilons = [np.inf, epsilon_star + 1, epsilon_star, epsilon_star * 0.5]
    
    for eps in epsilons:
        abc_count = np.sum(dists <= eps)
        perm_count = np.sum(dists_perm <= eps)
        
        results['epsilon_levels'].append(eps)
        results['abc_counts'].append(abc_count)
        results['perm_counts'].append(perm_count)
        
        if abc_count > 0:
            ratio = perm_count / abc_count
        else:
            ratio = np.inf if perm_count > 0 else 1.0
        results['improvement_ratios'].append(ratio)
        
        if eps == np.inf:
            print(f"All particles: ABC={abc_count}, permABC={perm_count}")
        else:
            print(f"ε={eps:.3f}: ABC={abc_count}, permABC={perm_count} (ratio: {ratio:.2f})")
    
    # Distance statistics
    print(f"\nDistance Statistics:")
    print(f"ABC - Mean: {np.mean(dists):.4f}, Median: {np.median(dists):.4f}")
    print(f"permABC - Mean: {np.mean(dists_perm):.4f}, Median: {np.median(dists_perm):.4f}")
    
    overall_improvement = np.mean(dists) / np.mean(dists_perm)
    print(f"Overall improvement ratio: {overall_improvement:.2f}")
    
    return results


def create_comparison_plot(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star):
    """Create scatter plot comparison between ABC and permABC."""
    # Define epsilon levels for comparison
    epsilons = [np.inf, epsilon_star + 1, epsilon_star]
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))
    
    # Plot parameters
    N_plot = 10000
    s = 1  # Half-width for true posterior region
    alpha_fig = 0.25
    fontsize = 12
    title_size = 15
    
    # Extract observed values
    y = y_obs[0, :, 0]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, epsilon in enumerate(epsilons):
        # Sample particles within epsilon tolerance
        abc_indices = np.where(dists <= epsilon)[0]
        perm_indices = np.where(dists_perm <= epsilon)[0]
        
        if len(abc_indices) > N_plot:
            abc_indices = np.random.choice(abc_indices, N_plot, replace=False)
        if len(perm_indices) > N_plot:
            perm_indices = np.random.choice(perm_indices, N_plot, replace=False)
        
        # Extract parameters
        mus_abc = thetas.loc[abc_indices].squeeze()
        mus_perm = thetas_perm.loc[perm_indices].squeeze()
        
        # Plot ABC results
        if len(mus_abc) > 0:
            axes[0].scatter(x=mus_abc[:, 0], y=mus_abc[:, 1], 
                           color=colors[i], alpha=alpha_fig, s=10)
        
        # Plot permABC results
        if len(mus_perm) > 0:
            axes[1].scatter(x=mus_perm[:, 0], y=mus_perm[:, 1], 
                           color=colors[i], alpha=alpha_fig, s=10)
    
    # Define regions for plotting
    prior_region = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]])
    posterior_region = np.array([
        [y[0] - s, y[1] - s], [y[0] + s, y[1] - s], 
        [y[0] + s, y[1] + s], [y[0] - s, y[1] + s], 
        [y[0] - s, y[1] - s]
    ])
    
    # Configure ABC plot
    axes[0].scatter(y_obs[0, 0, 0], y_obs[0, 1, 0], c='black', label='y', marker='x', s=100)
    axes[0].plot(posterior_region[:, 0], posterior_region[:, 1], 
                color='black', linestyle='--', linewidth=2, label='True posterior')
    axes[0].plot(prior_region[:, 0], prior_region[:, 1], 
                color='grey', linestyle='--', linewidth=1, label='Prior')
    axes[0].set_title("ABC", fontsize=title_size)
    axes[0].set_xlabel('μ₁', fontsize=fontsize)
    axes[0].set_ylabel('μ₂', fontsize=fontsize)
    axes[0].set_xlim(-2.2, 2.2)
    axes[0].set_ylim(-2.2, 2.2)
    axes[0].set_aspect('equal')
    axes[0].grid(False)
    
    # Configure permABC plot
    axes[1].scatter(y_obs[0, 0, 0], y_obs[0, 1, 0], c='black', label='y', marker='x', s=100)
    axes[1].plot(posterior_region[:, 0], posterior_region[:, 1], 
                color='black', linestyle='--', linewidth=2, label='True posterior')
    axes[1].plot(prior_region[:, 0], prior_region[:, 1], 
                color='grey', linestyle='--', linewidth=1, label='Prior')
    axes[1].set_title('permABC', fontsize=title_size)
    axes[1].set_xlabel('μ₁', fontsize=fontsize)
    axes[1].set_xlim(-2.2, 2.2)
    axes[1].set_ylim(-2.2, 2.2)
    axes[1].set_aspect('equal')
    axes[1].grid(False)
    
    # Add legend to first plot
    axes[0].legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def save_results(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star, 
                performance_results, seed, output_dir):
    """Save results to pickle and create plots."""
    # Create output directories
    results_dir = output_dir  # Direct storage in output_dir
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    figures_dir = os.path.join(BASE_DIR, "figures", "fig3")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create complete data dictionary for pickle
    save_data = {
        'seed': seed,
        'epsilon_star': epsilon_star,
        'n_particles': len(dists),
        'abc_mean_distance': np.mean(dists),
        'perm_mean_distance': np.mean(dists_perm),
        'abc_median_distance': np.median(dists),
        'perm_median_distance': np.median(dists_perm),
        'overall_improvement': np.mean(dists) / np.mean(dists_perm),
        'y_obs_1': y_obs[0, 0, 0],
        'y_obs_2': y_obs[0, 1, 0],
        'performance_results': performance_results,
        'thetas': thetas,
        'thetas_perm': thetas_perm,
        'dists': dists,
        'dists_perm': dists_perm,
        'y_obs': y_obs,
    }
    
    # Save complete data to pickle - directly in results_dir
    pkl_path = os.path.join(results_dir, f"fig3_uniform_seed_{seed}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Results saved to: {pkl_path}")
    
    # Create and save plots
    fig = create_comparison_plot(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star)
    
    # Save plots
    base_name = f"fig3_seed_{seed}"
    fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
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
    print(f"  Epsilon star: {data['epsilon_star']:.4f}")
    print(f"  Overall improvement: {data['overall_improvement']:.2f}")
    print(f"  ABC mean distance: {data['abc_mean_distance']:.4f}")
    print(f"  permABC mean distance: {data['perm_mean_distance']:.4f}")
    
    # Show epsilon-specific results
    perf = data['performance_results']
    for i, eps in enumerate(perf['epsilon_levels']):
        if eps != np.inf:
            abc_count = perf['abc_counts'][i]
            perm_count = perf['perm_counts'][i]
            improvement = perf['improvement_ratios'][i]
            print(f"  ε={eps:.3f}: ABC={abc_count}, permABC={perm_count} (ratio: {improvement:.2f})")
    
    # Recreate the exact plot from original data
    fig = create_comparison_plot(
        data['thetas'], data['thetas_perm'], data['dists'], data['dists_perm'],
        data['y_obs'], data['epsilon_star']
    )
 
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    figures_dir = os.path.join(BASE_DIR, "figures", "fig3")
    seed = data['seed']
    base_name = f"fig3_seed_{seed}"
    fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    plt.show()
    print("Full scatter plot recreated from pickle data.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate posterior comparison plots between ABC and permABC"
    )
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--nsim', type=int, default=1000000,
                       help='Number of simulations (default: 1,000,000)')
    parser.add_argument('--output-dir', type=str, default="experiments/results/fig3",
                       help='Output directory (default: experiments/results/fig3)')
    parser.add_argument('--rerun', action='store_true',
                       help='Force re-simulation even if cached results exist')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("Figure 3: Posterior comparison between permABC and vanilla ABC")
    print(f"Parameters: seed={args.seed}, nsim={args.nsim:,}")
    
    # Handle rerun case - force re-simulation
    if args.rerun:
        print("Rerun mode: forcing new simulation...")
        # Setup experiment and run new simulation
        model, y_obs, key, epsilon_star = setup_experiment(seed=args.seed)
        thetas, thetas_perm, dists, dists_perm = run_abc_comparison(
            model, y_obs, key, Nsim=args.nsim
        )
        performance_results = analyze_performance(dists, dists_perm, epsilon_star)
        pickle_path = save_results(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star,
                                 performance_results, args.seed, args.output_dir)
        print("New simulation complete!")
        print(f"Results: {pickle_path}")
        return
    
    # Check for existing results with this seed
    pickle_path = os.path.join(args.output_dir, f"fig3_uniform_seed_{args.seed}.pkl")
    
    if os.path.exists(pickle_path):
        print(f"Found existing results at {pickle_path}")
        print("Generating plot from cached data...")
        rerun_from_pickle(pickle_path)
        return
    
    # No existing results found, run new simulation
    print("No existing results found, running new simulation...")
    
    # Setup experiment
    model, y_obs, key, epsilon_star = setup_experiment(seed=args.seed)
    
    # Run comparison
    thetas, thetas_perm, dists, dists_perm = run_abc_comparison(
        model, y_obs, key, Nsim=args.nsim
    )
    
    # Analyze performance
    performance_results = analyze_performance(dists, dists_perm, epsilon_star)
    
    # Save results and create plots
    pickle_path = save_results(thetas, thetas_perm, dists, dists_perm, y_obs, epsilon_star,
                             performance_results, args.seed, args.output_dir)
    
    print("Analysis complete!")
    print("The plot shows how permABC better captures the true posterior")
    print("by resolving label-switching issues that affect standard ABC.")
    print(f"Results: {pickle_path}")


if __name__ == "__main__":
    main()