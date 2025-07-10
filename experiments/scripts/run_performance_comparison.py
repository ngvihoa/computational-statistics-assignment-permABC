#!/usr/bin/env python3
"""
Performance comparison runner for ABC methods.

This script runs comprehensive performance comparisons between various ABC algorithms.
Used by fig4 and fig6 with different osum settings.

Usage:
    python run_performance_comparison.py --K 20 --K_outliers 4 --osum
    python run_performance_comparison.py --K 20 --K_outliers 0 --no-osum --seed 42
    python run_performance_comparison.py --rerun experiments/results/performance_K_20_outliers_4_osum_True_seed_42.pkl
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
import time
from jax import random



from permabc.algorithms.smc import abc_smc, perm_abc_smc
from permabc.algorithms.pmc import abc_pmc
from permabc.algorithms.over_sampling import perm_abc_smc_os
from permabc.algorithms.under_matching import perm_abc_smc_um
from permabc.core.kernels import KernelTruncatedRW
from permabc.core.distances import optimal_index_distance
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta

def select_indices(L, m, facteur=2.0):
    """Extract m elements with exponential spacing."""
    n = len(L)
    if m >= n:
        return list(range(n))
    if m <= 0:
        return []
    
    indices = []
    for i in range(m):
        t = i / (m - 1) if m > 1 else 0
        if facteur != 1:
            mapped_t = (facteur ** t - 1) / (facteur - 1)
        else:
            mapped_t = t
        indice = int(mapped_t * (n - 1))
        indices.append(indice)
    
    return sorted(list(set(indices)))


def setup_experiment(K=20, K_outliers=4, seed=42, N_points=1000000, N_particles=1000):
    """Setup experimental parameters and generate synthetic data with outliers."""
    # Parameters
    stopping_rate = 0.0
    
    # Model parameters
    n = 10
    sigma0 = 10
    alpha, beta = 5, 5
    
    # Initialize model and generate true data
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    
    model = GaussianWithNoSummaryStats(K=K, n_obs=n, sigma_0=sigma0, alpha=alpha, beta=beta)
    true_theta = model.prior_generator(subkey, 1)
    true_theta = Theta(loc = true_theta.loc, glob=np.array([1.])[None, :])
    
    # Add outliers
    for i in range(K_outliers):
        key, subkey = random.split(key)
        sign = random.choice(subkey, a=np.array([-1, 1]), shape=(1,))
        key, subkey = random.split(key)
        if sign == 1:
            outlier_val = random.uniform(subkey, shape=(1,), minval=-3*sigma0, maxval=-2*sigma0)[0]
        else:
            outlier_val = random.uniform(subkey, shape=(1,), minval=2*sigma0, maxval=3*sigma0)[0]
        true_theta = Theta(
            loc=true_theta.loc.at[0, i, 0].set(outlier_val),
            glob=true_theta.glob
        )

    key, subkey = random.split(key)
    y_obs = model.data_generator(subkey, true_theta)
    
    return model, y_obs, true_theta, key, N_points, N_particles, stopping_rate, K




def run_vanilla_methods(key, model, N_points, y_obs):
    """Run vanilla ABC methods for baseline comparison."""
    print("Running Vanilla methods...")
    
    # Vanilla ABC
    model.reset_weights_distance()
    time_0 = time.time()
    key, key_theta, key_data = random.split(key, 3)
    thetas = model.prior_generator(key_theta, N_points)
    zs = model.data_generator(key_data, thetas)
    dists = model.distance(zs, y_obs)
    time_van = time.time() - time_0
    print(f"Time for vanilla ABC: {time_van:.2f}s")
    
    # permABC Vanilla
    model.reset_weights_distance()
    time_0 = time.time()
    key, key_theta, key_data = random.split(key, 3)
    thetas = model.prior_generator(key_theta, N_points)
    zs = model.data_generator(key_data, thetas)
    dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs)
    time_perm_van = time.time() - time_0
    print(f"Time for permABC Vanilla: {time_perm_van:.2f}s")
    
    return {
        'vanilla_abc': {'dists': dists, 'time': time_van, 'thetas': thetas, 'zs': zs},
        'vanilla_perm': {'dists': dists_perm, 'time': time_perm_van, 'thetas': thetas, 'zs': zs, 
                        'ys_index': ys_index, 'zs_index': zs_index}
    }


def run_smc_methods(key, model, N_particles, y_obs, stopping_rate, N_points, K):
    """Run SMC-based ABC methods."""
    kernel = KernelTruncatedRW
    results = {}
    
    # ABC SMC
    print("Running ABC SMC...")
    key, subkey = random.split(key)
    model.reset_weights_distance()
    out_smc = abc_smc(
        key=subkey, model=model, n_particles=N_particles, epsilon_target=0, y_obs=y_obs, 
        kernel=kernel, verbose=1, Final_iteration=0, update_weights_distance=False, 
        stopping_accept_rate=stopping_rate, N_sim_max=N_points*K
    )
    results['abc_smc'] = out_smc
    
    # ABC PMC
    print("Running ABC PMC...")
    key, subkey = random.split(key)
    model.reset_weights_distance()
    out_pmc = abc_pmc(
        key=subkey, model=model, n_particles=N_particles, epsilon_target=0, y_obs=y_obs, 
        alpha=0.95, verbose=1, update_weights_distance=False, 
        stopping_accept_rate=stopping_rate, N_sim_max=N_points*K
    )
    results['abc_pmc'] = out_pmc
    
    # permABC SMC
    print("Running permABC SMC...")
    key, subkey = random.split(key)
    model.reset_weights_distance()
    out_perm_smc = perm_abc_smc(
        key=subkey, model=model, n_particles=N_particles, epsilon_target=0, y_obs=y_obs, 
        kernel=kernel, verbose=1, Final_iteration=0, update_weights_distance=False, 
        stopping_accept_rate=stopping_rate, N_sim_max=N_points*K
    )
    results['perm_abc_smc'] = out_perm_smc
    
    return results


def run_osum_methods(key, model, N_particles, y_obs, K):
    """Run over-sampling and under-matching methods."""
    print("Running OSUM methods...")
    
    alpha_epsilon = 0.95
    alpha_M = 0.9
    alpha_L = 0.9
    N_epsilon = 10000
    kernel = KernelTruncatedRW
    
    results = {'over_sampling': {}, 'under_matching': {}}
    
    # Over-sampling
    M0s = np.array([1.5*K, 2*K, 5*K, 7*K, 10*K, 15*K, 20*K, 25*K], dtype=int)
    os_results = {'epsilons': [], 'time': [], 'n_sim': [], 'unique': [], 'full_results': []}
    
    for M0 in M0s:
        print(f"  Over-sampling M0 = {M0}")
        try:
            # Estimate epsilon
            key, subkey = random.split(key)
            thetas = model.prior_generator(subkey, N_epsilon, M0)
            key, subkey = random.split(key)
            zs = model.data_generator(subkey, thetas)
            dists_perm, ys_index, zs_index, _ = optimal_index_distance(
                model=model, zs=zs, y_obs=y_obs, epsilon=0, verbose=0, M=M0
            )
            epsilon = np.quantile(dists_perm, alpha_epsilon)
            
            # Run over-sampling
            model.reset_weights_distance()
            key, subkey = random.split(key)
            out_os = perm_abc_smc_os(
                key=subkey, model=model, n_particles=N_particles, y_obs=y_obs, 
                kernel=kernel, M_0=M0, epsilon=epsilon, alpha_M=alpha_M, 
                update_weights_distance=False, verbose=1, Final_iteration=0, duplicate=True
            )
            
            if out_os is not None:
                os_results['epsilons'].append(epsilon)
                os_results['n_sim'].append(np.sum(out_os["N_sim"]))
                os_results['unique'].append(out_os["unique_part"][-1])
                os_results['time'].append(out_os["time_final"])
                os_results['full_results'].append(out_os)
        except Exception as e:
            print(f"    Failed: {e}")
            os_results['full_results'].append(None)
            continue
    
    # Under-matching
    L0s = np.array(np.linspace(2, K, K), dtype=int)
    um_results = {'epsilons': [], 'time': [], 'n_sim': [], 'unique': [], 'full_results': []}
    
    for L0 in L0s:
        print(f"  Under-matching L0 = {L0}")
        try:
            # Estimate epsilon
            key, subkey = random.split(key)
            thetas = model.prior_generator(subkey, N_epsilon)
            key, subkey = random.split(key)
            zs = model.data_generator(subkey, thetas)
            dists_perm, ys_index, zs_index, _ = optimal_index_distance(
                model=model, zs=zs, y_obs=y_obs, epsilon=0, verbose=0, L=L0
            )
            epsilon = np.quantile(dists_perm, alpha_epsilon)
            
            # Run under-matching
            key, subkey = random.split(key)
            model.reset_weights_distance()
            out_um = perm_abc_smc_um(
                key=subkey, model=model, n_particles=N_particles, y_obs=y_obs, 
                kernel=kernel, L_0=L0, epsilon=epsilon, alpha_L=alpha_L, 
                update_weights_distance=False, verbose=1, Final_iteration=0
            )
            
            if out_um is not None:
                um_results['epsilons'].append(out_um["Eps_values"][-1])
                um_results['n_sim'].append(np.sum(out_um["N_sim"]))
                um_results['unique'].append(out_um["unique_part"][-1])
                um_results['time'].append(out_um["time_final"])
                um_results['full_results'].append(out_um)
        except Exception as e:
            print(f"    Failed: {e}")
            um_results['full_results'].append(None)
            continue
    
    # Convert to arrays and store
    for key_name in ['epsilons', 'time', 'n_sim', 'unique']:
        os_results[key_name] = np.array(os_results[key_name])
        um_results[key_name] = np.array(um_results[key_name])
    
    results['over_sampling'] = {'M0s': M0s, 'results': os_results}
    results['under_matching'] = {'L0s': L0s, 'results': um_results}
    
    return results


def process_results(vanilla_results, smc_results, osum_results, K, N_particles, include_osum=True):
    """Process all results for comparison."""
    N_sample = 1000
    
    # Process vanilla results
    dists = vanilla_results['vanilla_abc']['dists']
    dists_perm = vanilla_results['vanilla_perm']['dists']
    time_van = vanilla_results['vanilla_abc']['time']
    time_perm_van = vanilla_results['vanilla_perm']['time']
    N_points = len(dists)
    
    time_by_sim_van = time_van / N_points
    time_by_sim_perm_van = time_perm_van / N_points
    alphas = np.logspace(0, -3, 10)
    n_sim_van = 1 / alphas * N_sample
    n_sim_perm_van = 1 / alphas * N_sample
    
    processed_results = {
        'method': ['ABC-Vanilla'] * len(alphas) + ['permABC-Vanilla'] * len(alphas),
        'n_sim': np.concatenate([n_sim_van, n_sim_perm_van]),
        'time': np.concatenate([n_sim_van * time_by_sim_van, n_sim_perm_van * time_by_sim_perm_van]),
        'epsilon': np.concatenate([np.quantile(dists, alphas), np.quantile(dists_perm, alphas)])
    }
    
    # Process SMC results
    smc_data = []
    for name_key, display_name in [('abc_smc', 'ABC-SMC'), ('abc_pmc', 'ABC-PMC'), ('perm_abc_smc', 'permABC-SMC')]:
        if name_key in smc_results and smc_results[name_key] is not None:
            out = smc_results[name_key]
            n_sim = np.cumsum(out["N_sim"][1:])
            epsilons = np.array(out["Eps_values"])[1:]
            unique = np.array(out["unique_part"])[1:]
            time_vals = np.cumsum(out["Time"][1:])
            
            # Apply exponential spacing
            indices = select_indices(list(range(len(n_sim))), 10, facteur=2.0)
            
            # Normalize by unique particles
            n_sim_unique = n_sim[indices] / (K * unique[indices] * N_particles) * N_sample
            time_unique = time_vals[indices] / (K * unique[indices] * N_particles) * N_sample
            
            for i in indices:
                smc_data.append({
                    'method': display_name,
                    'n_sim': n_sim_unique[list(indices).index(i)],
                    'time': time_unique[list(indices).index(i)],
                    'epsilon': epsilons[i]
                })
    
    # Process OSUM results
    osum_data = []
    if include_osum and osum_results:
        # Over-sampling
        os_results = osum_results['over_sampling']['results']
        if len(os_results['unique']) > 0:
            os_n_sim = os_results['n_sim'] / (K * os_results['unique'] * N_particles) * N_sample
            os_time = os_results['time'] / (K * os_results['unique'] * N_particles) * N_sample
            
            for i in range(len(os_results['epsilons'])):
                osum_data.append({
                    'method': 'permABC-SMC-OS',
                    'n_sim': os_n_sim[i],
                    'time': os_time[i],
                    'epsilon': os_results['epsilons'][i]
                })
        
        # Under-matching
        um_results = osum_results['under_matching']['results']
        if len(um_results['unique']) > 0:
            um_n_sim = um_results['n_sim'] / (K * um_results['unique'] * N_particles) * N_sample
            um_time = um_results['time'] / (K * um_results['unique'] * N_particles) * N_sample
            
            for i in range(len(um_results['epsilons'])):
                osum_data.append({
                    'method': 'permABC-SMC-UM',
                    'n_sim': um_n_sim[i],
                    'time': um_time[i],
                    'epsilon': um_results['epsilons'][i]
                })
    
    # Combine all results
    all_results = []
    
    # Add vanilla results
    for i in range(len(processed_results['method'])):
        all_results.append({
            'method': processed_results['method'][i],
            'n_sim': processed_results['n_sim'][i],
            'time': processed_results['time'][i],
            'epsilon': processed_results['epsilon'][i]
        })
    
    # Add SMC results
    all_results.extend(smc_data)
    
    # Add OSUM results
    all_results.extend(osum_data)
    
    return pd.DataFrame(all_results)


def get_plot_style():
    """Get consistent plot styling for all performance plots."""
    # Color mapping
    colors = {
        'ABC-Vanilla': '#ff7f0e',
        'permABC-Vanilla': '#1f77b4', 
        'ABC-SMC': '#ff7f0e',
        'ABC-PMC': '#ff7f0e',
        'permABC-SMC': '#1f77b4',
        'permABC-SMC-OS': '#2ca02c',
        'permABC-SMC-UM': '#2ca02c'
    }
    
    # Marker mapping
    markers = {
        'ABC-Vanilla': 's',
        'permABC-Vanilla': 's',
        'ABC-SMC': 'o', 
        'ABC-PMC': 'o',
        'permABC-SMC': 'o',
        'permABC-SMC-OS': '^',
        'permABC-SMC-UM': '^'
    }
    
    # Line style mapping
    linestyles = {
        'ABC-Vanilla': '-',
        'permABC-Vanilla': '-',
        'ABC-SMC': '--',
        'ABC-PMC': '--', 
        'permABC-SMC': '--',
        'permABC-SMC-OS': '--',
        'permABC-SMC-UM': ':'
    }
    
    return colors, markers, linestyles


def create_nsim_plot(df, K, K_outliers, include_osum):
    """Create simulation efficiency plot."""
    colors, markers, linestyles = get_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot simulation efficiency
    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('epsilon')
        ax.plot(method_data['n_sim'], method_data['epsilon'], 
                label=method, color=colors.get(method, 'gray'),
                marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                markersize=8, linewidth=2)
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations per 1000 unique particles", fontsize=12)
    ax.set_ylabel("ε", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Simulation Efficiency (K={K}, Outliers={K_outliers})", fontsize=14)
    
    plt.tight_layout()
    return fig


def create_time_plot(df, K, K_outliers, include_osum):
    """Create time efficiency plot."""
    colors, markers, linestyles = get_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot time efficiency
    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('epsilon')
        ax.plot(method_data['time'], method_data['epsilon'], 
                label=method, color=colors.get(method, 'gray'),
                marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                markersize=8, linewidth=2)
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Time per 1000 unique particles (seconds)", fontsize=12)
    ax.set_ylabel("ε", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Time Efficiency (K={K}, Outliers={K_outliers})", fontsize=14)
    
    plt.tight_layout()
    return fig


def create_combined_plot(df, K, K_outliers, include_osum):
    """Create combined simulation and time efficiency plots."""
    colors, markers, linestyles = get_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Number of simulations
    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('epsilon')
        ax1.plot(method_data['n_sim'], method_data['epsilon'], 
                label=method, color=colors.get(method, 'gray'),
                marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                markersize=6, linewidth=2)
    
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of simulations per 1000 unique particles", fontsize=12)
    ax1.set_ylabel("ε", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"(a) Simulation Efficiency (K={K}, Outliers={K_outliers})", fontsize=12)
    
    # Plot 2: Time
    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('epsilon')
        ax2.plot(method_data['time'], method_data['epsilon'], 
                label=method, color=colors.get(method, 'gray'),
                marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                markersize=6, linewidth=2)
    
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("Time per 1000 unique particles (seconds)", fontsize=12)
    ax2.set_ylabel("ε", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"(b) Time Efficiency (K={K}, Outliers={K_outliers})", fontsize=12)
    
    plt.tight_layout()
    return fig


def save_results(df, vanilla_results, smc_results, osum_results, model, y_obs, true_theta, 
                K, K_outliers, include_osum, seed, N_particles, output_dir, plot_type='both'):
    """Save results to both pickle and CSV formats."""
    # Create output directories
    results_dir = os.path.join(output_dir, "results", "performance_comparison")
    figures_dir = os.path.join(output_dir, "figures", "fig4" if not include_osum else "fig6")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Add metadata to dataframe
    df['K'] = K
    df['K_outliers'] = K_outliers
    df['include_osum'] = include_osum
    df['seed'] = seed
    
    # Prepare comprehensive data for pickle
    full_data = {
        'seed': seed,
        'K': K,
        'K_outliers': K_outliers,
        'include_osum': include_osum,
        'N_particles': N_particles,
        'plot_type': plot_type,
        'summary_df': df,
        'results': {
            'vanilla': vanilla_results,
            'smc': smc_results,
            'osum': osum_results if include_osum else None
        },
        'experiment_setup': {
            'model': model,
            'y_obs': y_obs,
            'true_theta': true_theta
        },
        'metadata': {
            'created_with': 'run_performance_comparison.py',
            'analysis_type': 'performance_comparison',
            'methods_included': list(df['method'].unique()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save comprehensive pickle file
    base_filename = f"performance_K_{K}_outliers_{K_outliers}_osum_{include_osum}_seed_{seed}"
    pkl_path = os.path.join(results_dir, f"{base_filename}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(full_data, f)
    print(f"Full results saved to pickle: {pkl_path}")
    
    # Save summary to CSV for backward compatibility
    csv_path = os.path.join(results_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary saved to CSV: {csv_path}")
    
    # Create and save plots based on plot_type
    fig_num = "fig4" if not include_osum else "fig6"
    
    if plot_type == 'nsim':
        # Only simulation efficiency plot
        fig = create_nsim_plot(df, K, K_outliers, include_osum)
        base_name = f"{fig_num}_nsim_seed_{seed}"
        fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Simulation efficiency plot saved: {figures_dir}/{base_name}.*")
        
    elif plot_type == 'time':
        # Only time efficiency plot
        fig = create_time_plot(df, K, K_outliers, include_osum)
        base_name = f"{fig_num}_time_seed_{seed}"
        fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Time efficiency plot saved: {figures_dir}/{base_name}.*")
        
    elif plot_type == 'both':
        # Both plots - separate files
        # 1. Simulation efficiency
        fig_nsim = create_nsim_plot(df, K, K_outliers, include_osum)
        base_name_nsim = f"{fig_num}_nsim_seed_{seed}"
        fig_nsim.savefig(os.path.join(figures_dir, f"{base_name_nsim}.pdf"), 
                        dpi=300, bbox_inches='tight')
        fig_nsim.savefig(os.path.join(figures_dir, f"{base_name_nsim}.png"), 
                        dpi=300, bbox_inches='tight')
        plt.close(fig_nsim)
        
        # 2. Time efficiency
        fig_time = create_time_plot(df, K, K_outliers, include_osum)
        base_name_time = f"{fig_num}_time_seed_{seed}"
        fig_time.savefig(os.path.join(figures_dir, f"{base_name_time}.pdf"), 
                        dpi=300, bbox_inches='tight')
        fig_time.savefig(os.path.join(figures_dir, f"{base_name_time}.png"), 
                        dpi=300, bbox_inches='tight')
        plt.close(fig_time)
        
        # 3. Combined plot (legacy)
        fig_combined = create_combined_plot(df, K, K_outliers, include_osum)
        base_name_combined = f"{fig_num}_combined_seed_{seed}"
        fig_combined.savefig(os.path.join(figures_dir, f"{base_name_combined}.pdf"), 
                            dpi=300, bbox_inches='tight')
        fig_combined.savefig(os.path.join(figures_dir, f"{base_name_combined}.png"), 
                            dpi=300, bbox_inches='tight')
        plt.close(fig_combined)
        
        print(f"All plots saved: {figures_dir}/{fig_num}_*_seed_{seed}.*")
    
    return pkl_path, csv_path


def analyze_performance_metrics(df):
    """Analyze and print performance metrics."""
    print("\nPerformance Analysis:")
    print("=" * 50)
    
    # Group by method and show statistics
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        print(f"\n{method}:")
        print(f"  Data points: {len(method_data)}")
        print(f"  Epsilon range: [{method_data['epsilon'].min():.2e}, {method_data['epsilon'].max():.2e}]")
        print(f"  Simulation efficiency range: [{method_data['n_sim'].min():.0f}, {method_data['n_sim'].max():.0f}]")
        print(f"  Time efficiency range: [{method_data['time'].min():.2f}, {method_data['time'].max():.2f}]s")
    
    # Compare methods at similar epsilon levels
    print(f"\nComparison at median epsilon levels:")
    methods = df['method'].unique()
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            median_idx = len(method_data) // 2
            sorted_data = method_data.sort_values('epsilon')
            median_row = sorted_data.iloc[median_idx]
            print(f"  {method}: ε={median_row['epsilon']:.2e}, "
                  f"sims={median_row['n_sim']:.0f}, time={median_row['time']:.2f}s")


def rerun_from_file(file_path, plot_type='both'):
    """Recreate plots from existing pickle or CSV results."""
    print(f"Loading results from: {file_path}")
    
    # Determine file type and load accordingly
    if file_path.endswith('.pkl'):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        df = data['summary_df']
        K = data['K']
        K_outliers = data['K_outliers']
        include_osum = data['include_osum']
        seed = data['seed']
        
        print(f"Loaded pickle with full results")
        # Analyze full results if available
        if 'results' in data:
            print(f"Available result types: {list(data['results'].keys())}")
            
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        K = df['K'].iloc[0] if 'K' in df.columns else 20
        K_outliers = df['K_outliers'].iloc[0] if 'K_outliers' in df.columns else 4
        include_osum = df['include_osum'].iloc[0] if 'include_osum' in df.columns else True
        seed = df['seed'].iloc[0] if 'seed' in df.columns else 42
        
        print(f"Loaded CSV with summary data only")
    else:
        print("Unsupported file type. Use .pkl or .csv files.")
        return
    
    print(f"Recreating plots for K={K}, K_outliers={K_outliers}, osum={include_osum}, seed={seed}")
    print(f"Plot type: {plot_type}")
    
    # Analyze performance
    analyze_performance_metrics(df)
    
    # Create plots based on plot_type
    base_dir = os.path.dirname(file_path)
    fig_num = "fig4" if not include_osum else "fig6"
    
    if plot_type == 'nsim':
        fig = create_nsim_plot(df, K, K_outliers, include_osum)
        base_name = f"{fig_num}_nsim_seed_{seed}_rerun"
        fig.savefig(os.path.join(base_dir, f"{base_name}.pdf"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(base_dir, f"{base_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Simulation efficiency plot recreated: {base_dir}/{base_name}.*")
        
    elif plot_type == 'time':
        fig = create_time_plot(df, K, K_outliers, include_osum)
        base_name = f"{fig_num}_time_seed_{seed}_rerun"
        fig.savefig(os.path.join(base_dir, f"{base_name}.pdf"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(base_dir, f"{base_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Time efficiency plot recreated: {base_dir}/{base_name}.*")
        
    elif plot_type == 'both':
        # Create both separate plots
        fig_nsim = create_nsim_plot(df, K, K_outliers, include_osum)
        base_name_nsim = f"{fig_num}_nsim_seed_{seed}_rerun"
        fig_nsim.savefig(os.path.join(base_dir, f"{base_name_nsim}.pdf"), dpi=300, bbox_inches='tight')
        fig_nsim.savefig(os.path.join(base_dir, f"{base_name_nsim}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_nsim)
        
        fig_time = create_time_plot(df, K, K_outliers, include_osum)
        base_name_time = f"{fig_num}_time_seed_{seed}_rerun"
        fig_time.savefig(os.path.join(base_dir, f"{base_name_time}.pdf"), dpi=300, bbox_inches='tight')
        fig_time.savefig(os.path.join(base_dir, f"{base_name_time}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_time)
        
        # Also create combined plot
        fig_combined = create_combined_plot(df, K, K_outliers, include_osum)
        base_name_combined = f"{fig_num}_combined_seed_{seed}_rerun"
        fig_combined.savefig(os.path.join(base_dir, f"{base_name_combined}.pdf"), dpi=300, bbox_inches='tight')
        fig_combined.savefig(os.path.join(base_dir, f"{base_name_combined}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_combined)
        
        print(f"All plots recreated: {base_dir}/{fig_num}_*_seed_{seed}_rerun.*")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run performance comparison between ABC methods"
    )
    
    parser.add_argument('--K', type=int, default=20,
                       help='Number of components (default: 20)')
    parser.add_argument('--K_outliers', type=int, default=4,
                       help='Number of outlier components (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--N_points', type=int, default=1000000,
                       help='Number of points for vanilla methods (default: 1,000,000)')
    parser.add_argument('--N_particles', type=int, default=1000,
                       help='Number of particles for SMC methods (default: 1,000)')
    parser.add_argument('--osum', dest='include_osum', action='store_true',
                       help='Include over-sampling and under-matching methods')
    parser.add_argument('--no-osum', dest='include_osum', action='store_false',
                       help='Exclude over-sampling and under-matching methods')
    parser.add_argument('--plot', type=str, choices=['nsim', 'time', 'both'], default='both',
                       help='Type of plots to generate: nsim (simulations), time, or both (default: both)')
    parser.add_argument('--output-dir', type=str, default="experiments",
                       help='Output directory (default: experiments)')
    parser.add_argument('--rerun', type=str, default=None,
                       help='Path to pickle or CSV file for rerunning analysis')
    
    parser.set_defaults(include_osum=True)
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Handle rerun case
    if args.rerun:
        rerun_from_file(args.rerun, args.plot)
        return
    
    print("Performance comparison between ABC methods")
    print(f"Parameters: K={args.K}, K_outliers={args.K_outliers}, seed={args.seed}")
    print(f"N_points={args.N_points:,}, N_particles={args.N_particles:,}")
    print(f"Include OSUM methods: {args.include_osum}")
    print(f"Plot type: {args.plot}")
    
    # Setup experiment
    model, y_obs, true_theta, key, N_points, N_particles, stopping_rate, K = setup_experiment(
        K=args.K, K_outliers=args.K_outliers, seed=args.seed, 
        N_points=args.N_points, N_particles=args.N_particles
    )
    
    # Run vanilla methods
    vanilla_results = run_vanilla_methods(key, model, N_points, y_obs)
    
    # Run SMC methods
    smc_results = run_smc_methods(
        key, model, N_particles, y_obs, stopping_rate, N_points, K
    )
    
    # Run OSUM methods if requested
    if args.include_osum:
        osum_results = run_osum_methods(key, model, N_particles, y_obs, K)
    else:
        osum_results = None
    
    # Process results
    df = process_results(vanilla_results, smc_results, osum_results, K, N_particles, args.include_osum)
    
    # Analyze performance
    analyze_performance_metrics(df)
    
    # Save results and create plots
    pkl_path, csv_path = save_results(
        df, vanilla_results, smc_results, osum_results, model, y_obs, true_theta,
        args.K, args.K_outliers, args.include_osum, args.seed, N_particles, args.output_dir, args.plot
    )
    
    print("Performance comparison complete!")
    print(f"Full results: {pkl_path}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()