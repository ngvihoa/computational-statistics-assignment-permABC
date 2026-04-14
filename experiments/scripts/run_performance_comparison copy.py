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
from pathlib import Path


from permabc.algorithms.smc import abc_smc, perm_abc_smc
from permabc.algorithms.pmc import abc_pmc
from permabc.algorithms.over_sampling import perm_abc_smc_os
from permabc.algorithms.under_matching import perm_abc_smc_um
from permabc.core.kernels import KernelTruncatedRW
from permabc.core.distances import optimal_index_distance
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta


FACTOR = 1.  # Exponential spacing factor for indices selection
def select_indices(L, m, factor=FACTOR):
    """Extract m elements with exponential spacing."""
    n = len(L)
    if m >= n:
        return list(range(n))
    if m <= 0:
        return []
    
    indices = []
    for i in range(m):
        t = i / (m - 1) if m > 1 else 0
        if factor != 1:
            mapped_t = (factor ** t - 1) / (factor - 1)
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
    
    results = {
        'vanilla_abc': {'dists': dists, 'time': time_van, 'thetas': thetas, 'zs': zs},
        'vanilla_perm': {'dists': dists_perm, 'time': time_perm_van, 'thetas': thetas, 'zs': zs, 
                        'ys_index': ys_index, 'zs_index': zs_index}
    }
    return key, results


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
    
    return key, results


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
    
    return key, results


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
            
            # --- FIX: Loop over all points, no down-sampling here ---
            for i in range(len(n_sim)):
                # Normalization is now applied to every point
                n_sim_unique = n_sim[i] / (K * unique[i] * N_particles) * N_sample
                time_unique = time_vals[i] / (K * unique[i] * N_particles) * N_sample
                
                smc_data.append({
                    'method': display_name,
                    'n_sim': n_sim_unique,
                    'time': time_unique,
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
    
    # Couleurs ajustées pour correspondre au graphique
    colors = {
        'ABC-Vanilla': '#d62728',      # Rouge
        'permABC-Vanilla': '#2ca02c',  # Vert
        'ABC-SMC': '#ff7f0e',          # Orange foncé
        'ABC-PMC': '#ffbb78',          # Orange clair
        'permABC-SMC': '#1f77b4',      # Bleu
        'permABC-SMC-OS': '#e377c2',   # Rose
        'permABC-SMC-UM': '#9467bd'    # Violet
    }
    
    # Marqueurs correspondant au graphique
    markers = {
        'ABC-Vanilla': 's',       # Carré
        'permABC-Vanilla': 's',   # Carré
        'ABC-SMC': 'o',           # Cercle
        'ABC-PMC': 'o',           # Cercle
        'permABC-SMC': 'o',       # Cercle
        'permABC-SMC-OS': '^',    # Triangle
        'permABC-SMC-UM': '^'     # Triangle
    }
    
    # Styles de ligne correspondant au graphique
    linestyles = {
        'ABC-Vanilla': '-',       # Solide
        'permABC-Vanilla': '-',   # Solide
        'ABC-SMC': '--',          # Tirets
        'ABC-PMC': '--',          # Tirets
        'permABC-SMC': '--',      # Tirets
        'permABC-SMC-OS': '--',   # Tirets
        'permABC-SMC-UM': '--'    # Tirets (modifié de ':' à '--')
    }
    
    return colors, markers, linestyles




def create_nsim_plot(df, K, K_outliers, include_osum, ax = None):
    """Create simulation efficiency plot."""
    colors, markers, linestyles = get_plot_style()
    
    # Define the sampling factors for each SMC/PMC method
    smc_factors = {
        "permABC-SMC": 0.0001,
        "ABC-SMC": 0.0001,
        "ABC-PMC": 0.00001
    }
    
    len_indices = {
        "permABC-SMC": 10,
        "ABC-SMC": 10,
        "ABC-PMC": 10,
    }
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.get_figure()
    
    # Plot simulation efficiency
    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('epsilon')
        
        # Apply subsampling only to SMC and PMC methods for clarity
        if 'SMC' in method or 'PMC' in method:
            method_data = method_data[method_data['n_sim'] <= 1e6] # Increased limit
            
            # Get the specific factor for the method, or use a default
            factor = smc_factors.get(method, 2.0)
            length_indices = len_indices.get(method, 10)
            indices = select_indices(list(range(len(method_data))), length_indices, factor=factor)

            # Use .iloc to apply the indices
            ax.plot(method_data['n_sim'].iloc[indices], method_data['epsilon'].iloc[indices], 
                    label=method, color=colors.get(method, 'gray'),
                    marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                    markersize=8, linewidth=2)
        else:
            # Plot all points for other methods (e.g., Vanilla)
            ax.plot(method_data['n_sim'], method_data['epsilon'], 
                    label=method, color=colors.get(method, 'gray'),
                    marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                    markersize=8, linewidth=2)
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations per 1000 unique particles", fontsize=12)
    ax.set_ylabel("$\\varepsilon$", fontsize=12)
    ax.legend(fontsize=12)
    #ax.grid(False, which="both", ls="--", alpha=0.3)
    # ax.set_title(f"Simulation Efficiency (K={K}, Outliers={K_outliers})", fontsize=14)
    
    plt.tight_layout()
    return fig

def create_time_plot(df, K, K_outliers, include_osum, ax = None):
    """Create time efficiency plot."""
    colors, markers, linestyles = get_plot_style()

    # Define the sampling factors for each SMC/PMC method
    smc_factors = {
        "permABC-SMC": 0.0001,
        "ABC-SMC": 0.000001,
        "ABC-PMC": 0.00001
    }
    
    len_indices = {
        "permABC-SMC": 10,
        "ABC-SMC": 10,
        "ABC-PMC": 10,
    }
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.get_figure()
    
    # Plot time efficiency
    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('epsilon')
        
        # Apply subsampling only to SMC and PMC methods for clarity
        if 'SMC' in method or 'PMC' in method:
            method_data = method_data[method_data['n_sim'] <= 1e6] # Increased limit
            # Get the specific factor for the method, or use a default
            factor = smc_factors.get(method, 2.0)
            length_indices = len_indices.get(method, 10)
            indices = select_indices(list(range(len(method_data))), length_indices, factor=factor)

            # Use .iloc to apply the indices
            ax.plot(method_data['time'].iloc[indices], method_data['epsilon'].iloc[indices], 
                    label=method, color=colors.get(method, 'gray'),
                    marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                    markersize=8, linewidth=2)
        else:
            # Plot all points for other methods (e.g., Vanilla)
            ax.plot(method_data['time'], method_data['epsilon'], 
                    label=method, color=colors.get(method, 'gray'),
                    marker=markers.get(method, 'o'), linestyle=linestyles.get(method, '-'),
                    markersize=8, linewidth=2)
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Time per 1000 unique particles (seconds)", fontsize=12)
    ax.set_ylabel("$\\varepsilon$", fontsize=12)
    ax.legend(fontsize=12)
    #ax.grid(False, which="both", ls="--", alpha=0.3)
    # ax.set_title(f"Time Efficiency (K={K}, Outliers={K_outliers})", fontsize=14)
    
    plt.tight_layout()
    return fig


def create_combined_plot(df, K, K_outliers, include_osum):
    """Create combined simulation and time efficiency plots by calling individual plot functions."""
    
    # 1. Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    # 2. Call the simulation plot function on the first subplot
    create_nsim_plot(df, K, K_outliers, include_osum, ax=ax1)
    
    # 3. Call the time plot function on the second subplot
    create_time_plot(df, K, K_outliers, include_osum, ax=ax2)
    
    # 4. Final layout adjustment
    plt.tight_layout()
    
    return fig



import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
import time
from jax import random


def save_results(df, vanilla_results, smc_results, osum_results, model, y_obs, true_theta,
                K, K_outliers, include_osum, seed, N_particles, output_dir, plot_type='both'):
    """Save results to both pickle and CSV formats."""
    # Create output directories
    results_dir = os.path.join(output_dir, "experiments", "results", "performance_comparison")
    
    # Correctly determine figure directory based on the 'include_osum' flag for this specific run
    fig_num_str = "fig4" if not include_osum else "fig6"
    figures_dir = os.path.join(output_dir, "figures", fig_num_str)

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
        'include_osum': include_osum, # This reflects the data generation, not necessarily the plotting intent
        'N_particles': N_particles,
        'plot_type': plot_type,
        'summary_df': df,
        'results': {
            'vanilla': vanilla_results,
            'smc': smc_results,
            'osum': osum_results
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

    # Save comprehensive pickle file only if we are not in rerun mode from another file
    # This prevents overwriting the fig6 data when just creating a fig4 plot from it.
    base_filename = f"performance_K_{K}_outliers_{K_outliers}_osum_{include_osum}_seed_{seed}"
    pkl_path = os.path.join(results_dir, f"{base_filename}.pkl")
    if not os.path.exists(pkl_path): # Only save if it's a fresh run
        with open(pkl_path, "wb") as f:
            pickle.dump(full_data, f)
        print(f"Full results saved to pickle: {pkl_path}")

    # Save summary to CSV for backward compatibility
    csv_path = os.path.join(results_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary saved to CSV: {csv_path}")

    # Create and save plots based on plot_type
    fig_num = "fig4" if not include_osum else "fig6"

    # Create plots
    fig_nsim = create_nsim_plot(df, K, K_outliers, include_osum)
    base_name_nsim = f"{fig_num}_nsim_seed_{seed}"
    fig_nsim.savefig(os.path.join(figures_dir, f"{base_name_nsim}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_nsim)
    
    fig_time = create_time_plot(df, K, K_outliers, include_osum)
    base_name_time = f"{fig_num}_time_seed_{seed}"
    fig_time.savefig(os.path.join(figures_dir, f"{base_name_time}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_time)

    print(f"Plots saved in: {figures_dir}")

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


def rerun_from_file(file_path, plot_type='both', osum_flag_for_plot=False, output_dir_override=None):
    """Recreate plots from existing pickle file, respecting current run flags."""
    print(f"Loading results from: {file_path}")

    if not file_path.endswith('.pkl'):
        print("Error: Rerun requires a .pkl file to ensure all data is available.")
        return

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    df = data['summary_df']
    K = data['K']
    K_outliers = data['K_outliers']
    seed = data['seed']
    if not osum_flag_for_plot:
        print("Filtering out OSUM methods for Figure 4 generation...")
        methods_to_exclude = ['permABC-SMC-OS', 'permABC-SMC-UM']
        df_filtered = df[~df['method'].isin(methods_to_exclude)].copy()
    else:
        df_filtered = df
    # Determine the figure number based on the current command, not the loaded file.
    fig_num_str = "fig4" if not osum_flag_for_plot else "fig6"
    print(f"Recreating {fig_num_str} for K={K}, K_outliers={K_outliers}, seed={seed}")
    
    # Determine the output directory
    if output_dir_override:
        figures_dir = os.path.join(output_dir_override, "figures", fig_num_str)
    else:
        # Fallback to the directory of the pkl file
        figures_dir = os.path.join(os.path.dirname(file_path), "..", "..", "figures", fig_num_str)
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create the plots
    fig_nsim = create_nsim_plot(df_filtered, K, K_outliers, osum_flag_for_plot)
    fig_nsim.savefig(os.path.join(figures_dir, f"{fig_num_str}_nsim_seed_{seed}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_nsim)
    
    fig_time = create_time_plot(df_filtered, K, K_outliers, osum_flag_for_plot)
    fig_time.savefig(os.path.join(figures_dir, f"{fig_num_str}_time_seed_{seed}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_time)
    
    print(f"Plots recreated in: {figures_dir}")


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
    
    if args.rerun:
        # Pass the current command's flags to the rerun function
        rerun_from_file(args.rerun, args.plot, args.include_osum, args.output_dir)
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
    key, vanilla_results = run_vanilla_methods(key, model, N_points, y_obs)
    
    # Run SMC methods
    key, smc_results = run_smc_methods(
        key, model, N_particles, y_obs, stopping_rate, N_points, K
    )
    
    # Run OSUM methods if requested
    if args.include_osum:
        key, osum_results = run_osum_methods(key, model, N_particles, y_obs, K)
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