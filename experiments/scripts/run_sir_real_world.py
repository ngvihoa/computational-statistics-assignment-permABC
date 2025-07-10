#!/usr/bin/env python3
"""
SIR real world analysis for COVID-19 data at multiple scales.

This script runs permABC-SMC inference on real COVID-19 hospitalization data
at national, regional, and departmental scales in France.

Usage:
    python run_sir_real_world.py
    python run_sir_real_world.py --seed 42 --n_particles 1000
    python run_sir_real_world.py --rerun experiments/results/sir_real_world_seed_42.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse
from jax import random


from permabc.algorithms.smc import abc_smc, perm_abc_smc
from permabc.core.kernels import KernelTruncatedRW
from permabc.models.SIR import SIR_real_world


def smooth_moving_average(x, window=7):
    """Apply a centered moving average with padding to preserve length."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def load_population_data():
    """Load and process French population data."""
    print("Loading population data...")
    
    # Try to load from local file first, then from URL
    try:
        df_pop = pd.read_csv("experiments/data/donnees_departements.csv", sep=";")
    except FileNotFoundError:
        print("Local population data not found. Please ensure data files are available.")
        print("Expected file: experiments/data/donnees_departements.csv")
        raise
    
    # Create dictionaries for population and names
    dep_pop = {df_pop.loc[i]["CODDEP"]: df_pop.loc[i]["PTOT"] for i in range(df_pop.shape[0])}
    dep_name = {df_pop.loc[i]["CODDEP"]: df_pop.loc[i]["DEP"] for i in range(df_pop.shape[0])}
    
    # Regional data
    reg = np.unique(df_pop["CODREG"])
    reg_pop = {}
    reg_name = {}
    for r in reg:
        reg_pop[r] = df_pop[df_pop["CODREG"] == r]["PTOT"].sum()
        reg_name[r] = df_pop[df_pop["CODREG"] == r]["REG"].iloc[0]
    
    dep_reg = {df_pop.loc[i]["CODDEP"]: df_pop.loc[i]["CODREG"] for i in range(df_pop.shape[0])}
    
    return dep_pop, dep_name, reg_pop, reg_name, dep_reg


def load_hospitalization_data(dep_pop, dep_reg, reg_pop):
    """Load and process COVID-19 hospitalization data."""
    print("Loading hospitalization data...")
    
    try:
        df_dep = pd.read_csv("experiments/data/data-dep.csv", sep=";")
    except FileNotFoundError:
        print("Local hospitalization data not found. Please ensure data files are available.")
        print("Expected file: experiments/data/data-dep.csv")
        raise
    
    # Filter data
    df_dep = df_dep.iloc[np.where(df_dep.loc[:]["sexe"] == 0)]
    df_dep = df_dep[~df_dep['dep'].isin(["971", "972", "973", "974", "976", "978", "2A", "2B"])]
    print(f"Department data shape: {df_dep.shape}")
    
    # Parameters
    IHR = 0.036  # Infection Hospitalization Rate
    
    # Process dates
    date = np.unique(np.array(pd.to_datetime(df_dep.iloc[:]["jour"])))
    print(f"Number of days: {len(date)}")
    
    # Add region and population information
    df_dep["reg"] = ""
    df_dep["pop"] = ""
    
    dep_list = np.unique(df_dep["dep"])
    for d in dep_list:
        df_dep.loc[df_dep["dep"] == d, "reg"] = dep_reg[d]
        df_dep.loc[df_dep["dep"] == d, "pop"] = dep_pop[d]
    
    # Process departmental data
    burnin = 0
    I_obs_dep = np.array([
        smooth_moving_average(df_dep.loc[df_dep["dep"] == d]["hosp"], window=7) / dep_pop[d] * 100000 / IHR 
        for d in dep_list
    ])
    
    # Process regional data
    df_reg = df_dep.groupby(['jour', 'reg']).sum().reset_index()
    reg_list = np.unique(df_reg["reg"])
    I_obs_reg = np.array([
        smooth_moving_average(df_reg.loc[df_reg["reg"] == r]["hosp"], window=7) / reg_pop[r] * 100000 / IHR 
        for r in reg_list
    ])
    
    # Process national data
    df_fr = df_dep.groupby(['jour']).sum().reset_index()
    I_obs_fr = np.array(
        smooth_moving_average(df_fr["hosp"], window=7) / df_fr["pop"][0] * 100000 / IHR
    )
    
    print(f"Data shapes - FR: {I_obs_fr.shape}, Regional: {I_obs_reg.shape}, Departmental: {I_obs_dep.shape}")
    
    return I_obs_fr, I_obs_reg, I_obs_dep, date[burnin:], dep_list, reg_list


def focus_first_wave(I_obs_fr, I_obs_reg, I_obs_dep, date, n_day=120, day_0=0):
    """Focus analysis on the first wave of COVID-19."""
    date_V1 = date[day_0:n_day + day_0]
    print(f"First wave period: {date_V1[0]} to {date_V1[-1]}")
    
    I_obs_dep_V1 = I_obs_dep[:, day_0:n_day + day_0]
    I_obs_reg_V1 = I_obs_reg[:, day_0:n_day + day_0]
    I_obs_fr_V1 = I_obs_fr[day_0:n_day + day_0]
    
    print(f"First wave shapes - FR: {I_obs_fr_V1.shape}, Regional: {I_obs_reg_V1.shape}, Departmental: {I_obs_dep_V1.shape}")
    
    return I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, date_V1


def setup_models(I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, reg_list, dep_list, reg_pop, dep_pop, n_day):
    """Setup SIR models for different scales."""
    print("Setting up SIR models...")
    
    # Prior parameters
    low_I, high_I = 0.1, 2000
    low_R, high_R = 0.1, 2000
    low_gamma, high_gamma = 0.05, 4.0
    low_r0, high_r0 = 0.5, 4.0
    n_pop = 100000
    
    # National model
    K_fr = 1
    mod_fr = SIR_real_world(
        K=K_fr, n_obs=n_day, n_pop=n_pop,
        low_I=low_I, high_I=high_I,
        low_R=low_R, high_R=high_R,
        low_gamma=low_gamma, high_gamma=high_gamma,
        low_r0=low_r0, high_r0=high_r0
    )
    y_obs_fr = I_obs_fr_V1[None, None, :]
    
    # Regional model
    K_reg = len(reg_list)
    weights_region = np.array([reg_pop[r] for r in reg_list])
    weights_region = weights_region / np.sum(weights_region)
    
    mod_reg = SIR_real_world(
        K=K_reg, n_obs=n_day, n_pop=n_pop, weights_distance=weights_region,
        low_I=low_I, high_I=high_I,
        low_R=low_R, high_R=high_R,
        low_gamma=low_gamma, high_gamma=high_gamma,
        low_r0=low_r0, high_r0=high_r0
    )
    y_obs_reg = np.array(I_obs_reg_V1)[None, :]
    
    # Departmental model
    K_dep = len(dep_list)
    weights_dep = np.array([dep_pop[d] for d in dep_list])
    weights_dep = weights_dep / np.sum(weights_dep)
    
    mod_dep = SIR_real_world(
        K=K_dep, n_obs=n_day, n_pop=n_pop, weights_distance=weights_dep,
        low_I=low_I, high_I=high_I,
        low_R=low_R, high_R=high_R,
        low_gamma=low_gamma, high_gamma=high_gamma,
        low_r0=low_r0, high_r0=high_r0
    )
    y_obs_dep = np.array(I_obs_dep_V1)[None, :]
    
    return (mod_fr, y_obs_fr, K_fr), (mod_reg, y_obs_reg, K_reg), (mod_dep, y_obs_dep, K_dep)


def run_inference(models, n_particles=1000, seed=42):
    """Run permABC-SMC inference at all scales."""
    print("Running inference...")
    
    key = random.PRNGKey(seed)
    key, key_fr, key_fr_perm, key_reg, key_dep = random.split(key, 5)
    
    (mod_fr, y_obs_fr, K_fr), (mod_reg, y_obs_reg, K_reg), (mod_dep, y_obs_dep, K_dep) = models
    
    results = {}
    
    # National permABC-SMC
    print("Running national permABC-SMC...")
    permsmc_fr = perm_abc_smc(
        key=key_fr_perm,
        model=mod_fr,
        y_obs=y_obs_fr,
        epsilon_target=0.,
        n_particles=n_particles,
        alpha_epsilon=0.95,
        kernel=KernelTruncatedRW,
        num_blocks_gibbs=1,
        both_loc_glob=True,
        stopping_accept_rate_global=0.05,
        Final_iteration=50,
        stopping_epsilon_difference=0.,
        verbose=2
    )
    results['national_perm'] = permsmc_fr
    
    # National ABC-SMC for comparison
    print("Running national ABC-SMC...")
    smc_fr = abc_smc(
        key=key_fr,
        model=mod_fr,
        y_obs=y_obs_fr,
        epsilon_target=0.,
        n_particles=n_particles,
        alpha_epsilon=0.95,
        kernel=KernelTruncatedRW,
        num_blocks_gibbs=1,
        both_loc_glob=True,
        stopping_accept_rate_global=0.05,
        Final_iteration=50,
        stopping_epsilon_difference=0.,
        verbose=2
    )
    results['national_abc'] = smc_fr
    
    # Regional permABC-SMC
    print("Running regional permABC-SMC...")
    permsmc_reg = perm_abc_smc(
        key=key_reg,
        model=mod_reg,
        y_obs=y_obs_reg,
        epsilon_target=0.,
        n_particles=n_particles,
        alpha_epsilon=0.95,
        kernel=KernelTruncatedRW,
        num_blocks_gibbs=4,
        both_loc_glob=True,
        stopping_accept_rate_global=0.05,
        Final_iteration=50,
        stopping_epsilon_difference=0.,
        verbose=2
    )
    results['regional'] = permsmc_reg
    
    # Departmental permABC-SMC
    print("Running departmental permABC-SMC...")
    permsmc_dep = perm_abc_smc(
        key=key_dep,
        model=mod_dep,
        y_obs=y_obs_dep,
        epsilon_target=0.,
        n_particles=n_particles,
        alpha_epsilon=0.95,
        kernel=KernelTruncatedRW,
        num_blocks_gibbs=10,
        both_loc_glob=True,
        stopping_accept_rate_global=0.05,
        Final_iteration=100,
        parallel=False
    )
    results['departmental'] = permsmc_dep
    
    return results


def extract_posterior_summaries(results):
    """Extract posterior summaries for analysis."""
    summaries = []
    
    for scale, result in results.items():
        if result is not None:
            final_thetas = result['Thetas'][-1]
            
            # R0 (global parameter)
            r0_samples = final_thetas.glob[:, 0]
            
            # Local parameters (gamma)
            if final_thetas.loc.shape[1] > 0:
                gamma_samples = final_thetas.loc[:, :, 2]  # Assuming gamma is index 2
                
                for k in range(final_thetas.loc.shape[1]):
                    summaries.append({
                        'scale': scale,
                        'component': k,
                        'parameter': 'gamma',
                        'mean': np.mean(gamma_samples[:, k]),
                        'std': np.std(gamma_samples[:, k]),
                        'q025': np.percentile(gamma_samples[:, k], 2.5),
                        'q975': np.percentile(gamma_samples[:, k], 97.5)
                    })
            
            # R0 summary
            summaries.append({
                'scale': scale,
                'component': 0,
                'parameter': 'R0',
                'mean': np.mean(r0_samples),
                'std': np.std(r0_samples),
                'q025': np.percentile(r0_samples, 2.5),
                'q975': np.percentile(r0_samples, 97.5)
            })
    
    return pd.DataFrame(summaries)


def save_results(results, summary_df, models, data_info, seed, n_particles, n_day, output_dir):
    """Save results to both pickle and CSV formats."""
    # Create output directory
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Add metadata to summary
    summary_df['seed'] = seed
    
    # Prepare comprehensive data for pickle
    full_data = {
        'seed': seed,
        'n_particles': n_particles,
        'n_day': n_day,
        'summary_df': summary_df,
        'results': results,
        'models': models,
        'data_info': data_info,
        'metadata': {
            'created_with': 'run_sir_real_world.py',
            'analysis_type': 'permABC-SMC',
            'scales': list(results.keys()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save comprehensive pickle file
    pkl_path = os.path.join(results_dir, f"sir_real_world_seed_{seed}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(full_data, f)
    print(f"Full results saved to pickle: {pkl_path}")
    
    # Save summary to CSV for backward compatibility
    csv_path = os.path.join(results_dir, f"sir_real_world_seed_{seed}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary saved to CSV: {csv_path}")
    
    return pkl_path, csv_path, full_data


def analyze_convergence(results):
    """Analyze convergence of SMC algorithms."""
    print("\nConvergence Analysis:")
    print("=" * 50)
    
    for scale, result in results.items():
        if result is not None and 'Epsilons' in result:
            epsilons = result['Epsilons']
            n_iterations = len(epsilons)
            final_epsilon = epsilons[-1] if len(epsilons) > 0 else 'N/A'
            
            print(f"{scale.upper()}:")
            print(f"  Iterations: {n_iterations}")
            print(f"  Final epsilon: {final_epsilon}")
            
            if 'accept_rates' in result:
                accept_rates = result['accept_rates']
                if len(accept_rates) > 0:
                    final_accept_rate = accept_rates[-1]
                    print(f"  Final accept rate: {final_accept_rate:.3f}")
            
            if 'N_sim' in result:
                total_sims = np.sum(result['N_sim'])
                print(f"  Total simulations: {total_sims}")


def rerun_from_file(file_path):
    """Load and display results from existing pickle or CSV."""
    print(f"Loading results from: {file_path}")
    
    if file_path.endswith('.pkl'):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print("\nSIR Real World Analysis Results (from pickle):")
        print("=" * 50)
        print(f"Seed: {data['seed']}")
        print(f"Particles: {data['n_particles']}")
        print(f"Days analyzed: {data['n_day']}")
        print(f"Scales: {data['metadata']['scales']}")
        
        # Analyze convergence if results available
        if 'results' in data and data['results'] is not None:
            analyze_convergence(data['results'])
        
        # Show summary statistics
        df = data['summary_df']
        
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        print("\nSIR Real World Analysis Results (from CSV):")
        print("=" * 50)
        if 'seed' in df.columns:
            print(f"Seed: {df['seed'].iloc[0]}")
        
    else:
        print("Unsupported file type. Use .pkl or .csv files.")
        return
    
    # Display summary by scale
    for scale in df['scale'].unique():
        print(f"\n{scale.upper()} SCALE:")
        scale_data = df[df['scale'] == scale]
        
        for param in scale_data['parameter'].unique():
            param_data = scale_data[scale_data['parameter'] == param]
            print(f"  {param}:")
            print(f"    Components: {len(param_data)}")
            if len(param_data) > 1:
                print(f"    Mean range: [{param_data['mean'].min():.3f}, {param_data['mean'].max():.3f}]")
                print(f"    Overall mean: {param_data['mean'].mean():.3f} ± {param_data['std'].mean():.3f}")
            else:
                row = param_data.iloc[0]
                print(f"    Mean: {row['mean']:.3f} ± {row['std']:.3f}")
                print(f"    95% CI: [{row['q025']:.3f}, {row['q975']:.3f}]")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SIR real world analysis on COVID-19 data"
    )
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n_particles', type=int, default=1000,
                       help='Number of particles for SMC (default: 1000)')
    parser.add_argument('--n_day', type=int, default=120,
                       help='Number of days to analyze (default: 120)')
    parser.add_argument('--output-dir', type=str, default="experiments",
                       help='Output directory (default: experiments)')
    parser.add_argument('--rerun', type=str, default=None,
                       help='Path to pickle or CSV file for rerunning analysis')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Handle rerun case
    if args.rerun:
        rerun_from_file(args.rerun)
        return
    
    print("SIR Real World Analysis")
    print(f"Parameters: seed={args.seed}, n_particles={args.n_particles}, n_day={args.n_day}")
    
    try:
        # Load data
        dep_pop, dep_name, reg_pop, reg_name, dep_reg = load_population_data()
        I_obs_fr, I_obs_reg, I_obs_dep, date, dep_list, reg_list = load_hospitalization_data(
            dep_pop, dep_reg, reg_pop
        )
        
        # Focus on first wave
        I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, date_V1 = focus_first_wave(
            I_obs_fr, I_obs_reg, I_obs_dep, date, args.n_day
        )
        
        # Setup models
        models = setup_models(
            I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, reg_list, dep_list, 
            reg_pop, dep_pop, args.n_day
        )
        
        # Prepare data info for saving
        data_info = {
            'dep_list': dep_list,
            'reg_list': reg_list,
            'dep_pop': dep_pop,
            'reg_pop': reg_pop,
            'date_range': [str(date_V1[0]), str(date_V1[-1])],
            'I_obs_shapes': {
                'national': I_obs_fr_V1.shape,
                'regional': I_obs_reg_V1.shape,
                'departmental': I_obs_dep_V1.shape
            }
        }
        
        # Run inference
        results = run_inference(models, args.n_particles, args.seed)
        
        # Extract summaries
        summary_df = extract_posterior_summaries(results)
        
        # Analyze convergence
        analyze_convergence(results)
        
        # Save results
        pkl_path, csv_path, full_data = save_results(
            results, summary_df, models, data_info, args.seed, args.n_particles, args.n_day, args.output_dir
        )
        
        print("\nSIR analysis complete!")
        print(f"Full results: {pkl_path}")
        print(f"Summary: {csv_path}")
        
        # Print summary statistics
        print(f"\nPosterior Summary:")
        for scale in summary_df['scale'].unique():
            scale_data = summary_df[summary_df['scale'] == scale]
            r0_data = scale_data[scale_data['parameter'] == 'R0']
            if len(r0_data) > 0:
                print(f"  {scale}: R0 = {r0_data['mean'].iloc[0]:.3f} ± {r0_data['std'].iloc[0]:.3f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data files are available in experiments/data/")
        print("Required files:")
        print("  - experiments/data/donnees_departements.csv")
        print("  - experiments/data/data-dep.csv")


if __name__ == "__main__":
    main()