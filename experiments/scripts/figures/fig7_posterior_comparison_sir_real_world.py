#!/usr/bin/env python3
"""
Figure 7: Posterior comparison for SIR real world analysis.

This script creates posterior distribution plots comparing R0 and gamma parameters
across different scales (national, regional, departmental) for COVID-19 data.

Usage:
    python fig7_posterior_comparison_sir_real_world.py
    python fig7_posterior_comparison_sir_real_world.py --seed 42
    python fig7_posterior_comparison_sir_real_world.py --rerun experiments/results/sir_real_world_seed_42.pkl
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import argparse
import subprocess




def load_or_generate_results(seed, output_dir, force_regenerate=False):
    """Load existing results or generate new ones."""
    results_dir = os.path.join(output_dir, "results")
    pkl_path = os.path.join(results_dir, f"sir_real_world_seed_{seed}.pkl")
    csv_path = os.path.join(results_dir, f"sir_real_world_seed_{seed}.csv")
    
    # Check if pickle results exist and load them
    if os.path.exists(pkl_path) and not force_regenerate:
        print(f"Loading existing pickle results from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data['summary_df'], data, pkl_path
    
    # Check if CSV results exist and convert to expected format
    if os.path.exists(csv_path) and not force_regenerate:
        print(f"Loading existing CSV results from: {csv_path}")
        df = pd.read_csv(csv_path)
        # Create minimal data structure for backward compatibility
        data = {
            'summary_df': df,
            'results': None,
            'seed': seed,
            'metadata': {'loaded_from_csv': True}
        }
        return df, data, csv_path
    
    # Generate new results
    print("Generating new SIR results...")
    script_path = os.path.join(os.path.dirname(__file__), 'run_sir_real_world.py')
    
    cmd = [
        sys.executable, script_path,
        '--seed', str(seed),
        '--output-dir', output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running SIR analysis: {result.stderr}")
        raise RuntimeError("Failed to generate SIR results")
    
    # Load the generated results (should be pickle now)
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data['summary_df'], data, pkl_path
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        data = {
            'summary_df': df,
            'results': None,
            'seed': seed,
            'metadata': {'loaded_from_csv': True}
        }
        return df, data, csv_path
    else:
        raise FileNotFoundError(f"Expected results file not found: {pkl_path} or {csv_path}")


def create_posterior_plots(df, full_data=None):
    """Create posterior distribution plots."""
    # Setup colors for different scales
    colors = {
        'national_perm': '#1f77b4',  # Blue
        'national_abc': '#ff7f0e',   # Orange  
        'regional': '#2ca02c',       # Green
        'departmental': '#d62728'    # Red
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fontsize = 12
    title_size = 15
    
    # Plot 1: R0 distributions
    r0_data = df[df['parameter'] == 'R0']
    
    for scale in r0_data['scale'].unique():
        scale_data = r0_data[r0_data['scale'] == scale]
        if len(scale_data) > 0:
            # Try to use actual posterior samples if available
            if full_data and 'results' in full_data and full_data['results'] is not None:
                try:
                    if scale in full_data['results']:
                        final_thetas = full_data['results'][scale]['Thetas'][-1]
                        r0_samples = final_thetas.glob[:, 0]
                        # Plot density using actual samples
                        sns.kdeplot(r0_samples, ax=axes[0], color=colors.get(scale, 'gray'),
                                   label=scale.replace('_', ' ').title(), linewidth=2)
                        continue
                except Exception as e:
                    print(f"Warning: Could not use posterior samples for {scale}: {e}")
            
            # Fallback: create synthetic samples from mean and std
            mean_val = scale_data['mean'].iloc[0]
            std_val = scale_data['std'].iloc[0]
            samples = np.random.normal(mean_val, std_val, 1000)
            
            # Plot density
            sns.kdeplot(samples, ax=axes[0], color=colors.get(scale, 'gray'),
                       label=scale.replace('_', ' ').title(), linewidth=2)
    
    axes[0].set_xlabel("R₀", fontsize=fontsize)
    axes[0].set_ylabel("Density", fontsize=fontsize)
    axes[0].set_title("Distribution of R₀", fontsize=title_size)
    axes[0].set_xlim(1.0, 1.4)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Gamma (nu) distributions
    gamma_data = df[df['parameter'] == 'gamma']
    
    # Helper function to plot gamma for a scale
    def plot_gamma_for_scale(scale, color, label, alpha=0.5):
        scale_gamma = gamma_data[gamma_data['scale'] == scale]
        if len(scale_gamma) == 0:
            return
        
        # Try to use actual posterior samples if available
        if full_data and 'results' in full_data and full_data['results'] is not None:
            try:
                if scale in full_data['results']:
                    final_thetas = full_data['results'][scale]['Thetas'][-1]
                    gamma_samples = final_thetas.loc[:, :, 2]  # Assuming gamma is index 2
                    
                    for k in range(gamma_samples.shape[1]):
                        if k == 0:
                            sns.kdeplot(gamma_samples[:, k], ax=axes[1], color=color, 
                                       alpha=alpha, label=label)
                        else:
                            sns.kdeplot(gamma_samples[:, k], ax=axes[1], color=color, alpha=alpha)
                    return
            except Exception as e:
                print(f"Warning: Could not use posterior samples for {scale} gamma: {e}")
        
        # Fallback: synthetic samples from summary statistics
        for i, (_, row) in enumerate(scale_gamma.iterrows()):
            mean_val = row['mean']
            std_val = row['std']
            samples = np.random.normal(mean_val, std_val, 1000)
            
            if i == 0:
                sns.kdeplot(samples, ax=axes[1], color=color, alpha=alpha, label=label)
            else:
                sns.kdeplot(samples, ax=axes[1], color=color, alpha=alpha)
    
    # Plot different scales
    plot_gamma_for_scale('departmental', colors['departmental'], 'Departmental scale', alpha=0.2)
    plot_gamma_for_scale('regional', colors['regional'], 'Regional scale', alpha=0.5)
    plot_gamma_for_scale('national_perm', colors['national_perm'], 'National scale', alpha=1.0)
    
    axes[1].set_xlim(0, 4)
    axes[1].set_xlabel("ν", fontsize=fontsize)
    axes[1].set_ylabel("")
    axes[1].set_title("Distribution of ν", fontsize=title_size)
    axes[1].grid(True, alpha=0.3)
    # Note: Legend would be too crowded, so we skip it for the second plot
    
    plt.tight_layout()
    return fig


def create_summary_table(df):
    """Create a summary table of posterior statistics."""
    print("\nPosterior Summary Statistics:")
    print("=" * 80)
    
    # R0 summary
    print("\nR₀ Estimates:")
    r0_data = df[df['parameter'] == 'R0']
    for _, row in r0_data.iterrows():
        scale = row['scale'].replace('_', ' ').title()
        print(f"  {scale:15}: {row['mean']:.3f} ± {row['std']:.3f} "
              f"[{row['q025']:.3f}, {row['q975']:.3f}]")
    
    # Gamma summary by scale
    print("\nν (Gamma) Estimates:")
    gamma_data = df[df['parameter'] == 'gamma']
    
    for scale in gamma_data['scale'].unique():
        scale_gamma = gamma_data[gamma_data['scale'] == scale]
        scale_name = scale.replace('_', ' ').title()
        print(f"\n  {scale_name}:")
        print(f"    Components: {len(scale_gamma)}")
        print(f"    Mean range: [{scale_gamma['mean'].min():.3f}, {scale_gamma['mean'].max():.3f}]")
        print(f"    Overall mean: {scale_gamma['mean'].mean():.3f} ± {scale_gamma['std'].mean():.3f}")


def save_figure(fig, seed, output_dir):
    """Save the posterior comparison figure."""
    figures_dir = os.path.join(output_dir, "figures", "fig7")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save plots
    base_name = f"fig7_seed_{seed}"
    fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to: {figures_dir}/{base_name}.*")
    return figures_dir


def rerun_from_file(file_path):
    """Recreate plots from existing pickle or CSV results."""
    print(f"Loading results from: {file_path}")
    
    # Determine file type and load accordingly
    if file_path.endswith('.pkl'):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        df = data['summary_df']
        seed = data.get('seed', 42)
        full_data = data
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        seed = df['seed'].iloc[0] if 'seed' in df.columns else 42
        full_data = None
    else:
        print("Unsupported file type. Use .pkl or .csv files.")
        return
    
    print(f"Recreating plots for seed={seed}")
    
    # Create plots
    fig = create_posterior_plots(df, full_data)
    
    # Save plots in same directory as source file
    base_dir = os.path.dirname(file_path)
    base_name = f"fig7_seed_{seed}_rerun"
    
    fig.savefig(os.path.join(base_dir, f"{base_name}.pdf"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(base_dir, f"{base_name}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plots recreated: {base_dir}/{base_name}.*")
    
    # Print summary
    create_summary_table(df)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SIR real world posterior comparison plots"
    )
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default="experiments",
                       help='Output directory (default: experiments)')
    parser.add_argument('--rerun', type=str, default=None,
                       help='Path to pickle or CSV file for rerunning analysis')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regeneration of SIR results even if they exist')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Handle rerun case
    if args.rerun:
        rerun_from_file(args.rerun)
        return
    
    print("Figure 7: SIR real world posterior comparison")
    print(f"Parameters: seed={args.seed}")
    
    try:
        # Load or generate results
        df, full_data, results_path = load_or_generate_results(
            args.seed, args.output_dir, args.force_regenerate
        )
        
        # Create plots
        fig = create_posterior_plots(df, full_data)
        
        # Save figure
        figures_dir = save_figure(fig, args.seed, args.output_dir)
        
        # Print summary
        create_summary_table(df)
        
        print("\nFigure 7 analysis complete!")
        print(f"Data: {results_path}")
        print(f"Figures: {figures_dir}/fig7_seed_{args.seed}.*")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure data files are available or run run_sir_real_world.py first")


if __name__ == "__main__":
    main()