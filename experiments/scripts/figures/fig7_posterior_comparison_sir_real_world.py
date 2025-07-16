#!/usr/bin/env python3
"""
Figure 7: Posterior comparison for SIR real world analysis.

This script creates posterior distribution plots comparing R0 and gamma parameters
across different scales (national, regional, departmental) for COVID-19 data.
It reads pre-computed lightweight .pkl files.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from pathlib import Path

def load_inference_results(results_dir: str, seed: int) -> dict:
    """Loads all lightweight SIR inference results from the specified directory."""
    scales = ["national", "regional", "departmental"]
    all_data = {}
    print(f"Loading inference results from: {results_dir}")
    for scale in scales:
        # On charge les fichiers finaux (sans suffixe _migrated ou _light)
        file_path = Path(results_dir) / f"inference_sir_{scale}_seed_{seed}.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                all_data[scale] = pickle.load(f)
            print(f"  [✓] Loaded {scale} data.")
        else:
            print(f"  [✗] Warning: Could not find {scale} data at {file_path}")
            all_data[scale] = None
    return all_data

def create_posterior_plots(all_data: dict, include_regions: bool):
    """Create posterior distribution plots from loaded data."""
    colors = {'national': '#d62728', 'regional': '#2ca02c', 'departmental': '#1f77b4'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # --- Plot R0 distributions ---
    scales_to_plot_r0 = ['national', 'departmental']
    if include_regions:
        scales_to_plot_r0.append('regional')

    labels = {'national': 'National scale', 'regional': 'Regional scale ', 'departmental': 'Departmental scale'}
    for scale in scales_to_plot_r0:
        if all_data.get(scale):
            r0_samples = all_data[scale]['Thetas_final'].glob[:, 0]
            sns.kdeplot(r0_samples, ax=axes[0], color=colors.get(scale), label=labels.get(scale), linewidth=2)

    axes[0].set_xlabel("$R_0$", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Distribution of $R_0$", fontsize=14)
    axes[0].set_xlim(1., 1.4)
    axes[0].legend()
    # axes[0].grid(True, alpha=0.3)

    # --- Plot Gamma (nu) distributions ---
    scales_to_plot_gamma = ['national', 'departmental']
    if include_regions:
        scales_to_plot_gamma.append('regional')
        
    for scale in scales_to_plot_gamma:
        if all_data.get(scale):
            gamma_samples = all_data[scale]['Thetas_final'].loc[:, :, 2]
            for k in range(gamma_samples.shape[1]):
                label = scale.title() if k == 0 else None
                alpha = 1.0 if scale == 'national' else (0.5 if scale == 'regional' else 0.2)
                sns.kdeplot(gamma_samples[:, k], ax=axes[1], color=colors.get(scale), alpha=alpha, label=label)

    axes[1].set_xlabel("$\\nu$", fontsize=12)
    axes[1].set_title("Distribution of $\\nu$", fontsize=14)
    axes[1].set_ylabel("", fontsize=12)
    # axes[1].legend()
    axes[1].set_xlim(0, 4)
    # axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate SIR posterior comparison plots")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', type=str, 
                        default="experiments/results/sir_real_world_inference",
                        help='Directory with inference .pkl files')
    parser.add_argument('--figure-dir', type=str, 
                        default="figures/fig7",
                        help='Directory to save the plot')
    parser.add_argument('--regions', action='store_true', help='Include regional data in the plot')
    args = parser.parse_args()

    try:
        # 1. Charger les données (suppose qu'elles existent)
        all_data = load_inference_results(args.results_dir, args.seed)
        
        # 2. Créer le graphique
        fig = create_posterior_plots(all_data, args.regions)
        
        # 3. Sauvegarder la figure
        os.makedirs(args.figure_dir, exist_ok=True)
        if args.regions:
            fig_path = Path(args.figure_dir) / f"fig7_posterior_comparison_regions_seed_{args.seed}.pdf"
        else:
            fig_path = Path(args.figure_dir) / f"fig7_posterior_comparison_seed_{args.seed}.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        print(f"\nFigure 7 sauvegardée : {fig_path}")
        plt.show()
        plt.close(fig)

    except (FileNotFoundError, KeyError) as e:
        print(f"\n❌ ERREUR : {e}")
        print("Assurez-vous que les fichiers 'inference_sir_..._seed_42.pkl' existent et ont la bonne structure.")
        sys.exit(1)

if __name__ == "__main__":
    main()