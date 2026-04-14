#!/usr/bin/env python3
"""
Figure 7: Posterior comparison for SIR real world analysis.

KDE plots comparing R0 and gamma parameters across scales (national, regional,
departmental) for COVID-19 data. Reads pre-computed lightweight .pkl files.

Usage:
    python fig7_posterior_comparison_sir_real_world.py
    python fig7_posterior_comparison_sir_real_world.py --seed 42 --regions
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_matplotlib, save_figure, find_project_root

setup_matplotlib()

_PROJECT_ROOT = find_project_root()

# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "sir_real_world_inference"
FIGURES_DIR = _PROJECT_ROOT / "experiments" / "figures" / "fig7"


def _fig_path(seed, regions=False):
    tag = "regions" if regions else "no_regions"
    return FIGURES_DIR / f"fig7_posterior_comparison_{tag}_seed_{seed}.pdf"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_inference_results(seed):
    scales = ["national", "regional", "departmental"]
    all_data = {}
    print(f"Loading inference results from: {RESULTS_DIR}")
    for scale in scales:
        file_path = RESULTS_DIR / f"inference_sir_{scale}_seed_{seed}.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                all_data[scale] = pickle.load(f)
            print(f"  Loaded {scale} data.")
        else:
            print(f"  Warning: Could not find {scale} data at {file_path}")
            all_data[scale] = None
    return all_data


# ── Plotting ─────────────────────────────────────────────────────────────────

def create_posterior_plots(all_data, include_regions):
    colors = {'national': '#d62728', 'regional': '#2ca02c', 'departmental': '#1f77b4'}
    labels = {'national': 'National scale', 'regional': 'Regional scale', 'departmental': 'Departmental scale'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # R0 distributions
    scales_r0 = ['national', 'departmental']
    if include_regions:
        scales_r0.append('regional')

    for scale in scales_r0:
        if all_data.get(scale):
            r0_samples = all_data[scale]['Thetas_final'].glob[:, 0]
            sns.kdeplot(r0_samples, ax=axes[0], color=colors[scale],
                        label=labels[scale], linewidth=2)

    axes[0].set_xlabel(r"$R_0$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"Distribution of $R_0$")
    axes[0].set_xlim(1., 1.4)
    axes[0].legend()

    # Gamma (nu) distributions
    scales_gamma = ['national', 'departmental']
    if include_regions:
        scales_gamma.append('regional')

    for scale in scales_gamma:
        if all_data.get(scale):
            gamma_samples = all_data[scale]['Thetas_final'].loc[:, :, 2]
            for k in range(gamma_samples.shape[1]):
                label = labels[scale] if k == 0 else None
                alpha = 1.0 if scale == 'national' else (0.5 if scale == 'regional' else 0.2)
                sns.kdeplot(gamma_samples[:, k], ax=axes[1],
                            color=colors[scale], alpha=alpha, label=label)

    axes[1].set_xlabel(r"$\nu$")
    axes[1].set_title(r"Distribution of $\nu$")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 4)

    fig.tight_layout()
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Figure 7: SIR posterior comparison")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regions', action='store_true', help='Include regional data')
    args = parser.parse_args()

    print("Figure 7: SIR posterior comparison")
    print(f"Parameters: seed={args.seed}, regions={args.regions}")

    try:
        all_data = load_inference_results(args.seed)
        fig = create_posterior_plots(all_data, args.regions)
        fig_path = _fig_path(args.seed, args.regions)
        save_figure(fig, fig_path)
        print(f"Figure saved to: {fig_path}")
    except (FileNotFoundError, KeyError) as e:
        print(f"\nERROR: {e}")
        print("Make sure the inference_sir_*_seed_*.pkl files exist.")
        sys.exit(1)


if __name__ == "__main__":
    main()
