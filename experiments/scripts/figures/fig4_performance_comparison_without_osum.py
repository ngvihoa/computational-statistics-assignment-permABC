#!/usr/bin/env python3
"""
Figure 4: Performance comparison without over-sampling and under-matching methods.

This script generates performance comparison plots between ABC algorithms
excluding over-sampling and under-matching methods.

Usage:
    python fig4_performance_comparison_without_osum.py
    python fig4_performance_comparison_without_osum.py --K 20 --K_outliers 0 --seed 42
    python fig4_performance_comparison_without_osum.py --rerun experiments/results/performance_K_20_outliers_4_osum_False_seed_42.pkl
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def find_project_root(marker_file='pyproject.toml'):
    """Find the project root by searching upwards for a marker file."""
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / marker_file).exists():
            return current_path
        current_path = current_path.parent
    # Fallback if no marker file is found (less robust)
    return Path(__file__).resolve().parents[2]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 4: Performance comparison without OSUM methods"
    )
    
    parser.add_argument('--K', type=int, default=20,
                       help='Number of components (default: 20)')
    parser.add_argument('--K_outliers', type=int, default=0,
                       help='Number of outlier components (default: 0 for fig4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--N_points', type=int, default=1000000,
                       help='Number of points for vanilla methods (default: 1,000,000)')
    parser.add_argument('--N_particles', type=int, default=1000,
                       help='Number of particles for SMC methods (default: 1,000)')
    parser.add_argument('--plot', type=str, choices=['nsim', 'time', 'both'], default='both',
                       help='Type of plots to generate: nsim (simulations), time, or both (default: both)')
    parser.add_argument('--output-dir', type=str, default="experiments",
                       help='Output directory (default: experiments)')
    parser.add_argument('--rerun', type=str, default=None,
                       help='Path to pickle or CSV file for rerunning analysis')
    
    return parser.parse_args()


def main():
    """
    Main execution. Checks for existing results (including from fig6)
    before running the full experiment.
    """
    args = parse_arguments()

    print("Figure 4: Performance comparison without OSUM methods")

    # --- Path Construction ---
    project_root = find_project_root()
    scripts_dir = project_root / "experiments" / "scripts"
    results_dir = project_root / "experiments" / "results" / "performance_comparison"
    
    # Corrected: Set the output directory for the main script to the project root.
    # The main script will handle creating subdirectories like /figures/fig4/.
    output_dir_for_main_script = project_root
    
    main_script_path = scripts_dir / 'run_performance_comparison.py'

    # --- Check for existing results to reuse ---

    # 1. Define path for the specific results file (osum=False)
    results_filename_no_osum = f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_False_seed_{args.seed}.pkl"
    results_filepath_no_osum = results_dir / results_filename_no_osum

    # 2. Define path for the comprehensive results file (osum=True, from fig6)
    results_filename_with_osum = f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_True_seed_{args.seed}.pkl"
    results_filepath_with_osum = results_dir / results_filename_with_osum

    rerun_path = None
    if args.rerun:
        rerun_path = args.rerun
        print(f"User specified --rerun. Using file: {rerun_path}")
    elif results_filepath_no_osum.exists():
        rerun_path = str(results_filepath_no_osum)
        print(f"✅ Found specific results file (osum=False). Re-running in plot-only mode from: {rerun_path}")
    elif results_filepath_with_osum.exists():
        rerun_path = str(results_filepath_with_osum)
        print(f"✅ Found comprehensive results file (osum=True). Using it to generate Figure 4.")
    
    # --- Build the command to execute ---
    
    if rerun_path:
        # If we found ANY file to rerun from, build the rerun command
        cmd_args = [
            sys.executable, str(main_script_path),
            '--rerun', rerun_path,
            '--output-dir', str(output_dir_for_main_script),
            '--plot', args.plot,
            '--no-osum'  # <<< Important: Still tell the script to process as fig4
        ]
    else:
        # Otherwise, build the command for a full run
        print("No suitable results file found. Running full experiment...")
        cmd_args = [
            sys.executable, str(main_script_path),
            '--K', str(args.K),
            '--K_outliers', str(args.K_outliers),
            '--seed', str(args.seed),
            '--N_points', str(args.N_points),
            '--N_particles', str(args.N_particles),
            '--plot', args.plot,
            '--output-dir', str(output_dir_for_main_script),
            '--no-osum'
        ]

    # --- Execute the command ---
    
    print(f"\nExecuting: {' '.join(cmd_args)}\n")
    try:
        # Use check=True to automatically raise an error if the script fails
        subprocess.run(cmd_args, check=True)
        print("\nFigure 4 generation complete!")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Script failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
if __name__ == "__main__":
    main()