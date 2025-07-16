#!/usr/bin/env python3
"""
Figure 6: Performance comparison with over-sampling and under-matching methods.

This script generates performance comparison plots between ABC algorithms
including over-sampling and under-matching methods.

Usage:
    python fig6_performance_comparison_with_osum.py
    python fig6_performance_comparison_with_osum.py --K 20 --K_outliers 4 --seed 42
    python fig6_performance_comparison_with_osum.py --rerun experiments/results/performance_K_20_outliers_4_osum_True_seed_42.pkl
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 6: Performance comparison with OSUM methods"
    )
    
    parser.add_argument('--K', type=int, default=20,
                       help='Number of components (default: 20)')
    parser.add_argument('--K_outliers', type=int, default=4,
                       help='Number of outlier components (default: 4 for fig6)')
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
    """Main execution function that calls run_performance_comparison with osum=True."""
    args = parse_arguments()
    
    print("Figure 6: Performance comparison with OSUM methods")
    print(f"Parameters: K={args.K}, K_outliers={args.K_outliers}, seed={args.seed}")
    
    # --- Corrected Path Construction ---
    # Go up 3 levels to get from /experiments/scripts/figures to the project root
    project_root = Path(__file__).resolve().parents[3]
    main_script_path = project_root / 'experiments' / 'scripts' / 'run_performance_comparison.py'
    results_dir = project_root / 'experiments' / 'results' / 'performance_comparison'
    
    # --- Check for existing results to reuse ---
    rerun_path = None
    if args.rerun:
        rerun_path = args.rerun
        print(f"User specified --rerun. Using file: {rerun_path}")
    else:
        results_filename = f"performance_K_{args.K}_outliers_{args.K_outliers}_osum_True_seed_{args.seed}.pkl"
        results_filepath = results_dir / results_filename
        if results_filepath.exists():
            rerun_path = str(results_filepath)
            print(f"✅ Found existing results file. Re-running in plot-only mode from: {rerun_path}")

    # --- Build the command to execute ---
    if rerun_path:
        # Build the rerun command
        cmd_args = [
            sys.executable, str(main_script_path),
            '--rerun', rerun_path,
            '--output-dir', str(project_root), # Pass project root as output dir
            '--plot', args.plot,
            '--osum' # Important: Tell the script to process as fig6
        ]
    else:
        # Build the command for a full run
        print("No results file found. Running full experiment...")
        cmd_args = [
            sys.executable, str(main_script_path),
            '--K', str(args.K),
            '--K_outliers', str(args.K_outliers),
            '--seed', str(args.seed),
            '--N_points', str(args.N_points),
            '--N_particles', str(args.N_particles),
            '--plot', args.plot,
            '--output-dir', str(project_root), # Pass project root as output dir
            '--osum' # Ensure OSUM methods are included
        ]

    # --- Execute the command ---
    print(f"\nExecuting: {' '.join(cmd_args)}\n")
    try:
        subprocess.run(cmd_args, check=True)
        print("\nFigure 6 generation complete!")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Script failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()