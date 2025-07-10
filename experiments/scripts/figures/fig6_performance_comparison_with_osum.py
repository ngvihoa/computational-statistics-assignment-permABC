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
    print(f"N_points={args.N_points:,}, N_particles={args.N_particles:,}")
    print(f"Plot type: {args.plot}")
    
    # Path to the main performance comparison script
    script_path = os.path.join(os.path.dirname(__file__), 'run_performance_comparison.py')
    
    # Build command arguments
    cmd_args = [
        sys.executable, script_path,
        '--K', str(args.K),
        '--K_outliers', str(args.K_outliers),
        '--seed', str(args.seed),
        '--N_points', str(args.N_points),
        '--N_particles', str(args.N_particles),
        '--plot', args.plot,
        '--output-dir', args.output_dir,
        '--osum'  # Enable over-sampling and under-matching
    ]
    
    # Add rerun argument if provided
    if args.rerun:
        cmd_args.extend(['--rerun', args.rerun])
    
    print(f"Executing: {' '.join(cmd_args)}")
    
    # Execute the main script
    result = subprocess.run(cmd_args)
    
    if result.returncode == 0:
        print("\nFigure 6 generation complete!")
        print("This figure shows the performance comparison between:")
        print("  - ABC-Vanilla vs permABC-Vanilla")
        print("  - ABC-SMC vs permABC-SMC")
        print("  - ABC-PMC")
        print("  - permABC-SMC-OS (Over-Sampling)")
        print("  - permABC-SMC-UM (Under-Matching)")
        
        if args.plot == 'nsim':
            print("Plot generated: Simulation efficiency only")
        elif args.plot == 'time':
            print("Plot generated: Time efficiency only")
        else:
            print("Plots generated: Both simulation and time efficiency (separate files)")
            
        print(f"  - {args.K_outliers} outlier components added to test robustness")
    else:
        print(f"Error: Script failed with return code {result.returncode}")
    
    # Return the same exit code as the subprocess
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()