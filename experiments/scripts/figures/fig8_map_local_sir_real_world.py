#!/usr/bin/env python3
"""
Figure 8: Map of local SIR parameters for real world COVID-19 data.

This script creates a choropleth map of France showing the spatial distribution
of local gamma (nu) parameters estimated from departmental COVID-19 data.

Usage:
    python fig8_map_local_sir_real_world.py
    python fig8_map_local_sir_real_world.py --seed 42
    python fig8_map_local_sir_real_world.py --rerun experiments/results/sir_real_world_seed_42.pkl
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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


def load_french_map():
    """Load French departments geographic data."""
    try:
        import geopandas as gpd
    except ImportError:
        print("Error: geopandas is required for map plotting")
        print("Install it with: pip install geopandas")
        raise
    
    print("Loading French map data...")
    
    # Try to load from URL (as in original script)
    try:
        france = gpd.read_file(
            "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
        )
        print("Map loaded from internet!")
        return france
    except Exception as e:
        print(f"Failed to load map from internet: {e}")
        
        # Try local file
        local_map_path = "experiments/data/departements-version-simplifiee.geojson"
        if os.path.exists(local_map_path):
            print(f"Loading map from local file: {local_map_path}")
            return gpd.read_file(local_map_path)
        else:
            print(f"Local map file not found: {local_map_path}")
            raise FileNotFoundError("Could not load French map data")


def create_department_mapping():
    """Create mapping from department indices to department codes."""
    # This should match the department list from the original SIR analysis
    # Note: This is a simplified mapping - in practice, you'd want to load this
    # from the same data source used in run_sir_real_world.py
    
    departments = [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", 
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "21", 
        "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", 
        "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", 
        "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", 
        "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", 
        "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", 
        "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", 
        "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", 
        "92", "93", "94", "95"
    ]
    
    return {i: dept for i, dept in enumerate(departments)}


def extract_gamma_samples(full_data, component_index):
    """Extract gamma samples for a specific component from full results."""
    if (full_data is None or 
        'results' not in full_data or 
        full_data['results'] is None or
        'departmental' not in full_data['results']):
        return None
    
    try:
        dept_result = full_data['results']['departmental']
        final_thetas = dept_result['Thetas'][-1]
        gamma_samples = final_thetas.loc[:, :, 2]  # Assuming gamma is index 2
        
        if component_index < gamma_samples.shape[1]:
            return gamma_samples[:, component_index]
    except Exception as e:
        print(f"Warning: Could not extract gamma samples for component {component_index}: {e}")
    
    return None


def prepare_map_data(df, france_map, full_data=None):
    """Prepare data for choropleth mapping."""
    # Filter for departmental gamma parameters
    dept_gamma = df[
        (df['scale'] == 'departmental') & 
        (df['parameter'] == 'gamma')
    ].copy()
    
    if len(dept_gamma) == 0:
        raise ValueError("No departmental gamma parameters found in data")
    
    print(f"Found {len(dept_gamma)} departmental gamma parameters")
    
    # Create department mapping
    dept_mapping = create_department_mapping()
    
    # Create dictionaries mapping department codes to gamma values and statistics
    gamma_dict = {}
    gamma_std_dict = {}
    gamma_samples_dict = {}
    
    for _, row in dept_gamma.iterrows():
        component = int(row['component'])
        if component in dept_mapping:
            dept_code = dept_mapping[component]
            gamma_dict[dept_code] = row['mean']
            gamma_std_dict[dept_code] = row['std']
            
            # Try to get actual samples if available
            samples = extract_gamma_samples(full_data, component)
            if samples is not None:
                gamma_samples_dict[dept_code] = samples
    
    print(f"Mapped {len(gamma_dict)} departments to gamma values")
    
    # Add gamma values to the map dataframe
    france_map = france_map.copy()
    france_map['gamma'] = france_map['code'].map(gamma_dict)
    france_map['gamma_std'] = france_map['code'].map(gamma_std_dict)
    
    # Filter out departments without data (overseas territories, etc.)
    france_map_filtered = france_map[france_map['gamma'].notna()].copy()
    
    print(f"Map contains {len(france_map_filtered)} departments with data")
    
    return france_map_filtered, gamma_dict, gamma_std_dict, gamma_samples_dict


def create_map_plot(france_map_filtered, seed):
    """Create the choropleth map."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create the choropleth map
    france_map_filtered.plot(
        column="gamma", 
        ax=ax, 
        cmap="viridis", 
        edgecolor="black", 
        linewidth=0.5,
        legend=False
    )
    
    # Remove axes
    ax.set_axis_off()
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap="viridis", 
        norm=plt.Normalize(
            vmin=france_map_filtered['gamma'].min(), 
            vmax=france_map_filtered['gamma'].max()
        )
    )
    sm._A = []
    
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('ν (gamma)', fontsize=12)
    
    # Set title
    ax.set_title(f'Spatial Distribution of ν Parameter\n(COVID-19 Departmental Analysis)', 
                fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig


def save_figure_and_data(fig, seed, output_dir, gamma_dict, gamma_std_dict, gamma_samples_dict):
    """Save the map figure and additional data."""
    figures_dir = os.path.join(output_dir, "figures", "fig8")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save plots
    base_name = f"fig8_seed_{seed}"
    fig.savefig(os.path.join(figures_dir, f"{base_name}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save map-specific data as pickle
    map_data = {
        'seed': seed,
        'gamma_dict': gamma_dict,
        'gamma_std_dict': gamma_std_dict,
        'gamma_samples_dict': gamma_samples_dict,
        'department_mapping': create_department_mapping()
    }
    
    pkl_path = os.path.join(figures_dir, f"fig8_map_data_seed_{seed}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(map_data, f)
    
    print(f"Map saved to: {figures_dir}/{base_name}.*")
    print(f"Map data saved to: {pkl_path}")
    return figures_dir


def print_map_statistics(gamma_dict, gamma_std_dict=None, gamma_samples_dict=None):
    """Print summary statistics for the mapped parameters."""
    if not gamma_dict:
        print("No gamma values to analyze")
        return
    
    values = np.array(list(gamma_dict.values()))
    
    print("\nSpatial Distribution Statistics:")
    print("=" * 40)
    print(f"Departments mapped: {len(values)}")
    print(f"Gamma range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"Mean gamma: {values.mean():.3f} ± {values.std():.3f}")
    print(f"Median gamma: {np.median(values):.3f}")
    
    # Find departments with extreme values
    min_idx = np.argmin(values)
    max_idx = np.argmax(values)
    dept_codes = list(gamma_dict.keys())
    
    print(f"Lowest gamma: {values[min_idx]:.3f} (Department {dept_codes[min_idx]})")
    print(f"Highest gamma: {values[max_idx]:.3f} (Department {dept_codes[max_idx]})")
    
    # Additional statistics if available
    if gamma_std_dict:
        std_values = np.array([gamma_std_dict[dept] for dept in dept_codes])
        print(f"Average uncertainty (std): {std_values.mean():.3f}")
        print(f"Uncertainty range: [{std_values.min():.3f}, {std_values.max():.3f}]")
    
    if gamma_samples_dict:
        sample_counts = [len(samples) for samples in gamma_samples_dict.values()]
        print(f"Posterior samples available for {len(gamma_samples_dict)} departments")
        if sample_counts:
            print(f"Sample sizes: {min(sample_counts)} - {max(sample_counts)}")


def rerun_from_file(file_path):
    """Recreate map from existing pickle or CSV results."""
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
    
    print(f"Recreating map for seed={seed}")
    
    try:
        # Load map data
        france_map = load_french_map()
        
        # Prepare data
        france_map_filtered, gamma_dict, gamma_std_dict, gamma_samples_dict = prepare_map_data(
            df, france_map, full_data
        )
        
        # Create map
        fig = create_map_plot(france_map_filtered, seed)
        
        # Save map in same directory as source file
        base_dir = os.path.dirname(file_path)
        base_name = f"fig8_seed_{seed}_rerun"
        
        fig.savefig(os.path.join(base_dir, f"{base_name}.pdf"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(base_dir, f"{base_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save map data
        map_data = {
            'seed': seed,
            'gamma_dict': gamma_dict,
            'gamma_std_dict': gamma_std_dict,
            'gamma_samples_dict': gamma_samples_dict,
            'department_mapping': create_department_mapping()
        }
        
        pkl_path = os.path.join(base_dir, f"fig8_map_data_seed_{seed}_rerun.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(map_data, f)
        
        print(f"Map recreated: {base_dir}/{base_name}.*")
        print(f"Map data saved: {pkl_path}")
        
        # Print statistics
        print_map_statistics(gamma_dict, gamma_std_dict, gamma_samples_dict)
        
    except Exception as e:
        print(f"Error creating map: {e}")
        print("Map plotting requires geopandas and internet connection or local map data")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate map of local SIR parameters for France"
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
    
    print("Figure 8: Map of local SIR parameters")
    print(f"Parameters: seed={args.seed}")
    
    try:
        # Load or generate results
        df, full_data, results_path = load_or_generate_results(
            args.seed, args.output_dir, args.force_regenerate
        )
        
        # Load map data
        france_map = load_french_map()
        
        # Prepare data for mapping
        france_map_filtered, gamma_dict, gamma_std_dict, gamma_samples_dict = prepare_map_data(
            df, france_map, full_data
        )
        
        # Create map
        fig = create_map_plot(france_map_filtered, args.seed)
        
        # Save figure and data
        figures_dir = save_figure_and_data(
            fig, args.seed, args.output_dir, gamma_dict, gamma_std_dict, gamma_samples_dict
        )
        
        # Print statistics
        print_map_statistics(gamma_dict, gamma_std_dict, gamma_samples_dict)
        
        print("\nFigure 8 analysis complete!")
        print(f"Data: {results_path}")
        print(f"Map: {figures_dir}/fig8_seed_{args.seed}.*")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Map plotting requires:")
        print("  1. geopandas library (pip install geopandas)")
        print("  2. Internet connection or local map data")
        print("  3. SIR analysis results (run run_sir_real_world.py first)")


if __name__ == "__main__":
    main()