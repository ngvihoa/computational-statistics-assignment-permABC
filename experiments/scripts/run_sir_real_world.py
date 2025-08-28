#!/usr/bin/env python3
"""
SIR real world analysis for COVID-19 data at multiple scales.

This script can be run directly to generate all SIR inference results.
It also provides a utility function `check_and_run_sir_inference` that can be
imported by other scripts to ensure data exists before plotting.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse
import subprocess
from jax import random
from pathlib import Path

# Assuming permabc is installed in editable mode `pip install -e .`
from permabc.algorithms.smc import perm_abc_smc
from permabc.core.kernels import KernelTruncatedRW
from permabc.models.SIR import SIR_real_world 


## --------------------------------------------------------------------
## Utility Function (to be imported by figure scripts)
## -------------------------------------------------------------------

def check_and_run_sir_inference(results_dir: str, seed: int, force_rerun: bool = False):
    """
    Checks if all SIR inference files exist for a given seed.
    If not, or if force_rerun is True, it runs this script to generate them.
    """
    scales = ["national", "regional", "departmental"]
    # Check for the migrated files
    expected_files = [Path(results_dir) / f"inference_sir_{scale}_seed_{seed}_migrated.pkl" for scale in scales]
    
    all_files_exist = all(f.exists() for f in expected_files)
    
    if all_files_exist and not force_rerun:
        print("All required SIR inference files found.")
        return

    print("One or more SIR files missing or rerun forced. Generating new results...")
    this_script_path = Path(__file__).resolve()
    
    cmd = [sys.executable, str(this_script_path), '--seed', str(seed), '--results-dir', results_dir]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate necessary SIR results: {e}")


## --------------------------------------------------------------------
## Data Loading and Model Setup Functions
## --------------------------------------------------------------------

def load_population_data(data_dir: Path) -> tuple:
    """Load and process French population data."""
    print("Loading population data...")
    pop_file = data_dir / "donnees_departements.csv"
    df_pop = pd.read_csv(pop_file, sep=";")
    
    dep_pop = {row["CODDEP"]: row["PTOT"] for _, row in df_pop.iterrows()}
    dep_name = {row["CODDEP"]: row["DEP"] for _, row in df_pop.iterrows()}
    dep_reg = {row["CODDEP"]: row["CODREG"] for _, row in df_pop.iterrows()}

    reg_pop, reg_name = {}, {}
    for r in df_pop["CODREG"].unique():
        reg_df = df_pop[df_pop["CODREG"] == r]
        reg_pop[r] = reg_df["PTOT"].sum()
        reg_name[r] = reg_df["REG"].iloc[0]
        
    return dep_pop, dep_name, reg_pop, reg_name, dep_reg

def load_hospitalization_data(data_dir: Path, dep_pop: dict, dep_reg: dict, reg_pop: dict) -> tuple:
    """Load and process COVID-19 hospitalization data."""
    print("Loading hospitalization data...")
    hosp_file = data_dir / "data-dep.csv"
    df_dep = pd.read_csv(hosp_file, sep=";")

    df_dep = df_dep[df_dep["sexe"] == 0]
    df_dep = df_dep[~df_dep['dep'].isin(["971", "972", "973", "974", "976", "978", "2A", "2B"])]
    
    IHR = 0.036
    def smooth_moving_average(x, window=7):
        return np.convolve(x, np.ones(window) / window, mode='same')

    date = np.unique(pd.to_datetime(df_dep["jour"]))
    dep_list = np.unique(df_dep["dep"])

    df_dep["reg"] = df_dep["dep"].map(dep_reg)
    df_dep["pop"] = df_dep["dep"].map(dep_pop)

    I_obs_dep = np.array([smooth_moving_average(df_dep[df_dep["dep"] == d]["hosp"]) / dep_pop[d] * 100000 / IHR for d in dep_list])
    
    df_reg = df_dep.groupby(['jour', 'reg']).sum(numeric_only=True).reset_index()
    reg_list = np.unique(df_reg["reg"])
    I_obs_reg = np.array([smooth_moving_average(df_reg[df_reg["reg"] == r]["hosp"]) / reg_pop[r] * 100000 / IHR for r in reg_list])
    
    df_fr = df_dep.groupby(['jour']).sum(numeric_only=True).reset_index()
    pop_fr = sum(dep_pop.values())
    I_obs_fr = smooth_moving_average(df_fr["hosp"]) / pop_fr * 100000 / IHR
    
    return I_obs_fr, I_obs_reg, I_obs_dep, date, dep_list, reg_list

def focus_first_wave(I_obs_fr, I_obs_reg, I_obs_dep, date, n_day=120):
    """Focus analysis on the first wave of COVID-19."""
    date_V1 = date[:n_day]
    I_obs_dep_V1 = I_obs_dep[:, :n_day]
    I_obs_reg_V1 = I_obs_reg[:, :n_day]
    I_obs_fr_V1 = I_obs_fr[:n_day]
    return I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, date_V1

def setup_models(I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, reg_list, dep_list, reg_pop, dep_pop, n_day):
    """Setup SIR models for different scales."""
    params = {'low_I': 0.1, 'high_I': 2000, 'low_R': 0.1, 'high_R': 2000,
              'low_gamma': 0.05, 'high_gamma': 4.0, 'low_r0': 0.5, 'high_r0': 4.0, 'n_pop': 100000}

    mod_fr = SIR_real_world(K=1, n_obs=n_day, **params)
    y_obs_fr = I_obs_fr_V1[None, None, :]

    weights_region = np.array([reg_pop[r] for r in reg_list])
    mod_reg = SIR_real_world(K=len(reg_list), n_obs=n_day, weights_distance=weights_region / np.sum(weights_region), **params)
    y_obs_reg = I_obs_reg_V1[None, :]

    weights_dep = np.array([dep_pop[d] for d in dep_list])
    mod_dep = SIR_real_world(K=len(dep_list), n_obs=n_day, weights_distance=weights_dep / np.sum(weights_dep), **params)
    y_obs_dep = I_obs_dep_V1[None, :]
    
    return (mod_fr, y_obs_fr), (mod_reg, y_obs_reg), (mod_dep, y_obs_dep)

## --------------------------------------------------------------------
## Main Execution Logic
## --------------------------------------------------------------------

def save_single_inference(result: dict, scale: str, metadata: dict, results_dir: str):
    """Saves a single, well-structured inference result to a pickle file."""
    os.makedirs(results_dir, exist_ok=True)
    
    # Structure de données finale, propre et complète
    save_data = {
        'result': result,    # Dictionnaire contenant 'Thetas', 'Weights', etc.
        'scale': scale,      # 'national', 'regional', ou 'departmental'
        'metadata': metadata # Dictionnaire contenant 'seed', 'dep_list', etc.
    }
    
    seed = metadata.get('seed', 'unknown')
    file_path = os.path.join(results_dir, f"inference_sir_{scale}_seed_{seed}_migrated.pkl")
    
    with open(file_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Results for '{scale}' saved to: {file_path}")

def parse_arguments():
    """Parse command line arguments for direct execution."""
    parser = argparse.ArgumentParser(description="Run SIR real world analysis")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_particles', type=int, default=1000)
    parser.add_argument('--n_day', type=int, default=120)
    parser.add_argument('--results-dir', type=str, 
                        default="experiments/results/sir_real_world_inference",
                        help='Path to save .pkl results')
    return parser.parse_args()

def main():
    """Main execution function to run all SIR inferences."""
    args = parse_arguments()
    print(f"--- Running SIR Real World Analysis (Seed: {args.seed}) ---")
    
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "experiments" / "data"

    # ... (Le reste de la fonction main reste le même, chargeant les données et créant les modèles) ...
    dep_pop, dep_name, reg_pop, reg_name, dep_reg = load_population_data(data_dir)
    I_obs_fr, I_obs_reg, I_obs_dep, date, dep_list, reg_list = load_hospitalization_data(data_dir, dep_pop, dep_reg, reg_pop)
    I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, date_V1 = focus_first_wave(I_obs_fr, I_obs_reg, I_obs_dep, date, args.n_day)
    (mod_fr, y_obs_fr), (mod_reg, y_obs_reg), (mod_dep, y_obs_dep) = setup_models(
        I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, reg_list, dep_list, reg_pop, dep_pop, args.n_day)
    
    key = random.PRNGKey(args.seed)
    # Les métadonnées à sauvegarder avec chaque résultat
    common_metadata = {
        'seed': args.seed, 'n_particles': args.n_particles, 'n_day': args.n_day,
        'dep_list': dep_list, 'reg_list': reg_list, 'dep_name': dep_name, 'reg_name': reg_name
    }

    # Lancer et sauvegarder chaque inférence
    scales_to_run = {"national": (mod_fr, y_obs_fr), "regional": (mod_reg, y_obs_reg), "departmental": (mod_dep, y_obs_dep)}
    for scale, (model, y_obs) in scales_to_run.items():
        print(f"\n--- Starting {scale.title()} Inference ---")
        key, subkey = random.split(key)
        result = perm_abc_smc(key=subkey, model=model, y_obs=y_obs, epsilon_target=0., n_particles=args.n_particles, kernel=KernelTruncatedRW, verbose=1, parallel=True)
        save_single_inference(result, scale, common_metadata, args.results_dir)

    print("\n--- All SIR inferences complete. ---")

if __name__ == "__main__":
    main()
