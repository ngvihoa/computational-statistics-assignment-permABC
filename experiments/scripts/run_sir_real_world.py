#!/usr/bin/env python3
"""
SIR real world analysis for COVID-19 data at multiple scales.

Supports multiple inference methods:
  - permABC-SMC: permutation-enhanced SMC
  - ABC-SMC (Gibbs {H}b): ABC-SMC with Gibbs blocks (no permutation)
  - permABC-SMC (Gibbs {H}b): permABC-SMC with Gibbs blocks
  - ABC-Gibbs: true model-specific ABC-Gibbs (local d(y_k,z_k) per component)
  - ABC-SMC: standard ABC-SMC without permutation
  - permABC-SMC-OS: over-sampling
  - permABC-SMC-UM: under-matching

Usage:
    python run_sir_real_world.py --seed 42 --scales regional
    python run_sir_real_world.py --seed 42 --scales regional --methods "permABC-SMC" "ABC-Gibbs"
    python run_sir_real_world.py --seed 42 --scales national regional departmental --n_particles 500
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse
import subprocess
import time as _time
from jax import random
from pathlib import Path

from permabc.algorithms.smc import perm_abc_smc, abc_smc
from permabc.algorithms.over_sampling import perm_abc_smc_os
from permabc.algorithms.under_matching import perm_abc_smc_um
from permabc.core.kernels import KernelTruncatedRW
from permabc.models.SIR import SIR_real_world
from permabc.utils.functions import Theta

# Import true ABC-Gibbs for SIR
sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_gibbs_sir import run_gibbs_sampler_sir


# ── Method Registry ─────────────────────────────────────────────────────────

# Template with {H} placeholder for number of Gibbs blocks
_METHOD_REGISTRY_TEMPLATE = {
    "permABC-SMC":              {"tag": "perm_smc",        "type": "perm_smc"},
    "ABC-SMC (Gibbs {H}b)":    {"tag": "abc_smc_g{H}",    "type": "abc_smc_gibbs"},
    "permABC-SMC (Gibbs {H}b)":{"tag": "perm_smc_g{H}",   "type": "perm_smc_gibbs"},
    "ABC-Gibbs":                {"tag": "abc_gibbs_true",   "type": "abc_gibbs_true"},
    "ABC-SMC":                  {"tag": "abc_smc",          "type": "abc_smc"},
    "permABC-SMC-OS":           {"tag": "perm_smc_os",      "type": "perm_smc_os"},
    "permABC-SMC-UM":           {"tag": "perm_smc_um",      "type": "perm_smc_um"},
}


def _expand_registry(H):
    """Expand {H} placeholders in the method registry."""
    registry = {}
    for name_tpl, info_tpl in _METHOD_REGISTRY_TEMPLATE.items():
        name = name_tpl.replace("{H}", str(H))
        info = {k: v.replace("{H}", str(H)) if isinstance(v, str) else v
                for k, v in info_tpl.items()}
        registry[name] = info
    return registry


# ── Utility Function (for figure scripts) ──────────────────────────────────

def check_and_run_sir_inference(results_dir: str, seed: int, force_rerun: bool = False):
    """
    Checks if all SIR inference files exist for a given seed.
    If not, or if force_rerun is True, it runs this script to generate them.
    """
    scales = ["national", "regional", "departmental"]
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


# ── Data Loading ────────────────────────────────────────────────────────────

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


# ── Saving Results ──────────────────────────────────────────────────────────

def save_lightweight(result, scale, method_tag, metadata, results_dir):
    """Save a lightweight pkl compatible with fig7bis/fig8 scripts.

    Expected keys by figure scripts:
      - Thetas_final: Theta object with .loc and .glob
      - n_iterations, total_n_sim
    """
    os.makedirs(results_dir, exist_ok=True)

    seed = metadata.get('seed', 'unknown')
    file_path = os.path.join(results_dir, f"inference_sir_{scale}_{method_tag}_seed_{seed}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved: {file_path}")


def save_single_inference(result: dict, scale: str, metadata: dict, results_dir: str):
    """Saves a single inference result (legacy format for fig7/fig8)."""
    os.makedirs(results_dir, exist_ok=True)

    save_data = {
        'result': result,
        'scale': scale,
        'metadata': metadata,
    }

    seed = metadata.get('seed', 'unknown')
    file_path = os.path.join(results_dir, f"inference_sir_{scale}_seed_{seed}_migrated.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"  Saved (migrated): {file_path}")


def save_comparison_summary(all_results, metadata, results_dir):
    """Save a summary CSV comparing all methods."""
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for (method_name, scale), info in all_results.items():
        rows.append({
            'method': method_name,
            'scale': scale,
            'n_iterations': info.get('n_iterations', ''),
            'total_n_sim': info.get('total_n_sim', ''),
            'time': info.get('time_final', ''),
            'final_epsilon': info.get('final_epsilon', ''),
        })
    df = pd.DataFrame(rows)
    seed = metadata.get('seed', 'unknown')
    csv_path = os.path.join(results_dir, f"comparison_summary_seed_{seed}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Summary saved: {csv_path}")


# ── Method Runners ──────────────────────────────────────────────────────────

def _smc_result_to_lightweight(result, method_name):
    """Convert an SMC result dict to the lightweight format expected by figures."""
    thetas_final = result['Thetas'][-1]
    return {
        'Thetas_final': thetas_final,
        'n_iterations': len(result['Eps_values']),
        'total_n_sim': sum(result['N_sim']),
        'time_final': result['time_final'],
        'final_epsilon': result['Eps_values'][-1],
        'Eps_values': result['Eps_values'],
        'N_sim': result['N_sim'],
        'method': method_name,
    }


def _gibbs_result_to_lightweight(locals_chain, r0_chain, eps_loc, eps_glob, times, n_sim_per_iter, K, method_name, n_particles_out=500):
    """Convert ABC-Gibbs output to the lightweight format expected by figures.

    The Gibbs sampler produces a single chain. We take the last n_particles_out
    iterations as pseudo-particles.
    """
    T = len(r0_chain) - 1  # total iterations
    burn_in = max(0, T - n_particles_out)

    # Build a Theta from the chain (last n_particles_out iterations)
    loc_samples = locals_chain[burn_in + 1:]  # (n_particles_out, K, 3)
    r0_samples = r0_chain[burn_in + 1:]       # (n_particles_out,)
    glob_samples = r0_samples[:, None]         # (n_particles_out, 1)

    import jax.numpy as jnp
    thetas_final = Theta(loc=jnp.array(loc_samples), glob=jnp.array(glob_samples))

    return {
        'Thetas_final': thetas_final,
        'n_iterations': T,
        'total_n_sim': T * n_sim_per_iter,
        'time_final': float(np.sum(times)),
        'final_epsilon': float(eps_glob[-1]) if len(eps_glob) > 0 else np.inf,
        'Eps_values': eps_glob.tolist(),
        'method': method_name,
    }


def run_method(method_name, method_info, key, model, y_obs, args,
               epsilon_from_perm_smc=None):
    """Dispatch to the appropriate algorithm based on method type.

    Parameters
    ----------
    epsilon_from_perm_smc : float or None
        Final epsilon from permABC-SMC run, used as starting epsilon for OS/UM.
    """
    mtype = method_info['type']
    Fi = args.final_iteration
    common_kwargs = dict(
        key=key, model=model, y_obs=y_obs,
        epsilon_target=0., n_particles=args.n_particles,
        kernel=KernelTruncatedRW, verbose=1, parallel=True,
        Final_iteration=Fi,
    )

    if mtype == "perm_smc":
        result = perm_abc_smc(**common_kwargs)
        return _smc_result_to_lightweight(result, method_name)

    elif mtype == "abc_smc":
        result = abc_smc(
            key=key, model=model, y_obs=y_obs,
            epsilon_target=0., n_particles=args.n_particles,
            kernel=KernelTruncatedRW, verbose=1,
            Final_iteration=Fi,
        )
        return _smc_result_to_lightweight(result, method_name)

    elif mtype == "abc_smc_gibbs":
        result = abc_smc(
            key=key, model=model, y_obs=y_obs,
            epsilon_target=0., n_particles=args.n_particles,
            kernel=KernelTruncatedRW, verbose=1,
            num_blocks_gibbs=args.num_gibbs_blocks,
            Final_iteration=Fi,
        )
        return _smc_result_to_lightweight(result, method_name)

    elif mtype == "perm_smc_gibbs":
        result = perm_abc_smc(
            **common_kwargs,
            num_blocks_gibbs=args.num_gibbs_blocks,
        )
        return _smc_result_to_lightweight(result, method_name)

    elif mtype == "abc_gibbs_true":
        K = model.K
        y_obs_2d = np.array(y_obs).squeeze()
        if y_obs_2d.ndim == 1:
            y_obs_2d = y_obs_2d[None, :]

        locals_chain, r0_chain, eps_loc, eps_glob, times, n_sim_per_iter = \
            run_gibbs_sampler_sir(
                key=key, model=model, y_obs_2d=y_obs_2d,
                T=args.gibbs_T, M_loc=args.gibbs_M_loc, M_glob=args.gibbs_M_glob,
            )
        return _gibbs_result_to_lightweight(
            locals_chain, r0_chain, eps_loc, eps_glob, times,
            n_sim_per_iter, K, method_name, n_particles_out=args.n_particles,
        )

    elif mtype == "perm_smc_os":
        K = model.K
        M_0 = max(2 * K, K + 5)
        eps_start = epsilon_from_perm_smc if epsilon_from_perm_smc is not None else np.inf
        result = perm_abc_smc_os(
            key=key, model=model, y_obs=y_obs,
            n_particles=args.n_particles, kernel=KernelTruncatedRW,
            M_0=M_0, epsilon=eps_start, verbose=1, Final_iteration=Fi,
        )
        return _smc_result_to_lightweight(result, method_name)

    elif mtype == "perm_smc_um":
        K = model.K
        L_0 = max(1, K // 2)
        eps_start = epsilon_from_perm_smc if epsilon_from_perm_smc is not None else np.inf
        result = perm_abc_smc_um(
            key=key, model=model, y_obs=y_obs,
            n_particles=args.n_particles, kernel=KernelTruncatedRW,
            L_0=L_0, epsilon=eps_start, verbose=1, Final_iteration=Fi,
        )
        return _smc_result_to_lightweight(result, method_name)

    else:
        raise ValueError(f"Unknown method type: {mtype}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SIR real world analysis")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_particles', type=int, default=1000)
    parser.add_argument('--n_day', type=int, default=120)
    parser.add_argument('--results-dir', type=str,
                        default="experiments/results/sir_real_world_inference")
    parser.add_argument('--scales', nargs='+',
                        default=["national", "regional", "departmental"],
                        choices=["national", "regional", "departmental"])
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to run (default: all). Use quotes for names with spaces.')
    parser.add_argument('--num_gibbs_blocks', type=int, default=3,
                        help='Number of Gibbs blocks for ABC-SMC/permABC-SMC Gibbs methods')
    # ABC-Gibbs true parameters
    parser.add_argument('--gibbs_T', type=int, default=2000,
                        help='Number of Gibbs iterations for true ABC-Gibbs')
    parser.add_argument('--gibbs_M_loc', type=int, default=50,
                        help='Number of prior candidates per component (local step)')
    parser.add_argument('--gibbs_M_glob', type=int, default=100,
                        help='Number of prior candidates for R0 (global step)')
    parser.add_argument('--final_iteration', type=int, default=100,
                        help='Final_iteration for SMC methods (continue after convergence)')
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    print(f"=== SIR Real World Analysis (Seed: {args.seed}) ===")

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "experiments" / "data"

    # Load data
    dep_pop, dep_name, reg_pop, reg_name, dep_reg = load_population_data(data_dir)
    I_obs_fr, I_obs_reg, I_obs_dep, date, dep_list, reg_list = load_hospitalization_data(data_dir, dep_pop, dep_reg, reg_pop)
    I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, date_V1 = focus_first_wave(I_obs_fr, I_obs_reg, I_obs_dep, date, args.n_day)
    (mod_fr, y_obs_fr), (mod_reg, y_obs_reg), (mod_dep, y_obs_dep) = setup_models(
        I_obs_fr_V1, I_obs_reg_V1, I_obs_dep_V1, reg_list, dep_list, reg_pop, dep_pop, args.n_day)

    scales_map = {
        "national": (mod_fr, y_obs_fr),
        "regional": (mod_reg, y_obs_reg),
        "departmental": (mod_dep, y_obs_dep),
    }

    # Build method registry
    registry = _expand_registry(args.num_gibbs_blocks)

    # Determine which methods to run
    if args.methods is None:
        methods_to_run = list(registry.keys())
    else:
        methods_to_run = args.methods
        # Validate
        for m in methods_to_run:
            if m not in registry:
                print(f"ERROR: Unknown method '{m}'. Available: {list(registry.keys())}")
                sys.exit(1)

    key = random.PRNGKey(args.seed)
    common_metadata = {
        'seed': args.seed, 'n_particles': args.n_particles, 'n_day': args.n_day,
        'dep_list': dep_list, 'reg_list': reg_list, 'dep_name': dep_name, 'reg_name': reg_name,
    }

    all_results = {}

    for scale in args.scales:
        model, y_obs = scales_map[scale]
        print(f"\n{'='*60}")
        print(f"  Scale: {scale.upper()} (K={model.K})")
        print(f"{'='*60}")

        # Run permABC-SMC first to get its final epsilon for OS/UM
        epsilon_from_perm_smc = None
        ordered_methods = list(methods_to_run)
        if "permABC-SMC" in ordered_methods:
            ordered_methods.remove("permABC-SMC")
            ordered_methods.insert(0, "permABC-SMC")

        for method_name in ordered_methods:
            method_info = registry[method_name]
            print(f"\n--- {method_name} ---")
            t0 = _time.perf_counter()

            key, subkey = random.split(key)
            try:
                lightweight = run_method(
                    method_name, method_info, subkey, model, y_obs, args,
                    epsilon_from_perm_smc=epsilon_from_perm_smc,
                )
                elapsed = _time.perf_counter() - t0
                print(f"  Done in {elapsed:.1f}s | N_sim={lightweight.get('total_n_sim', '?'):,} | "
                      f"eps_final={lightweight.get('final_epsilon', '?')}")

                # Capture permABC-SMC epsilon for OS/UM
                if method_name == "permABC-SMC":
                    epsilon_from_perm_smc = lightweight.get('final_epsilon', None)
                    print(f"  -> epsilon for OS/UM: {epsilon_from_perm_smc}")

                save_lightweight(lightweight, scale, method_info['tag'], common_metadata, args.results_dir)
                all_results[(method_name, scale)] = lightweight

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

        # Also save legacy migrated format for permABC-SMC (fig7/fig8 compatibility)
        if "permABC-SMC" in methods_to_run and ("permABC-SMC", scale) in all_results:
            lw = all_results[("permABC-SMC", scale)]
            # Build a result dict that mimics the old format
            legacy_result = {
                'Thetas': [lw['Thetas_final']],
                'Weights': [np.ones(args.n_particles) / args.n_particles],
                'Eps_values': lw.get('Eps_values', []),
            }
            save_single_inference(legacy_result, scale, common_metadata, args.results_dir)

    # Save comparison summary
    if all_results:
        save_comparison_summary(all_results, common_metadata, args.results_dir)

    print(f"\n=== All inferences complete. ===")


if __name__ == "__main__":
    main()
