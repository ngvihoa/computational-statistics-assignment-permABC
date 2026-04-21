#!/usr/bin/env python3
"""
SIR synthetic data validation: generate data from known parameters,
run permABC-SMC, and check parameter recovery.

This addresses the reviewer concern (Mock M5 / Biometrika Referee 1):
  "Without a simulated-data experiment where the ground truth is known,
   it is impossible to tell if permABC produces a good approximation."

Also runs sensitivity analysis on the noise parameter sigma.

Usage:
    python run_sir_synthetic_validation.py --seed 42 --K 10
    python run_sir_synthetic_validation.py --seed 42 --K 10 --sigmas 0.01 0.05 0.10
"""

import os
import sys
import argparse
import pickle
import time as _time

import numpy as np
import jax.numpy as jnp
from jax import random

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from permabc.algorithms.smc import perm_abc_smc
from permabc.core.kernels import KernelTruncatedRW
from permabc.models.SIR import SIR_real_world
from permabc.utils.functions import Theta


def generate_true_parameters(key, K):
    """Generate plausible true parameters for the synthetic SIR experiment.

    Instead of sampling from the (very wide) prior, we draw parameters
    from a realistic subregion to ensure epidemiologically meaningful
    trajectories that the algorithm can actually recover.
    """
    key1, key2, key3, key4, key5 = random.split(key, 5)

    # Global: R0 in [1.5, 3.0] (realistic COVID first-wave range)
    R0_true = random.uniform(key1, shape=(1, 1), minval=1.5, maxval=3.0)

    # Local recovery rate gamma_k in [0.1, 0.5]
    gamma_true = random.uniform(key2, shape=(1, K, 1), minval=0.1, maxval=0.5)

    # Initial infected I0_k in [1, 200] (per 100k)
    I0_true = random.uniform(key3, shape=(1, K, 1), minval=1.0, maxval=200.0)

    # Initial recovered R0_k in [0.1, 50]
    R0_init_true = random.uniform(key4, shape=(1, K, 1), minval=0.1, maxval=50.0)

    loc_true = jnp.concatenate([I0_true, R0_init_true, gamma_true], axis=2)
    glob_true = R0_true

    return Theta(loc=loc_true, glob=glob_true)


def run_single_experiment(key, K, n_obs, sigma, n_particles, true_theta, results_dir, seed):
    """Run one permABC-SMC experiment on synthetic SIR data."""

    sigma_str = f"{sigma:.3f}".replace('.', 'p')
    out_file = os.path.join(results_dir,
                            f"sir_synthetic_K{K}_sigma_{sigma_str}_seed_{seed}.pkl")

    if os.path.exists(out_file):
        print(f"  Already exists: {out_file}, skipping.")
        return

    # --- Setup model ---
    model = SIR_real_world(
        K=K, n_obs=n_obs, n_pop=100000, sigma=sigma,
        low_I=0.1, high_I=2000, low_R=0.1, high_R=2000,
        low_gamma=0.05, high_gamma=4.0, low_r0=0.5, high_r0=4.0,
    )

    # --- Simulate synthetic observed data from true parameters ---
    key, subkey = random.split(key)
    y_obs = model.data_generator(subkey, true_theta)
    print(f"  Synthetic data generated: shape {y_obs.shape}")
    print(f"  y_obs range: [{float(y_obs.min()):.2f}, {float(y_obs.max()):.2f}]")

    # --- Run permABC-SMC ---
    print(f"  Running permABC-SMC (K={K}, sigma={sigma}, N={n_particles})...")
    key, subkey = random.split(key)
    t0 = _time.time()
    result = perm_abc_smc(
        key=subkey,
        model=model,
        n_particles=n_particles,
        epsilon_target=0.0,
        y_obs=y_obs,
        kernel=KernelTruncatedRW,
        verbose=1,
        try_identity=True,
        try_swaps=True,
        try_lsa=True,
        Final_iteration=0,
    )
    elapsed = _time.time() - t0
    print(f"  Done in {elapsed:.1f}s, {len(result['Eps_values'])} iterations, "
          f"final eps={result['Eps_values'][-1]:.4f}")

    # --- Posterior predictive: simulate from posterior particles ---
    key, subkey = random.split(key)
    thetas_final = result['Thetas'][-1]
    n_pp = min(200, n_particles)
    pp_theta = Theta(loc=thetas_final.loc[:n_pp], glob=thetas_final.glob[:n_pp])
    y_pred = model.data_generator(subkey, pp_theta)

    # --- Save ---
    save_data = {
        'true_theta': true_theta,
        'y_obs': np.array(y_obs),
        'Thetas_final': result['Thetas'][-1],
        'Weights_final': result['Weights'][-1],
        'Eps_values': result['Eps_values'],
        'N_sim': result['N_sim'],
        'time_final': elapsed,
        'y_pred': np.array(y_pred),
        'K': K,
        'sigma': sigma,
        'n_obs': n_obs,
        'n_particles': n_particles,
        'seed': seed,
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"  Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="SIR synthetic validation")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--K', type=int, default=10,
                        help='Number of departments (compartments)')
    parser.add_argument('--n_obs', type=int, default=120,
                        help='Number of time observations (days)')
    parser.add_argument('--n_particles', type=int, default=1000)
    parser.add_argument('--sigmas', nargs='+', type=float,
                        default=[0.01, 0.05, 0.10],
                        help='Noise sigma values for sensitivity analysis')
    parser.add_argument('--results-dir', type=str,
                        default='experiments/experiments/results/sir_synthetic')
    args = parser.parse_args()

    print("=" * 60)
    print(f"SIR Synthetic Validation: K={args.K}, seed={args.seed}")
    print(f"Sigmas: {args.sigmas}")
    print("=" * 60)

    key = random.PRNGKey(args.seed)

    # Generate true parameters (same for all sigma values)
    key, subkey = random.split(key)
    true_theta = generate_true_parameters(subkey, args.K)

    R0_true = float(true_theta.glob[0, 0])
    gamma_true = np.array(true_theta.loc[0, :, 2])
    I0_true = np.array(true_theta.loc[0, :, 0])
    print(f"\nTrue parameters:")
    print(f"  R0 = {R0_true:.3f}")
    print(f"  gamma = {gamma_true}")
    print(f"  I0    = {I0_true}")

    # Run for each sigma
    for sigma in args.sigmas:
        print(f"\n{'='*60}")
        print(f"sigma = {sigma}")
        print(f"{'='*60}")
        key, subkey = random.split(key)
        run_single_experiment(
            subkey, args.K, args.n_obs, sigma,
            args.n_particles, true_theta,
            args.results_dir, args.seed,
        )

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
