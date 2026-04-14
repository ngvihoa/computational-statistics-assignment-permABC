"""
Hamming distance analysis: σ_t* vs σ_{t+1}* and swap warm-start recovery.

Demonstrates that:
1. Consecutive optimal assignments are close in Hamming distance
2. Swap warm-start from σ_t* almost always recovers σ_{t+1}*

This provides evidence that swap is the best sequential compromise for
assignment in permABC-SMC.

Usage:
  python analyze_hamming_swap.py --K 5 10 20 --seed 42
  python analyze_hamming_swap.py --K 5 10 20 50 --n-repeats 5 --seed 0
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from jax import random

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.assignment.solvers.lsa import solve_lsa
from permabc.assignment.solvers.swap import do_swap
from permabc.assignment.distances import compute_total_distance


def normalized_hamming(sigma1, sigma2):
    """Normalized Hamming distance between two index arrays.

    For a single particle: fraction of positions that differ.
    For batched: mean over particles.
    """
    if sigma1.ndim == 1:
        return np.mean(sigma1 != sigma2)
    # batched: (N, K)
    return np.mean(sigma1 != sigma2, axis=1)


def run_lsa_reference(key, model, y_obs, n_particles, alpha, n_iter_max, n_sim_max):
    """Run LSA-SMC and return per-iteration permutations."""
    res = perm_abc_smc(
        key=key, model=model, n_particles=n_particles,
        epsilon_target=0.0, y_obs=y_obs,
        kernel=KernelTruncatedRW, alpha_epsilon=alpha,
        verbose=0, N_iteration_max=n_iter_max, N_sim_max=n_sim_max,
        try_identity=True, try_swaps=False, try_lsa=True,
    )
    return res


def analyze_consecutive_hamming(res, model, y_obs):
    """Compute Hamming distance between consecutive optimal assignments.

    Also test whether swap warm-start from σ_t* recovers σ_{t+1}*.
    """
    n_iters = len(res['Eps_values'])
    zs_list = res['Zs']          # list of arrays (n_particles, K, dim)
    ys_list = res['Ys_index']     # list of arrays (n_particles, K)
    zs_idx_list = res['Zs_index'] # list of arrays (n_particles, K)
    eps_list = res['Eps_values']

    rows = []

    for t in range(1, n_iters):
        zs_prev_idx = zs_idx_list[t-1]  # σ_{t-1}*
        zs_curr_idx = zs_idx_list[t]     # σ_t*
        ys_prev = ys_list[t-1]
        ys_curr = ys_list[t]

        N = zs_curr_idx.shape[0]
        K = zs_curr_idx.shape[1]
        eps_prev = float(eps_list[t-1])
        eps_curr = float(eps_list[t])

        # --- Hamming(σ_{t-1}*, σ_t*) ---
        hamming_per_particle = normalized_hamming(zs_prev_idx, zs_curr_idx)

        # --- Can swap from σ_{t-1}* find σ_t*? ---
        # We need the cost matrix at time t to test this
        zs_t = np.asarray(zs_list[t])
        local_mats = np.asarray(model.distance_matrices_loc(zs_t, y_obs, K, K))
        local_mats = np.where(np.isinf(local_mats), 1e12, local_mats)
        global_d = np.asarray(model.distance_global(zs_t, y_obs))

        # Distance with LSA optimal (σ_t*)
        dist_lsa = compute_total_distance(zs_curr_idx, ys_curr, local_mats, global_d)

        # Distance with previous assignment (σ_{t-1}*)
        dist_prev = compute_total_distance(zs_prev_idx, ys_prev, local_mats, global_d)

        # Swap from σ_{t-1}*
        ys_swap, zs_swap = do_swap(local_mats, ys_prev.copy(), zs_prev_idx.copy(), max_sweeps=3)
        dist_swap = compute_total_distance(zs_swap, ys_swap, local_mats, global_d)

        # Hamming between swap result and LSA optimal
        hamming_swap_vs_lsa = normalized_hamming(zs_swap, zs_curr_idx)

        # Per-particle: did swap recover the LSA optimal?
        swap_exact_match = np.all(zs_swap == zs_curr_idx, axis=1)

        # Ratio: dist_swap / dist_lsa (1.0 = swap found optimal)
        ratio_swap = dist_swap / np.maximum(dist_lsa, 1e-15)
        ratio_prev = dist_prev / np.maximum(dist_lsa, 1e-15)

        rows.append({
            "iter": t,
            "epsilon_prev": eps_prev,
            "epsilon_curr": eps_curr,
            "N": N,
            "K": K,
            # Hamming stats
            "hamming_mean": float(np.mean(hamming_per_particle)),
            "hamming_median": float(np.median(hamming_per_particle)),
            "hamming_max": float(np.max(hamming_per_particle)),
            "hamming_zero_frac": float(np.mean(hamming_per_particle == 0)),
            # Previous assignment quality on new cost matrix
            "ratio_prev_mean": float(np.mean(ratio_prev)),
            "ratio_prev_median": float(np.median(ratio_prev)),
            # Swap recovery
            "ratio_swap_mean": float(np.mean(ratio_swap)),
            "ratio_swap_median": float(np.median(ratio_swap)),
            "swap_exact_match_frac": float(np.mean(swap_exact_match)),
            "hamming_swap_vs_lsa_mean": float(np.mean(hamming_swap_vs_lsa)),
            # Distance info
            "dist_lsa_mean": float(np.mean(dist_lsa)),
            "dist_prev_mean": float(np.mean(dist_prev)),
            "dist_swap_mean": float(np.mean(dist_swap)),
        })

    return pd.DataFrame(rows)


def generate_plots(df, out_dir):
    """Generate publication-quality Hamming analysis figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'legend.fontsize': 9, 'figure.dpi': 150,
    })

    K_values = sorted(df['K'].unique())
    colors = {5: '#1f77b4', 10: '#ff7f0e', 20: '#2ca02c', 50: '#d62728', 100: '#9467bd'}

    # =============================================
    # Figure 1: Hamming(σ_t*, σ_{t+1}*) vs iteration
    # =============================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for K in K_values:
        sub = df[df['K'] == K].groupby('iter').agg({
            'hamming_mean': ['mean', 'std'],
            'hamming_zero_frac': 'mean',
        }).reset_index()
        sub.columns = ['iter', 'hamming_mean', 'hamming_std', 'zero_frac']
        c = colors.get(K, 'gray')
        ax.plot(sub['iter'], sub['hamming_mean'], '-o', ms=3, color=c, label=f'K={K}')
        ax.fill_between(sub['iter'],
                        sub['hamming_mean'] - sub['hamming_std'],
                        sub['hamming_mean'] + sub['hamming_std'],
                        alpha=0.15, color=c)
    ax.set_xlabel('SMC iteration $t$')
    ax.set_ylabel(r'$d_H(\sigma_t^*, \sigma_{t+1}^*) / K$')
    ax.set_title(r'Hamming distance between consecutive optima')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # =============================================
    # Figure 2: Fraction where σ_t* = σ_{t+1}* (zero Hamming)
    # =============================================
    ax = axes[1]
    for K in K_values:
        sub = df[df['K'] == K].groupby('iter')['hamming_zero_frac'].mean().reset_index()
        c = colors.get(K, 'gray')
        ax.plot(sub['iter'], sub['hamming_zero_frac'], '-o', ms=3, color=c, label=f'K={K}')
    ax.set_xlabel('SMC iteration $t$')
    ax.set_ylabel(r'$\Pr(\sigma_t^* = \sigma_{t+1}^*)$')
    ax.set_title('Fraction of particles with unchanged optimal $\\sigma$')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_hamming_consecutive.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  -> fig_hamming_consecutive.pdf")

    # =============================================
    # Figure 3: Swap recovery — ratio and exact match
    # =============================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for K in K_values:
        sub = df[df['K'] == K].groupby('iter').agg({
            'ratio_swap_mean': 'mean',
            'ratio_prev_mean': 'mean',
        }).reset_index()
        c = colors.get(K, 'gray')
        ax.plot(sub['iter'], sub['ratio_swap_mean'], '-o', ms=3, color=c,
                label=f'Swap from $\\sigma_t^*$, K={K}')
        ax.plot(sub['iter'], sub['ratio_prev_mean'], '--s', ms=3, color=c, alpha=0.5,
                label=f'$\\sigma_t^*$ reused, K={K}')
    ax.axhline(y=1.0, color='black', ls=':', lw=0.8, label='LSA optimal')
    ax.set_xlabel('SMC iteration $t$')
    ax.set_ylabel(r'Distance ratio $d / d_{\mathrm{LSA}}$')
    ax.set_title('Swap warm-start quality vs reusing $\\sigma_t^*$')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for K in K_values:
        sub = df[df['K'] == K].groupby('iter')['swap_exact_match_frac'].mean().reset_index()
        c = colors.get(K, 'gray')
        ax.plot(sub['iter'], sub['swap_exact_match_frac'], '-o', ms=3, color=c, label=f'K={K}')
    ax.set_xlabel('SMC iteration $t$')
    ax.set_ylabel(r'$\Pr(\sigma_{\mathrm{swap}} = \sigma_{\mathrm{LSA}}^*)$')
    ax.set_title('Fraction where swap recovers the exact LSA optimum')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_swap_recovery.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  -> fig_swap_recovery.pdf")

    # =============================================
    # Figure 4: Hamming vs epsilon (log-scale)
    # =============================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for K in K_values:
        sub = df[df['K'] == K].groupby('iter').agg({
            'epsilon_curr': 'mean',
            'hamming_mean': 'mean',
        }).reset_index()
        c = colors.get(K, 'gray')
        ax.plot(sub['epsilon_curr'], sub['hamming_mean'], '-o', ms=3, color=c, label=f'K={K}')
    ax.set_xlabel(r'$\varepsilon_t$')
    ax.set_ylabel(r'$d_H(\sigma_t^*, \sigma_{t+1}^*) / K$')
    ax.set_title(r'Hamming distance vs $\varepsilon$')
    ax.invert_xaxis()
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for K in K_values:
        sub = df[df['K'] == K].groupby('iter').agg({
            'epsilon_curr': 'mean',
            'swap_exact_match_frac': 'mean',
        }).reset_index()
        c = colors.get(K, 'gray')
        ax.plot(sub['epsilon_curr'], sub['swap_exact_match_frac'], '-o', ms=3, color=c, label=f'K={K}')
    ax.set_xlabel(r'$\varepsilon_t$')
    ax.set_ylabel(r'$\Pr(\sigma_{\mathrm{swap}} = \sigma_{\mathrm{LSA}}^*)$')
    ax.set_title(r'Swap exact recovery vs $\varepsilon$')
    ax.invert_xaxis()
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_hamming_vs_epsilon.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  -> fig_hamming_vs_epsilon.pdf")

    # =============================================
    # Summary table
    # =============================================
    print("\n=== SUMMARY TABLE ===")
    summary = df.groupby('K').agg({
        'hamming_mean': ['mean', 'std'],
        'hamming_zero_frac': 'mean',
        'ratio_swap_mean': 'mean',
        'swap_exact_match_frac': 'mean',
        'ratio_prev_mean': 'mean',
    }).round(4)
    summary.columns = ['hamming_mean', 'hamming_std', 'σ_t=σ_{t+1} frac',
                       'swap_ratio', 'swap_exact_frac', 'prev_ratio']
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(description='Hamming analysis: consecutive σ* and swap recovery')
    parser.add_argument('--K', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--n-obs', type=int, default=10)
    parser.add_argument('--n-particles', type=int, default=500)
    parser.add_argument('--n-repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true', help='Quick run: 200 particles, 1 repeat')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_particles = 200
        args.n_repeats = 1

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'results', 'hamming_analysis')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'hamming_raw.csv')

    if args.plot_only:
        if not os.path.exists(csv_path):
            print(f"ERROR: {csv_path} not found. Run without --plot-only first.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        generate_plots(df, out_dir)
        return

    all_dfs = []
    key = random.PRNGKey(args.seed)
    total = len(args.K) * args.n_repeats
    done = 0
    t_start = time.perf_counter()

    for K in args.K:
        for rep in range(args.n_repeats):
            done += 1
            model = GaussianWithNoSummaryStats(K=K, n_obs=args.n_obs)
            key, k1, k2, k3 = random.split(key, 4)

            true_theta = model.prior_generator(k1, 1)
            y_obs = np.asarray(model.data_generator(k2, true_theta))

            print(f"[{done}/{total}] K={K}, repeat={rep} — running LSA-SMC...", end=' ', flush=True)
            t0 = time.perf_counter()

            res = run_lsa_reference(
                k3, model, y_obs, args.n_particles,
                alpha=0.5, n_iter_max=50, n_sim_max=500_000,
            )

            dt = time.perf_counter() - t0
            n_iters = len(res['Eps_values'])
            print(f"{n_iters} iters in {dt:.1f}s")

            df_pair = analyze_consecutive_hamming(res, model, y_obs)
            df_pair['repeat'] = rep
            df_pair['seed'] = args.seed
            all_dfs.append(df_pair)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    elapsed = time.perf_counter() - t_start
    print(f"Total time: {elapsed:.1f}s")

    generate_plots(df, out_dir)


if __name__ == '__main__':
    main()
