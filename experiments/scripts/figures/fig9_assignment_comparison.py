#!/usr/bin/env python3
"""
Figure 9: Assignment method comparison for permABC-SMC.

Compares assignment strategies on the Gaussian toy model:
  LSA, Hilbert, Hilbert+Swap, Swap, Smart Swap, Smart Hilbert,
  Smart H+S, Sinkhorn, Sinkhorn+Swap.

For each K in {5, 10, 20} (configurable) runs permABC-SMC and
records per-iteration:  epsilon, cumulative N_sim, cumulative time,
score_joint (joint diagnostic), KL_sigma2.

Generates:
  - 2x3 combined figure: score_joint / KL_sigma2  vs  {N_sim, Time, Epsilon}
  - Individual panels as standalone PDFs
  - Multi-K overlay plots

Usage:
    python fig9_assignment_comparison.py
    python fig9_assignment_comparison.py --K 20 --seed 42
    python fig9_assignment_comparison.py --rerun experiments/results/fig9_assignment_K_20_seed_42.pkl
"""

import sys
import argparse
import pickle
import time as time_mod
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import (
    ASSIGNMENT_COLORS, ASSIGNMENT_MARKERS, ASSIGNMENT_LINESTYLES,
    setup_matplotlib, save_figure, find_project_root,
    detect_joint_key, joint_ylabel, extract_series, plot_method_panel,
)

setup_matplotlib()

# Ensure project root is on sys.path for permabc imports
_PROJECT_ROOT = find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from jax import random
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta

# ── Import KL diagnostics ─────────────────────────────────────────────────────
sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "scripts"))
from diagnostics import (
    build_sigma2_reference_bins,
    empirical_kl_sigma2_abc_vs_true,
    expected_neg_log_joint_true,
)

# ── Assignment configurations ────────────────────────────────────────────────

ASSIGNMENT_CONFIGS = [
    ("LSA",              dict(cascade=["identity", "lsa"])),
    ("Hilbert",          dict(cascade=["identity", "hilbert"])),
    ("Hilbert+Swap",     dict(cascade=["identity", "hilbert", "swap"])),
    ("Swap",             dict(cascade=["identity", "swap"])),
    ("Smart Swap",       dict(cascade=["identity", "swap", "lsa"])),
    ("Smart Hilbert",    dict(cascade=["identity", "hilbert", "lsa"])),
    ("Smart H+S",        dict(cascade=["identity", "hilbert", "swap", "lsa"])),
    ("Sinkhorn",         dict(cascade=["identity", "sinkhorn"])),
    ("Sinkhorn+Swap",    dict(cascade=["identity", "sinkhorn", "swap"])),
]


# ── Experiment setup ──────────────────────────────────────────────────────────

def setup_experiment(K, seed, n_obs=10):
    sigma0 = 10
    alpha, beta = 5, 5
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    model = GaussianWithNoSummaryStats(K=K, n_obs=n_obs, sigma_0=sigma0, alpha=alpha, beta=beta)
    true_theta = model.prior_generator(subkey, 1)
    true_theta = Theta(loc=true_theta.loc, glob=np.array([1.0])[None, :])
    key, subkey = random.split(key)
    y_obs = model.data_generator(subkey, true_theta)
    return model, y_obs, true_theta, key


def _safe_at(seq, idx):
    if seq is None or idx < 0 or idx >= len(seq):
        return None
    return seq[idx]


def extract_per_iteration_metrics(out, model, y_obs, K, N_particles, n_sample=1000):
    """Extract per-iteration metrics from perm_abc_smc output."""
    sigma2_edges = build_sigma2_reference_bins(model, y_obs, nbins=80)
    rows = []
    n_sim_arr = np.asarray(out["N_sim"], dtype=float)
    eps_arr = np.asarray(out["Eps_values"], dtype=float)
    time_arr = np.asarray(out["Time"], dtype=float)
    unique_arr = np.asarray(out["unique_part"], dtype=float)

    if n_sim_arr.size < 2:
        return rows

    n_sim_cum = np.cumsum(n_sim_arr[1:])
    time_cum = np.cumsum(time_arr[1:])
    epsilons = eps_arr[1:]
    unique = unique_arr[1:]

    thetas_list = out["Thetas"]
    weights_list = out["Weights"]
    perm_list = out.get("Zs_index", None)

    n_steps = min(
        len(n_sim_cum), len(epsilons), len(time_cum), len(unique),
        max(0, len(thetas_list) - 1),
    )
    for i in range(n_steps):
        thetas_i = _safe_at(thetas_list, i + 1)
        weights_i = _safe_at(weights_list, i + 1)
        perm_i = _safe_at(perm_list, i + 1)
        if thetas_i is None:
            continue

        denom = K * max(float(unique[i]), 1e-12) * max(float(N_particles), 1.0)
        n_sim_norm = float(n_sim_cum[i]) / denom * n_sample
        time_norm = float(time_cum[i]) / denom * n_sample

        kl_s2 = empirical_kl_sigma2_abc_vs_true(
            model, y_obs, thetas_i, weights=weights_i, edges=sigma2_edges
        )
        score_joint = expected_neg_log_joint_true(
            model, y_obs, thetas_i, weights=weights_i, perm=perm_i
        )

        rows.append({
            "n_sim": n_sim_norm,
            "time": time_norm,
            "n_sim_raw": float(n_sim_cum[i]),
            "time_raw": float(time_cum[i]),
            "epsilon": float(epsilons[i]),
            "kl_sigma2": kl_s2,
            "score_joint": score_joint,
        })
    return rows


# ── Run one configuration ─────────────────────────────────────────────────────

def run_single(key, model, y_obs, K, N_particles, N_sim_max, stopping_rate,
               cascade):
    key, subkey = random.split(key)
    model.reset_weights_distance()

    out = perm_abc_smc(
        key=subkey,
        model=model,
        n_particles=N_particles,
        epsilon_target=0,
        y_obs=y_obs,
        kernel=KernelTruncatedRW,
        verbose=1,
        Final_iteration=0,
        update_weights_distance=False,
        stopping_accept_rate=stopping_rate,
        N_sim_max=N_sim_max,
        try_identity="identity" in cascade,
        try_hilbert="hilbert" in cascade,
        try_sinkhorn="sinkhorn" in cascade,
        try_swaps="swap" in cascade,
        try_lsa="lsa" in cascade,
    )
    rows = extract_per_iteration_metrics(out, model, y_obs, K, N_particles)
    return key, out, rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_figures(all_records, K, seed, figures_dir):
    """Generate all Figure 9 plots."""
    methods = [name for name, _ in ASSIGNMENT_CONFIGS]
    present = [m for m in methods if any(r.get("method") == m for r in all_records)]
    jk = detect_joint_key(all_records)
    jl = joint_ylabel(jk)
    log_y_joint = jk != "score_joint"

    panels = [
        ("n_sim", jk,  r"$N_{\mathrm{sim}}$ (per 1000 unique)",  jl),
        ("time",  jk,  "Time (s, per 1000 unique)",              jl),
        ("epsilon", jk, r"$\varepsilon$",                          jl),
        ("n_sim", "kl_sigma2", r"$N_{\mathrm{sim}}$ (per 1000 unique)",  r"$\mathrm{KL}_{\sigma^2}$"),
        ("time",  "kl_sigma2", "Time (s, per 1000 unique)",              r"$\mathrm{KL}_{\sigma^2}$"),
        ("epsilon","kl_sigma2",r"$\varepsilon$",                          r"$\mathrm{KL}_{\sigma^2}$"),
    ]

    _colors = ASSIGNMENT_COLORS
    _markers = ASSIGNMENT_MARKERS
    _ls = ASSIGNMENT_LINESTYLES

    # ── 2x3 combined figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (xk, yk, xl, yl) in enumerate(panels):
        row, col = divmod(idx, 3)
        ly = log_y_joint if yk == jk else True
        plot_method_panel(axes[row, col], all_records, present, xk, yk, xl, yl,
                          log_y=ly, colors=_colors, markers=_markers, linestyles=_ls)
        if row == 0 and col == 0:
            axes[row, col].legend(fontsize=9, loc="best")

    fig.suptitle(
        f"Figure 9: Assignment comparison  —  K = {K},  seed = {seed}",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, figures_dir / f"fig9_K_{K}_seed_{seed}.pdf")
    print(f"Saved combined: {figures_dir / f'fig9_K_{K}_seed_{seed}.pdf'}")

    # ── Individual panels ─────────────────────────────────────────────────
    joint_prefix = "score_joint" if jk == "score_joint" else "kl_joint"
    panel_names = [
        f"{joint_prefix}_vs_nsim", f"{joint_prefix}_vs_time", f"{joint_prefix}_vs_epsilon",
        "kl_sigma2_vs_nsim", "kl_sigma2_vs_time", "kl_sigma2_vs_epsilon",
    ]
    for (xk, yk, xl, yl), name in zip(panels, panel_names):
        fig_s, ax_s = plt.subplots(figsize=(7, 5))
        ly = log_y_joint if yk == jk else True
        plot_method_panel(ax_s, all_records, present, xk, yk, xl, yl,
                          log_y=ly, colors=_colors, markers=_markers, linestyles=_ls)
        ax_s.legend(fontsize=9)
        fig_s.tight_layout()
        save_figure(fig_s, figures_dir / f"fig9_{name}_K_{K}_seed_{seed}.pdf")

    # ── Raw cost panels ───────────────────────────────────────────────────
    raw_panels = [
        ("n_sim_raw", jk,  r"$N_{\mathrm{sim}}$ (total)",  jl),
        ("time_raw",  jk,  "Time (s, total)",               jl),
        ("n_sim_raw", "epsilon",   r"$N_{\mathrm{sim}}$ (total)",  r"$\varepsilon$"),
        ("time_raw",  "epsilon",   "Time (s, total)",               r"$\varepsilon$"),
        ("n_sim_raw", "kl_sigma2", r"$N_{\mathrm{sim}}$ (total)",  r"$\mathrm{KL}_{\sigma^2}$"),
        ("time_raw",  "kl_sigma2", "Time (s, total)",               r"$\mathrm{KL}_{\sigma^2}$"),
    ]
    fig_r, axes_r = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (xk, yk, xl, yl) in enumerate(raw_panels):
        row, col = divmod(idx, 3)
        ly = log_y_joint if yk == jk else True
        plot_method_panel(axes_r[row, col], all_records, present, xk, yk, xl, yl,
                          log_y=ly, colors=_colors, markers=_markers, linestyles=_ls)
        if row == 0 and col == 0:
            axes_r[row, col].legend(fontsize=9, loc="best")
    fig_r.suptitle(
        f"Figure 9 (raw cost): Assignment comparison  —  K = {K},  seed = {seed}",
        fontsize=14, y=1.01,
    )
    fig_r.tight_layout()
    save_figure(fig_r, figures_dir / f"fig9_raw_K_{K}_seed_{seed}.pdf")
    print(f"Saved raw-cost: {figures_dir / f'fig9_raw_K_{K}_seed_{seed}.pdf'}")

    # ── Epsilon vs N_sim / Time (standalone) ──────────────────────────────
    for xk, xl, tag in [
        ("n_sim", r"$N_{\mathrm{sim}}$ (per 1000 unique)", "nsim"),
        ("time",  "Time (s, per 1000 unique)", "time"),
        ("n_sim_raw", r"$N_{\mathrm{sim}}$ (total)", "nsim_raw"),
        ("time_raw",  "Time (s, total)", "time_raw"),
    ]:
        fig_e, ax_e = plt.subplots(figsize=(7, 5))
        plot_method_panel(ax_e, all_records, present, xk, "epsilon",
                          xl, r"$\varepsilon$",
                          colors=_colors, markers=_markers, linestyles=_ls)
        ax_e.legend(fontsize=9)
        fig_e.tight_layout()
        save_figure(fig_e, figures_dir / f"fig9_epsilon_vs_{tag}_K_{K}_seed_{seed}.pdf")

    print(f"Saved all individual panels in {figures_dir}")


# ── Multi-K sweep ─────────────────────────────────────────────────────────────

def generate_multi_K_figures(all_K_records, Ks, seed, figures_dir):
    """Generate multi-K overlay plots: one subplot per K."""
    methods = [name for name, _ in ASSIGNMENT_CONFIGS]
    _colors = ASSIGNMENT_COLORS
    _markers = ASSIGNMENT_MARKERS
    _ls = ASSIGNMENT_LINESTYLES

    n_K = len(Ks)
    ncols = min(n_K, 4)
    nrows = (n_K + ncols - 1) // ncols

    for ykey, yl_func, xk, xl, fname_tag in [
        (None, None, "time_raw", "Time (s)", "joint_vs_time"),
        (None, None, "n_sim_raw", r"$N_{\mathrm{sim}}$", "joint_vs_nsim"),
        ("epsilon", r"$\varepsilon$", "time_raw", "Time (s)", "epsilon_vs_time"),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
        for i, K in enumerate(Ks):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            recs = [rec for rec in all_K_records if rec.get("K") == K]
            present = [m for m in methods if any(rec.get("method") == m for rec in recs)]

            if ykey is None:
                jk = detect_joint_key(recs)
                jl = joint_ylabel(jk)
                ly = jk != "score_joint"
                yk, yl_str = jk, jl
            else:
                yk, yl_str, ly = ykey, yl_func, True

            plot_method_panel(ax, recs, present, xk, yk, xl, yl_str,
                              log_y=ly, colors=_colors, markers=_markers, linestyles=_ls)
            ax.set_title(f"K = {K}", fontsize=12)
            if i == 0:
                ax.legend(fontsize=8, loc="best")

        for j in range(i + 1, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle(f"Figure 9: {fname_tag.replace('_', ' ')} — varying K (seed={seed})",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        save_figure(fig, figures_dir / f"fig9_multiK_{fname_tag}_seed_{seed}.pdf")
        print(f"Saved multi-K figure: fig9_multiK_{fname_tag}_seed_{seed}.pdf")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description="Figure 9: Assignment comparison")
    parser.add_argument("--K", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N_particles", type=int, default=1000)
    parser.add_argument("--N_sim_max", type=float, default=None)
    parser.add_argument("--n_obs", type=int, default=10)
    parser.add_argument("--stopping_rate", type=float, default=0.0)
    parser.add_argument("--rerun", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    figures_dir = _PROJECT_ROOT / "experiments" / "figures" / "fig9"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = _PROJECT_ROOT / "experiments" / "results" / "assignment_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    Ks = sorted(args.K)
    seed = args.seed

    # ── Rerun mode ─────────────────────────────────────────────────────────
    if args.rerun:
        print(f"Loading from {args.rerun}")
        with open(args.rerun, "rb") as f:
            saved = pickle.load(f)
        all_K_records = saved["all_records"]
        Ks = saved["Ks"]
        seed = saved["seed"]

        for K in Ks:
            recs_K = [r for r in all_K_records if r.get("K") == K]
            generate_figures(recs_K, K, seed, figures_dir)
        if len(Ks) > 1:
            generate_multi_K_figures(all_K_records, Ks, seed, figures_dir)
        print("Done (rerun).")
        return

    # ── Run experiments ────────────────────────────────────────────────────
    all_K_records = []

    for K in Ks:
        print(f"\n{'='*60}")
        print(f"  K = {K}")
        print(f"{'='*60}")

        N_sim_max = args.N_sim_max if args.N_sim_max else K * 1_000_000
        model, y_obs, true_theta, key = setup_experiment(K, seed, n_obs=args.n_obs)

        records_K = []

        for name, cfg in ASSIGNMENT_CONFIGS:
            print(f"\n  --- {name}  (cascade={cfg['cascade']}) ---")
            t0 = time_mod.time()
            key, out, rows = run_single(
                key, model, y_obs, K, args.N_particles, N_sim_max,
                args.stopping_rate,
                cascade=cfg["cascade"],
            )
            wall = time_mod.time() - t0
            print(f"  {name}: {len(rows)} iterations, wall = {wall:.1f}s")

            for r in rows:
                r["method"] = name
                r["K"] = K
            records_K.extend(rows)

        all_K_records.extend(records_K)
        generate_figures(records_K, K, seed, figures_dir)

    # ── Save pickle ────────────────────────────────────────────────────────
    pkl_path = results_dir / f"fig9_assignment_Ks_{'_'.join(map(str,Ks))}_seed_{seed}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"all_records": all_K_records, "Ks": Ks, "seed": seed}, f)
    print(f"\nSaved results to {pkl_path}")

    # ── Multi-K figures ────────────────────────────────────────────────────
    if len(Ks) > 1:
        generate_multi_K_figures(all_K_records, Ks, seed, figures_dir)

    print("\nFigure 9 generation complete.")


if __name__ == "__main__":
    main()
