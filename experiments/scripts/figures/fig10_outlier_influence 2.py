#!/usr/bin/env python3
"""
Figure 10: Influence of K' (outlier components) on performance.

Loads existing results for K=20 with varying K' and produces:
  1. Degradation plots: final metric vs K' for each method
  2. Per-K' performance panels: KL_sigma2 vs Nsim/Time for each K'
  3. Relative efficiency: ratio vs ABC-SMC

Usage:
    python fig10_outlier_influence.py
    python fig10_outlier_influence.py --K 20 --seed 42
    python fig10_outlier_influence.py --osum          # use osum=True pickles
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Shared plot config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import (
    METHOD_COLORS, METHOD_MARKERS, METHOD_LINESTYLES, METHOD_ORDER,
    METHODS_EXCLUDE_NO_OSUM, setup_matplotlib, save_figure,
    records_from_pkl, extract_series, plot_method_panel, find_project_root,
)

setup_matplotlib()

_PROJECT_ROOT = find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_filter_records(pkl_path, exclude=None):
    """Load records from pickle, optionally excluding methods."""
    records, data = records_from_pkl(pkl_path)
    if exclude:
        records = [r for r in records if r.get("method") not in exclude]
    return records, data


def add_gibbs(records, data, K, seed):
    if any(r.get("method") == "ABC-Gibbs" for r in records):
        return records
    try:
        scripts_dir = Path(__file__).resolve().parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from abc_gibbs_gaussian import run_gibbs_for_benchmark
        from jax import random as jrandom

        exp_setup = data.get("experiment_setup", {})
        model_obj = exp_setup.get("model") or data.get("model")
        y_obs_obj = exp_setup.get("y_obs") or data.get("y_obs")
        if model_obj is None or y_obs_obj is None:
            return records

        perm_nsims = [r.get("n_sim_raw") or r.get("n_sim", 0) for r in records
                      if r.get("method") == "permABC-SMC"]
        perm_nsims = [x for x in perm_nsims if x and float(x) > 0]
        budget = int(max(perm_nsims)) if perm_nsims else 500_000
        gibbs_key = jrandom.PRNGKey(seed + 999)
        gibbs_recs = run_gibbs_for_benchmark(gibbs_key, model_obj, y_obs_obj, K, budget)
        records.extend(gibbs_recs)
        print(f"    +{len(gibbs_recs)} ABC-Gibbs checkpoints")
    except Exception as e:
        print(f"    WARNING: ABC-Gibbs failed: {e}")
    return records


def get_final_metric(records, method, metric):
    """Return the metric value at the last checkpoint (highest n_sim)."""
    best_r, best_nsim = None, -1
    for r in records:
        if r.get("method") != method:
            continue
        nsim = r.get("n_sim_raw") or r.get("n_sim", 0)
        if nsim is None:
            continue
        if float(nsim) > best_nsim:
            best_nsim = float(nsim)
            best_r = r
    if best_r is None:
        return np.nan
    v = best_r.get(metric)
    if v is None or not np.isfinite(v):
        return np.nan
    return float(v)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description="Figure 10: K' influence on performance")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--osum", action="store_true", help="Use osum=True pickles")
    parser.add_argument("--K-outliers", type=int, nargs="+", default=None,
                        help="Specific K' values (default: auto-detect)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    K, seed = args.K, args.seed
    osum_str = "True" if args.osum else "False"

    results_dir = _PROJECT_ROOT / "experiments" / "results" / "performance_comparison"
    figures_dir = _PROJECT_ROOT / "experiments" / "figures" / "fig10"
    figures_dir.mkdir(parents=True, exist_ok=True)

    exclude = None if args.osum else METHODS_EXCLUDE_NO_OSUM

    # ── Auto-detect available K' values ───────────────────────────────────
    if args.K_outliers:
        K_outliers_range = sorted(args.K_outliers)
    else:
        K_outliers_range = []
        for ko in range(K + 1):
            pkl = results_dir / f"performance_K_{K}_outliers_{ko}_osum_{osum_str}_seed_{seed}.pkl"
            if pkl.exists():
                K_outliers_range.append(ko)

    if len(K_outliers_range) < 2:
        print(f"ERROR: Need at least 2 K' values. Found: {K_outliers_range}")
        print(f"Looking for: performance_K_{K}_outliers_*_osum_{osum_str}_seed_{seed}.pkl")
        sys.exit(1)

    print(f"K={K}, seed={seed}, osum={args.osum}")
    print(f"K' values found: {K_outliers_range}")

    # ── Load all data ─────────────────────────────────────────────────────
    all_data = {}
    for ko in K_outliers_range:
        pkl = results_dir / f"performance_K_{K}_outliers_{ko}_osum_{osum_str}_seed_{seed}.pkl"
        records, data = load_and_filter_records(pkl, exclude=exclude)
        print(f"  K'={ko}: {len(records)} records, methods={sorted(set(r.get('method') for r in records))}")
        records = add_gibbs(records, data, K, seed)
        all_data[ko] = records

    methods = [m for m in METHOD_ORDER
               if any(any(r.get("method") == m for r in recs) for recs in all_data.values())]
    print(f"Methods across all K': {methods}")

    tag = "osum" if args.osum else "noosum"

    # ── Figure 1: Degradation plots ──────────────────────────────────────
    metrics = [
        ("kl_sigma2",   r"$\mathrm{KL}_{\sigma^2}$ (final)",                      True),
        ("score_joint", r"$-\mathbb{E}_q[\log p^*(\theta \mid y)]$ (final)",       False),
        ("epsilon",     r"$\varepsilon$ (final)",                                   True),
    ]

    fig_deg, axes_deg = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, ylabel_str, log_y) in zip(axes_deg, metrics):
        for m in methods:
            vals = [get_final_metric(all_data[ko], m, metric) for ko in K_outliers_range]
            if all(np.isnan(v) for v in vals):
                continue
            ax.plot(K_outliers_range, vals,
                    label=m, color=METHOD_COLORS.get(m, "gray"),
                    marker=METHOD_MARKERS.get(m, "o"),
                    linestyle=METHOD_LINESTYLES.get(m, "-"),
                    linewidth=2, markersize=7)
        ax.set_xlabel(r"$K'$ (outlier components)", fontsize=11)
        ax.set_ylabel(ylabel_str, fontsize=11)
        ax.set_xticks(K_outliers_range)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes_deg[0].legend(fontsize=8, loc="best")
    fig_deg.suptitle(f"Performance degradation with K' outliers — K={K}, seed={seed}",
                     fontsize=13, y=1.02)
    fig_deg.tight_layout()
    save_figure(fig_deg, figures_dir / f"fig10_degradation_{tag}_K_{K}_seed_{seed}.pdf")
    print(f"\nSaved degradation figure")

    # ── Figure 2: Per-K' performance panels ──────────────────────────────
    n_K = len(K_outliers_range)
    ncols = min(n_K, 3)
    nrows = (n_K + ncols - 1) // ncols

    for ykey, ylabel_str, ytag, log_y in [
        ("kl_sigma2", r"$\mathrm{KL}_{\sigma^2}$", "kl_sigma2", True),
        ("score_joint", r"$-\mathbb{E}_q[\log p^*(\theta \mid y)]$", "score_joint", False),
    ]:
        for xkey, xlabel, xtag in [
            ("n_sim", r"$N_{\mathrm{sim}}$ (per 1000 unique)", "nsim"),
            ("time", "Time (s, per 1000 unique)", "time"),
        ]:
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
            for i, ko in enumerate(K_outliers_range):
                r, c = divmod(i, ncols)
                ax = axes[r][c]
                plot_method_panel(ax, all_data[ko], methods, xkey, ykey, xlabel, ylabel_str, log_y=log_y)
                ax.set_title(f"K' = {ko}", fontsize=11)
                if i == 0:
                    ax.legend(fontsize=7, loc="best")

            for j in range(i + 1, nrows * ncols):
                r, c = divmod(j, ncols)
                axes[r][c].set_visible(False)

            fig.suptitle(f"{ylabel_str} vs {xlabel} — K={K}, seed={seed}",
                         fontsize=13, y=1.02)
            fig.tight_layout()
            save_figure(fig, figures_dir / f"fig10_panels_{ytag}_vs_{xtag}_{tag}_K_{K}_seed_{seed}.pdf")
            print(f"Saved: fig10_panels_{ytag}_vs_{xtag}")

    # ── Figure 3: Relative efficiency vs ABC-SMC ─────────────────────────
    ref_method = "ABC-SMC"
    comp_methods = [m for m in methods if m != ref_method]

    for metric, ylabel_str, mtag in [
        ("kl_sigma2", r"$\mathrm{KL}_{\sigma^2}$ ratio", "kl_sigma2"),
        ("score_joint", r"Score ratio", "score_joint"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in comp_methods:
            ratios = []
            for ko in K_outliers_range:
                ref_val = get_final_metric(all_data[ko], ref_method, metric)
                comp_val = get_final_metric(all_data[ko], m, metric)
                if np.isfinite(ref_val) and np.isfinite(comp_val) and ref_val != 0:
                    ratios.append(comp_val / ref_val)
                else:
                    ratios.append(np.nan)
            if all(np.isnan(r) for r in ratios):
                continue
            ax.plot(K_outliers_range, ratios,
                    label=f"{m}",
                    color=METHOD_COLORS.get(m, "gray"),
                    marker=METHOD_MARKERS.get(m, "o"),
                    linestyle=METHOD_LINESTYLES.get(m, "-"),
                    linewidth=2, markersize=7)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="ABC-SMC (ref)")
        ax.set_xlabel(r"$K'$ (outlier components)", fontsize=11)
        ax.set_ylabel(f"{ylabel_str} (method / ABC-SMC)", fontsize=11)
        ax.set_xticks(K_outliers_range)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.set_title(f"Relative efficiency vs ABC-SMC — K={K}, seed={seed}", fontsize=12)
        fig.tight_layout()
        save_figure(fig, figures_dir / f"fig10_relative_{mtag}_{tag}_K_{K}_seed_{seed}.pdf")
        print(f"Saved: fig10_relative_{mtag}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Final KL_sigma2 by method and K'")
    print(f"{'='*70}")
    header = f"{'Method':<20}" + "".join(f"  K'={ko:<5}" for ko in K_outliers_range)
    print(header)
    print("-" * len(header))
    for m in methods:
        vals = [get_final_metric(all_data[ko], m, "kl_sigma2") for ko in K_outliers_range]
        line = f"{m:<20}" + "".join(f"  {v:<8.4f}" if np.isfinite(v) else "  N/A     " for v in vals)
        print(line)

    print("\nDone.")


if __name__ == "__main__":
    main()
