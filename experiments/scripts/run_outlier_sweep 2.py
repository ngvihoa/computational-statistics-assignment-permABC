#!/usr/bin/env python3
"""
Sweep over K_outliers (K') from 0 to max_K_outliers for K=20.

For each K', runs (or reuses) run_performance_comparison.py with --no-osum,
then runs ABC-Gibbs and produces:
  1. Per-K' performance figures (fig4bis style, 2x3 panels)
  2. Degradation figure: final KL_sigma2 / score_joint vs K' for each method
  3. Relative efficiency: ratio permABC/ABC-SMC vs K'

Usage:
    python run_outlier_sweep.py --K 20 --seed 42
    python run_outlier_sweep.py --K 20 --K-outliers-range 0 5 --seed 42
"""

import os
import sys
import argparse
import subprocess
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root ──────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = None
for _p in _THIS.parents:
    if (_p / "pyproject.toml").exists():
        _PROJECT_ROOT = _p
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Style (same as fig4bis) ───────────────────────────────────────────────────
COLORS = {
    "ABC-Vanilla": "#d62728",
    "permABC-Vanilla": "#2ca02c",
    "ABC-SMC": "#ff7f0e",
    "ABC-PMC": "#ffbb78",
    "ABC-Gibbs": "#8c564b",
    "permABC-SMC": "#1f77b4",
}
MARKERS = {
    "ABC-Vanilla": "s",
    "permABC-Vanilla": "s",
    "ABC-SMC": "o",
    "ABC-PMC": "o",
    "ABC-Gibbs": "v",
    "permABC-SMC": "o",
}
LINESTYLES = {
    "ABC-Vanilla": "-",
    "permABC-Vanilla": "-",
    "ABC-SMC": "--",
    "ABC-PMC": "--",
    "ABC-Gibbs": "-.",
    "permABC-SMC": "--",
}
METHOD_ORDER = ["ABC-Vanilla", "permABC-Vanilla", "ABC-SMC", "ABC-PMC", "ABC-Gibbs", "permABC-SMC"]
METHODS_EXCLUDE = {"permABC-SMC-OS", "permABC-SMC-UM"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="K' outlier sweep")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--K-outliers-range", type=int, nargs=2, default=[0, 5],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N_points", type=int, default=1_000_000)
    parser.add_argument("--N_particles", type=int, default=1000)
    return parser.parse_args()


def load_or_run(K, K_outliers, seed, N_points, N_particles, results_dir):
    """Load existing pickle or run the experiment."""
    pkl_path = results_dir / f"performance_K_{K}_outliers_{K_outliers}_osum_False_seed_{seed}.pkl"
    if pkl_path.exists():
        print(f"  Found existing: {pkl_path.name}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data, pkl_path

    print(f"  Running experiment for K'={K_outliers}...")
    main_script = _PROJECT_ROOT / "experiments" / "scripts" / "run_performance_comparison.py"
    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_PROJECT_ROOT) + (":" + pypath if pypath else "")
    cmd = [
        sys.executable, str(main_script),
        "--K", str(K), "--K_outliers", str(K_outliers),
        "--seed", str(seed), "--N_points", str(N_points),
        "--N_particles", str(N_particles),
        "--no-osum", "--output-dir", str(_PROJECT_ROOT),
    ]
    subprocess.run(cmd, check=True, cwd=str(_PROJECT_ROOT), env=env)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data, pkl_path


def extract_records(data):
    """Extract records from pickle, excluding OS/UM."""
    df = data.get("summary_df")
    if df is None:
        return []
    if hasattr(df, "to_dict"):
        records = df.to_dict(orient="records")
    else:
        records = list(df)
    return [r for r in records if r.get("method") not in METHODS_EXCLUDE]


def add_gibbs(records, data, K, seed):
    """Add ABC-Gibbs checkpoints if not already present."""
    if any(r.get("method") == "ABC-Gibbs" for r in records):
        return records

    try:
        scripts_dir = _THIS.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from abc_gibbs_gaussian import run_gibbs_for_benchmark
        from jax import random as jrandom

        exp_setup = data.get("experiment_setup", {})
        model_obj = exp_setup.get("model")
        if model_obj is None:
            model_obj = data.get("model")
        y_obs_obj = exp_setup.get("y_obs")
        if y_obs_obj is None:
            y_obs_obj = data.get("y_obs")

        if model_obj is not None and y_obs_obj is not None:
            perm_nsims = [r.get("n_sim_raw") or r.get("n_sim", 0) for r in records
                          if r.get("method") == "permABC-SMC"]
            perm_nsims = [x for x in perm_nsims if x and float(x) > 0]
            budget = int(max(perm_nsims)) if perm_nsims else 500_000
            gibbs_key = jrandom.PRNGKey(seed + 999)
            gibbs_records = run_gibbs_for_benchmark(
                gibbs_key, model_obj, y_obs_obj, K, budget,
            )
            records.extend(gibbs_records)
            print(f"    Added {len(gibbs_records)} ABC-Gibbs checkpoints")
        else:
            print("    WARNING: model/y_obs not in pickle")
    except Exception as e:
        print(f"    WARNING: ABC-Gibbs failed: {e}")

    return records


def get_final_metric(records, method, metric):
    """Get the final (lowest epsilon or best diagnostic) value for a method."""
    vals = []
    for r in records:
        if r.get("method") != method:
            continue
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return np.nan
    if metric == "epsilon":
        return min(vals)
    if metric in ("kl_sigma2", "score_joint"):
        # Return the value at the last checkpoint (highest n_sim)
        best_r = None
        best_nsim = -1
        for r in records:
            if r.get("method") != method:
                continue
            nsim = r.get("n_sim_raw") or r.get("n_sim", 0)
            if nsim and float(nsim) > best_nsim:
                best_nsim = float(nsim)
                best_r = r
        if best_r is not None:
            v = best_r.get(metric)
            if v is not None and np.isfinite(v):
                return float(v)
    return np.nan


def main():
    args = parse_arguments()
    K = args.K
    seed = args.seed
    K_min, K_max = args.K_outliers_range
    K_outliers_range = list(range(K_min, K_max + 1))

    results_dir = _PROJECT_ROOT / "experiments" / "results" / "performance_comparison"
    figures_dir = _PROJECT_ROOT / "experiments" / "figures" / "outlier_sweep"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect data for each K' ──────────────────────────────────────────
    all_data = {}
    for K_out in K_outliers_range:
        print(f"\nK' = {K_out}:")
        data, pkl_path = load_or_run(K, K_out, seed, args.N_points, args.N_particles, results_dir)
        records = extract_records(data)
        records = add_gibbs(records, data, K, seed)
        all_data[K_out] = records
        print(f"    Methods: {sorted(set(r.get('method') for r in records))}")
        print(f"    Records: {len(records)}")

    # ── Degradation figures ───────────────────────────────────────────────
    methods = [m for m in METHOD_ORDER
               if any(any(r.get("method") == m for r in recs) for recs in all_data.values())]

    for metric, ylabel, tag in [
        ("kl_sigma2", r"$\mathrm{KL}_{\sigma^2}$ (final)", "kl_sigma2"),
        ("score_joint", r"$-\mathbb{E}_q[\log p^*(\theta \mid y)]$ (final)", "score_joint"),
        ("epsilon", r"$\varepsilon$ (final)", "epsilon"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in methods:
            vals = [get_final_metric(all_data[ko], m, metric) for ko in K_outliers_range]
            if all(np.isnan(v) for v in vals):
                continue
            ax.plot(K_outliers_range, vals,
                    label=m, color=COLORS.get(m, "gray"),
                    marker=MARKERS.get(m, "o"),
                    linestyle=LINESTYLES.get(m, "-"),
                    linewidth=2, markersize=7)
        ax.set_xlabel(r"$K'$ (number of outlier components)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(K_outliers_range)
        if metric != "score_joint":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(f"Degradation with outliers — K={K}, seed={seed}", fontsize=12)
        fig.tight_layout()
        out = figures_dir / f"degradation_{tag}_K_{K}_seed_{seed}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"\nSaved: {out}")

    # ── Relative efficiency: permABC-SMC / ABC-SMC ────────────────────────
    ref_method = "ABC-SMC"
    comp_methods = ["permABC-SMC", "permABC-Vanilla", "ABC-Gibbs"]
    for metric, ylabel, tag in [
        ("kl_sigma2", r"Ratio $\mathrm{KL}_{\sigma^2}$", "kl_sigma2"),
        ("score_joint", r"Ratio score$_{\mathrm{joint}}$", "score_joint"),
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
                    label=f"{m} / {ref_method}",
                    color=COLORS.get(m, "gray"),
                    marker=MARKERS.get(m, "o"),
                    linestyle=LINESTYLES.get(m, "-"),
                    linewidth=2, markersize=7)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(r"$K'$ (number of outlier components)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(K_outliers_range)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(f"Relative efficiency vs ABC-SMC — K={K}, seed={seed}", fontsize=12)
        fig.tight_layout()
        out = figures_dir / f"relative_{tag}_K_{K}_seed_{seed}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {out}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Summary: final KL_sigma2 for each method and K'")
    print(f"{'='*70}")
    header = f"{'Method':<20}" + "".join(f"K'={ko:>3}" + " " * 6 for ko in K_outliers_range)
    print(header)
    print("-" * len(header))
    for m in methods:
        vals = [get_final_metric(all_data[ko], m, "kl_sigma2") for ko in K_outliers_range]
        line = f"{m:<20}" + "".join(f"{v:>9.4f}" for v in vals)
        print(line)

    print("\nDone.")


if __name__ == "__main__":
    main()
