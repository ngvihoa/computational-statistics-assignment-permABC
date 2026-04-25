#!/usr/bin/env python3
"""Plot weather rain probability inference results.

This script is intentionally separate from run_weather_rain_probability.py, in
the same spirit as the SIR real-world figure scripts.

Usage:
    python3 my-reproduces/scripts_real_world/plot_weather_rain_probability.py --seed 42 --scale regional
    python3 my-reproduces/scripts_real_world/plot_weather_rain_probability.py --seed 42 --scales national regional provincial
    python3 my-reproduces/scripts_real_world/plot_weather_rain_probability.py --seed 42 --scales national regional provincial --include_extra
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHODS = [
    ("permABC-SMC", "perm_smc", "#1f77b4"),
    ("ABC-SMC", "abc_smc", "#8c564b"),
    ("ABC-SMC (Gibbs 3b)", "abc_smc_g3", "#d62728"),
    ("permABC-SMC (Gibbs 3b)", "perm_smc_g3", "#2ca02c"),
    ("ABC-Gibbs", "abc_gibbs_true", "#ff7f0e"),
    ("permABC-SMC-OS", "perm_smc_os", "#e377c2"),
    ("permABC-SMC-UM", "perm_smc_um", "#9467bd"),
]


def load_results(results_dir: Path, seed: int, scale: str) -> Dict[str, dict]:
    loaded = {}
    for display_name, tag, _ in METHODS:
        path = results_dir / f"inference_weather_rain_{scale}_{tag}_seed_{seed}.pkl"
        if not path.exists():
            print(f"  Not found: {display_name} ({path.name})")
            continue

        with path.open("rb") as file:
            data = pickle.load(file)
        loaded[display_name] = data
        n_part = "?"
        theta = data.get("Thetas_final")
        if theta is not None and hasattr(theta, "glob"):
            n_part = np.asarray(theta.glob).shape[0]
        print(
            f"  Loaded {display_name}: {data.get('n_iterations', '?')} iters, "
            f"N_sim={data.get('total_n_sim', '?'):,}, n_particles={n_part}"
        )
    return loaded


def _summary_frame(all_data: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for method, data in all_data.items():
        rows.append(
            {
                "method": method,
                "n_iterations": float(data.get("n_iterations", np.nan)),
                "total_n_sim": float(data.get("total_n_sim", np.nan)),
                "time_final": float(data.get("time_final", np.nan)),
                "final_epsilon": float(data.get("final_epsilon", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _positive_log_y(ax, values) -> None:
    values = np.asarray(values, dtype=float)
    if np.any(np.isfinite(values) & (values > 0)):
        ax.set_yscale("log")


def _plot_smooth_posterior(ax, values, color: str, label: str, bins: int = 35) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    if values.size < 2 or np.allclose(values, values[0]):
        ax.hist(
            values,
            bins=max(5, min(bins, max(1, values.size))),
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
            alpha=0.95,
        )
        return

    grid_min = float(np.min(values))
    grid_max = float(np.max(values))
    span = grid_max - grid_min
    if span <= 0:
        span = max(1.0, abs(grid_min) * 0.1)

    grid = np.linspace(grid_min - 0.1 * span, grid_max + 0.1 * span, 400)

    try:
        kde = gaussian_kde(values)
        density = kde(grid)
        ax.plot(grid, density, color=color, linewidth=2.2, label=label)
    except Exception:
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
            alpha=0.95,
        )


def _infer_k(all_data: Dict[str, dict]) -> int | str:
    for data in all_data.values():
        theta = data.get("Thetas_final")
        if theta is not None and hasattr(theta, "loc"):
            loc = np.asarray(theta.loc)
            if loc.ndim >= 2:
                return int(loc.shape[1])
        meta = data.get("meta", {})
        labels = meta.get("labels")
        if labels:
            return len(labels)
    return "?"


def _title(subject: str, scale: str, seed: int, k_value: int | str) -> str:
    return f"{subject} ({scale}, K={k_value}; seed={seed})"


def _bar_metric(all_data: Dict[str, dict], scale: str, seed: int, metric: str, title: str, ylabel: str):
    df = _summary_frame(all_data).sort_values("final_epsilon")
    x = np.arange(len(df))
    labels = df["method"].tolist()
    values = df[metric].to_numpy(dtype=float)
    k_value = _infer_k(all_data)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.suptitle(_title(title, scale, seed, k_value), fontsize=13, fontweight="bold")
    ax.bar(x, values, color="#4C78A8", alpha=0.9)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    _positive_log_y(ax, values)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_final_epsilon(all_data: Dict[str, dict], scale: str, seed: int):
    return _bar_metric(all_data, scale, seed, "final_epsilon", "Final epsilon by method", "Final epsilon")


def plot_total_nsim(all_data: Dict[str, dict], scale: str, seed: int):
    return _bar_metric(all_data, scale, seed, "total_n_sim", "Simulation cost by method", "Total N_sim")


def plot_runtime(all_data: Dict[str, dict], scale: str, seed: int):
    return _bar_metric(all_data, scale, seed, "time_final", "Runtime by method", "Seconds")


def plot_epsilon_by_iteration(all_data: Dict[str, dict], scale: str, seed: int):
    colors = {name: color for name, _, color in METHODS}
    k_value = _infer_k(all_data)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.suptitle(_title("Epsilon by iteration", scale, seed, k_value), fontsize=13, fontweight="bold")

    for method_name, data in all_data.items():
        eps = np.asarray(data.get("Eps_values", []), dtype=float)
        if len(eps) == 0:
            continue
        steps = np.arange(len(eps))
        mask_steps = np.isfinite(eps) & (eps > 0)
        color = colors.get(method_name, "gray")

        if np.any(mask_steps):
            ax.plot(steps[mask_steps], eps[mask_steps], marker="o", linewidth=1.8, color=color, label=method_name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Epsilon")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_epsilon_by_nsim(all_data: Dict[str, dict], scale: str, seed: int):
    colors = {name: color for name, _, color in METHODS}
    k_value = _infer_k(all_data)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.suptitle(_title("Epsilon by cumulative simulations", scale, seed, k_value), fontsize=13, fontweight="bold")

    for method_name, data in all_data.items():
        eps = np.asarray(data.get("Eps_values", []), dtype=float)
        n_sim = np.asarray(data.get("N_sim", []), dtype=float)
        if len(eps) == 0:
            continue
        if len(n_sim) == 0:
            n_sim = np.ones(len(eps), dtype=float)

        n = min(len(eps), len(n_sim))
        eps = eps[:n]
        n_sim_cum = np.cumsum(n_sim[:n])
        mask = np.isfinite(eps) & (eps > 0) & np.isfinite(n_sim_cum) & (n_sim_cum > 0)
        color = colors.get(method_name, "gray")

        if np.any(mask):
            ax.plot(n_sim_cum[mask], eps[mask], marker="o", linewidth=1.8, color=color, label=method_name)

    ax.set_xlabel("Cumulative N_sim")
    ax.set_ylabel("Epsilon")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_posterior_alpha(all_data: Dict[str, dict], scale: str, seed: int):
    colors = {name: color for name, _, color in METHODS}
    k_value = _infer_k(all_data)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.suptitle(_title("Posterior local intercepts", scale, seed, k_value), fontsize=13, fontweight="bold")

    for method_name, data in all_data.items():
        theta = data.get("Thetas_final")
        if theta is None:
            continue
        color = colors.get(method_name, "gray")

        alpha = np.asarray(theta.loc, dtype=float)
        if alpha.size:
            _plot_smooth_posterior(ax, alpha.reshape(-1), color=color, label=method_name)

    ax.set_xlabel("Component intercept alpha")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_posterior_beta(all_data: Dict[str, dict], scale: str, seed: int):
    colors = {name: color for name, _, color in METHODS}
    k_value = _infer_k(all_data)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.suptitle(_title("Posterior feature coefficients", scale, seed, k_value), fontsize=13, fontweight="bold")

    for method_name, data in all_data.items():
        theta = data.get("Thetas_final")
        if theta is None:
            continue
        color = colors.get(method_name, "gray")

        beta = np.asarray(theta.glob, dtype=float)
        if beta.size:
            _plot_smooth_posterior(ax, beta.reshape(-1), color=color, label=method_name)

    ax.set_xlabel("Feature coefficient beta")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_posterior_parameters(all_data: Dict[str, dict], scale: str, seed: int):
    """Combined posterior view: alpha (left) and beta (right)."""
    colors = {name: color for name, _, color in METHODS}
    k_value = _infer_k(all_data)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.suptitle(
        _title("Posterior parameter distributions", scale, seed, k_value),
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    for method_name, data in all_data.items():
        theta = data.get("Thetas_final")
        if theta is None:
            continue
        color = colors.get(method_name, "gray")

        alpha = np.asarray(theta.loc, dtype=float)
        if alpha.size:
            _plot_smooth_posterior(ax_left, alpha.reshape(-1), color=color, label=method_name)

        beta = np.asarray(theta.glob, dtype=float)
        if beta.size:
            _plot_smooth_posterior(ax_right, beta.reshape(-1), color=color, label=method_name)

    ax_left.set_title("Local intercepts alpha")
    ax_left.set_xlabel("alpha")
    ax_left.set_ylabel("Density")
    ax_left.grid(alpha=0.25)

    ax_right.set_title("Global coefficients beta")
    ax_right.set_xlabel("beta")
    ax_right.set_ylabel("Density")
    ax_right.grid(alpha=0.25)

    handles, labels = ax_right.get_legend_handles_labels()
    if handles:
        # Keep legend under title to avoid overlap when saving with tight bbox.
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=3,
            fontsize=8,
            frameon=False,
        )

    # Reserve explicit headroom for title + legend.
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    return fig


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot weather rain probability ABC inference results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=None,
        choices=["national", "regional", "provincial"],
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["national", "regional", "provincial"],
        help="Single scale shortcut.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="my-reproduces/results/weather_rain_inference",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="my-reproduces/results/weather_rain_inference/figures",
    )
    parser.add_argument(
        "--include_extra",
        action="store_true",
        help="Also save extra diagnostic plots (N_sim and epsilon trajectories).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scales = [args.scale] if args.scale else (args.scales or ["national", "regional", "provincial"])
    results_dir = REPO_ROOT / args.results_dir
    figures_dir = REPO_ROOT / args.figures_dir

    for scale in scales:
        print(f"\nWeather rain figures: scale={scale}, seed={args.seed}")
        all_data = load_results(results_dir, args.seed, scale)
        if not all_data:
            print("  No results found for this scale.")
            continue

        # Default outputs follow the 3 requested comparisons:
        # 1) algorithm runtime, 2) posterior distributions, 3) final epsilon.
        figures = [
            ("runtime", plot_runtime(all_data, scale, args.seed)),
            ("final_epsilon", plot_final_epsilon(all_data, scale, args.seed)),
            ("posterior_parameters", plot_posterior_parameters(all_data, scale, args.seed)),
        ]

        if args.include_extra:
            figures.extend([
                ("total_nsim", plot_total_nsim(all_data, scale, args.seed)),
                ("epsilon_by_iteration", plot_epsilon_by_iteration(all_data, scale, args.seed)),
                ("epsilon_by_nsim", plot_epsilon_by_nsim(all_data, scale, args.seed)),
                ("posterior_alpha", plot_posterior_alpha(all_data, scale, args.seed)),
                ("posterior_beta", plot_posterior_beta(all_data, scale, args.seed)),
            ])

        for suffix, fig in figures:
            save_figure(
                fig,
                figures_dir / f"weather_rain_{scale}_{suffix}_seed_{args.seed}.png",
            )


if __name__ == "__main__":
    main()
