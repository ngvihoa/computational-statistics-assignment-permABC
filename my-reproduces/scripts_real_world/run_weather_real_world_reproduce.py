#!/usr/bin/env python3
"""
Apply permABC on real-world weather data.

This script follows the same high-level procedure as the real-world SIR runner:
1) Load and clean real data.
2) Build observations at multiple scales.
3) Run permABC-SMC and baselines on each scale.
4) Save lightweight outputs and a comparison table.

Modeling choice
---------------
We use GaussianWithNoSummaryStats because the weather file is a tabular panel.
For each component (province/region/country), we treat the daily values as i.i.d.
samples from a latent Gaussian mean + shared variance model. Therefore we sort
each component series before inference (distributional matching, not time-series dynamics).
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time as _time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from jax import random

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from permabc.algorithms.over_sampling import perm_abc_smc_os
from permabc.algorithms.smc import abc_smc, perm_abc_smc
from permabc.algorithms.under_matching import perm_abc_smc_um
from permabc.core.kernels import KernelTruncatedRW
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats


_METHOD_REGISTRY_TEMPLATE: Dict[str, Dict[str, str]] = {
    "permABC-SMC": {"tag": "perm_smc", "type": "perm_smc"},
    "ABC-SMC": {"tag": "abc_smc", "type": "abc_smc"},
    "ABC-SMC (Gibbs {H}b)": {"tag": "abc_smc_g{H}", "type": "abc_smc_gibbs"},
    "permABC-SMC (Gibbs {H}b)": {"tag": "perm_smc_g{H}", "type": "perm_smc_gibbs"},
    "permABC-SMC-OS": {"tag": "perm_smc_os", "type": "perm_smc_os"},
    "permABC-SMC-UM": {"tag": "perm_smc_um", "type": "perm_smc_um"},
}


def expand_registry(num_gibbs_blocks: int) -> Dict[str, Dict[str, str]]:
    """Expand the method template by replacing {H} with --num_gibbs_blocks."""
    registry: Dict[str, Dict[str, str]] = {}
    h_txt = str(num_gibbs_blocks)

    for name_tpl, info_tpl in _METHOD_REGISTRY_TEMPLATE.items():
        method_name = name_tpl.replace("{H}", h_txt)
        method_info = {
            key: value.replace("{H}", h_txt) if isinstance(value, str) else value
            for key, value in info_tpl.items()
        }
        registry[method_name] = method_info

    return registry


def slugify(text: str) -> str:
    """Create a compact ASCII-safe slug for file names."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def load_weather_dataframe(csv_path: Path, feature: str, country: str) -> pd.DataFrame:
    """Load weather panel data and keep the required columns only."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Weather dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"province", "region", "country", "date", feature}
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    df = df[["province", "region", "country", "date", feature]].copy()
    df = df[df["country"] == country].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

    # Remove rows with invalid dates/values to avoid NaNs in the pivot matrix.
    df = df.dropna(subset=["date", feature, "province", "region"])
    if len(df) == 0:
        raise ValueError("No rows available after cleaning/filtering weather dataset.")

    return df


def pivot_weather_observation(
    df: pd.DataFrame,
    feature: str,
    scale: str,
    max_days: int,
    max_components: int,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build observation matrix y_obs with shape (K, n_obs).

    scale = national   -> K = 1 (country-level daily average)
    scale = regional   -> K = n_regions
    scale = provincial -> K = n_provinces
    """
    if scale == "national":
        national = (
            df.groupby("date", as_index=False)[feature]
            .mean()
            .sort_values("date")
            .head(max_days)
        )
        values = national[feature].to_numpy(dtype=float)
        matrix = values[None, :]
        labels = ["national"]
        dates = [str(d.date()) for d in national["date"].tolist()]
        return matrix, labels, dates

    group_col = "region" if scale == "regional" else "province"

    # Aggregate duplicates by mean at (component, date), then pivot to wide matrix.
    grouped = (
        df.groupby([group_col, "date"], as_index=False)[feature]
        .mean()
    )

    pivot = grouped.pivot(index=group_col, columns="date", values=feature)
    pivot = pivot.sort_index(axis=0)
    pivot = pivot.sort_index(axis=1)

    # Keep a stable date window from the beginning of the dataset.
    if max_days > 0 and pivot.shape[1] > max_days:
        pivot = pivot.iloc[:, :max_days]

    # Keep only components with complete data on the retained dates.
    pivot = pivot.dropna(axis=0, how="any")

    # Optional speed control for very large K.
    if max_components > 0 and pivot.shape[0] > max_components:
        pivot = pivot.iloc[:max_components, :]

    if pivot.shape[0] == 0 or pivot.shape[1] == 0:
        raise ValueError(f"Scale '{scale}' has no complete matrix after filtering.")

    matrix = pivot.to_numpy(dtype=float)
    labels = [str(x) for x in pivot.index.tolist()]
    dates = [str(d.date()) for d in pivot.columns.tolist()]

    return matrix, labels, dates


def build_gaussian_model_and_obs(
    matrix_k_nobs: np.ndarray,
    mu_0: float,
    sigma_0: float,
    alpha: float,
    beta: float,
) -> Tuple[GaussianWithNoSummaryStats, np.ndarray]:
    """Instantiate Gaussian model and format observations for permABC.

    Important:
    - GaussianWithNoSummaryStats sorts simulated values along n_obs axis.
    - To compare like-with-like, we also sort observed values per component.
    """
    k_comp, n_obs = matrix_k_nobs.shape

    model = GaussianWithNoSummaryStats(
        K=k_comp,
        n_obs=n_obs,
        mu_0=mu_0,
        sigma_0=sigma_0,
        alpha=alpha,
        beta=beta,
    )

    y_obs = np.sort(matrix_k_nobs, axis=1)[None, :, :]
    return model, y_obs


def smc_result_to_lightweight(result: dict, method_name: str) -> dict:
    """Convert SMC outputs to a small and consistent storage format."""
    return {
        "Thetas_final": result["Thetas"][-1],
        "n_iterations": len(result["Eps_values"]),
        "total_n_sim": int(np.sum(result["N_sim"])),
        "time_final": float(result["time_final"]),
        "final_epsilon": float(result["Eps_values"][-1]),
        "Eps_values": result["Eps_values"],
        "N_sim": result["N_sim"],
        "method": method_name,
    }


def run_method(
    method_name: str,
    method_info: dict,
    key,
    model,
    y_obs,
    args,
    epsilon_from_perm_smc: float | None,
) -> dict:
    """Run one method and return the normalized lightweight result."""
    mtype = method_info["type"]

    common_kwargs = dict(
        key=key,
        model=model,
        y_obs=y_obs,
        epsilon_target=0.0,
        n_particles=args.n_particles,
        kernel=KernelTruncatedRW,
        verbose=args.verbose,
        parallel=True,
        Final_iteration=args.final_iteration,
    )

    if mtype == "perm_smc":
        return smc_result_to_lightweight(perm_abc_smc(**common_kwargs), method_name)

    if mtype == "abc_smc":
        result = abc_smc(
            key=key,
            model=model,
            y_obs=y_obs,
            epsilon_target=0.0,
            n_particles=args.n_particles,
            kernel=KernelTruncatedRW,
            verbose=args.verbose,
            Final_iteration=args.final_iteration,
        )
        return smc_result_to_lightweight(result, method_name)

    if mtype == "abc_smc_gibbs":
        result = abc_smc(
            key=key,
            model=model,
            y_obs=y_obs,
            epsilon_target=0.0,
            n_particles=args.n_particles,
            kernel=KernelTruncatedRW,
            verbose=args.verbose,
            num_blocks_gibbs=args.num_gibbs_blocks,
            Final_iteration=args.final_iteration,
        )
        return smc_result_to_lightweight(result, method_name)

    if mtype == "perm_smc_gibbs":
        result = perm_abc_smc(
            **common_kwargs,
            num_blocks_gibbs=args.num_gibbs_blocks,
        )
        return smc_result_to_lightweight(result, method_name)

    if mtype == "perm_smc_os":
        k_comp = model.K
        m0 = max(2 * k_comp, k_comp + 5)
        eps_start = epsilon_from_perm_smc if epsilon_from_perm_smc is not None else np.inf
        result = perm_abc_smc_os(
            key=key,
            model=model,
            y_obs=y_obs,
            n_particles=args.n_particles,
            kernel=KernelTruncatedRW,
            M_0=m0,
            epsilon=eps_start,
            verbose=args.verbose,
            Final_iteration=args.final_iteration,
        )
        return smc_result_to_lightweight(result, method_name)

    if mtype == "perm_smc_um":
        k_comp = model.K
        l0 = max(1, k_comp // 2)
        eps_start = epsilon_from_perm_smc if epsilon_from_perm_smc is not None else np.inf
        result = perm_abc_smc_um(
            key=key,
            model=model,
            y_obs=y_obs,
            n_particles=args.n_particles,
            kernel=KernelTruncatedRW,
            L_0=l0,
            epsilon=eps_start,
            verbose=args.verbose,
            Final_iteration=args.final_iteration,
        )
        return smc_result_to_lightweight(result, method_name)

    raise ValueError(f"Unknown method type: {mtype}")


def save_lightweight(
    result: dict,
    scale: str,
    feature: str,
    method_tag: str,
    metadata: dict,
    results_dir: Path,
) -> None:
    """Save one method result per (scale, feature, seed)."""
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_slug = slugify(feature)
    seed = metadata.get("seed", "unknown")

    path = results_dir / f"inference_weather_{scale}_{feature_slug}_{method_tag}_seed_{seed}.pkl"
    with path.open("wb") as f:
        pickle.dump(result, f)

    print(f"  Saved: {path}")


def save_summary(all_results: dict, metadata: dict, results_dir: Path) -> None:
    """Save cross-method summary CSV for quick benchmark comparison."""
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (method_name, scale), info in all_results.items():
        rows.append(
            {
                "method": method_name,
                "scale": scale,
                "n_iterations": info.get("n_iterations", ""),
                "total_n_sim": info.get("total_n_sim", ""),
                "time_final": info.get("time_final", ""),
                "final_epsilon": info.get("final_epsilon", ""),
            }
        )

    out_df = pd.DataFrame(rows)
    seed = metadata.get("seed", "unknown")
    feature_slug = slugify(metadata.get("feature", "feature"))
    csv_path = results_dir / f"comparison_weather_{feature_slug}_seed_{seed}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"  Summary saved: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply permABC to real-world weather data")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_particles", type=int, default=500)
    parser.add_argument("--final_iteration", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument("--feature", type=str, default="day.maxtemp_c")
    parser.add_argument("--country", type=str, default="Vietnam")
    parser.add_argument("--max_days", type=int, default=120)
    parser.add_argument("--max_components", type=int, default=0)

    parser.add_argument(
        "--scales",
        nargs="+",
        default=["national", "regional", "provincial"],
        choices=["national", "regional", "provincial"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to run; default is all registered methods.",
    )

    parser.add_argument("--num_gibbs_blocks", type=int, default=3)

    # Gaussian prior hyperparameters.
    parser.add_argument("--mu_0", type=float, default=0.0)
    parser.add_argument("--sigma_0", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=5.0)

    parser.add_argument(
        "--data_csv",
        type=str,
        default="my-reproduces/data/df_weather_clean.csv",
        help="Path to weather CSV file relative to repository root.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="my-reproduces/results/weather_real_world_inference",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"=== Weather permABC reproduction | seed={args.seed} | feature={args.feature} ===")

    registry = expand_registry(args.num_gibbs_blocks)
    if args.methods is None:
        methods_to_run = list(registry.keys())
    else:
        methods_to_run = args.methods
        unknown = [m for m in methods_to_run if m not in registry]
        if unknown:
            raise ValueError(f"Unknown method(s): {unknown}. Available: {list(registry.keys())}")

    csv_path = REPO_ROOT / args.data_csv
    results_dir = REPO_ROOT / args.results_dir

    weather_df = load_weather_dataframe(
        csv_path=csv_path,
        feature=args.feature,
        country=args.country,
    )

    # Pre-build observation matrices per scale once.
    scale_data = {}
    for scale in args.scales:
        matrix, labels, dates = pivot_weather_observation(
            df=weather_df,
            feature=args.feature,
            scale=scale,
            max_days=args.max_days,
            max_components=args.max_components,
        )
        model, y_obs = build_gaussian_model_and_obs(
            matrix_k_nobs=matrix,
            mu_0=args.mu_0,
            sigma_0=args.sigma_0,
            alpha=args.alpha,
            beta=args.beta,
        )
        scale_data[scale] = {
            "model": model,
            "y_obs": y_obs,
            "labels": labels,
            "dates": dates,
            "shape": matrix.shape,
        }

    key = random.PRNGKey(args.seed)

    metadata = {
        "seed": args.seed,
        "feature": args.feature,
        "country": args.country,
        "n_particles": args.n_particles,
        "final_iteration": args.final_iteration,
        "max_days": args.max_days,
        "max_components": args.max_components,
        "mu_0": args.mu_0,
        "sigma_0": args.sigma_0,
        "alpha": args.alpha,
        "beta": args.beta,
    }

    all_results: Dict[Tuple[str, str], dict] = {}

    for scale in args.scales:
        bundle = scale_data[scale]
        model = bundle["model"]
        y_obs = bundle["y_obs"]
        k_comp, n_obs = bundle["shape"]

        print("\n" + "=" * 72)
        print(f"Scale: {scale.upper()} | K={k_comp} | n_obs={n_obs}")
        print("=" * 72)

        # Run permABC-SMC first to transfer final epsilon to OS/UM starts.
        epsilon_from_perm_smc = None

        ordered_methods = list(methods_to_run)
        if "permABC-SMC" in ordered_methods:
            ordered_methods.remove("permABC-SMC")
            ordered_methods.insert(0, "permABC-SMC")

        for method_name in ordered_methods:
            method_info = registry[method_name]
            print(f"\n--- {method_name} ---")

            key, subkey = random.split(key)
            t0 = _time.perf_counter()

            try:
                result = run_method(
                    method_name=method_name,
                    method_info=method_info,
                    key=subkey,
                    model=model,
                    y_obs=y_obs,
                    args=args,
                    epsilon_from_perm_smc=epsilon_from_perm_smc,
                )

                elapsed = _time.perf_counter() - t0
                print(
                    f"  Done in {elapsed:.1f}s | "
                    f"N_sim={result.get('total_n_sim', 0):,} | "
                    f"eps_final={result.get('final_epsilon', '?')}"
                )

                if method_name == "permABC-SMC":
                    epsilon_from_perm_smc = result.get("final_epsilon", None)
                    print(f"  Calibration epsilon for OS/UM: {epsilon_from_perm_smc}")

                # Attach scale-specific metadata for reproducibility.
                result["meta"] = {
                    "labels": bundle["labels"],
                    "dates": bundle["dates"],
                    "scale": scale,
                }

                save_lightweight(
                    result=result,
                    scale=scale,
                    feature=args.feature,
                    method_tag=method_info["tag"],
                    metadata=metadata,
                    results_dir=results_dir,
                )
                all_results[(method_name, scale)] = result

            except Exception as exc:
                print(f"  FAILED: {exc}")
                import traceback

                traceback.print_exc()

    if all_results:
        save_summary(all_results=all_results, metadata=metadata, results_dir=results_dir)

    print("\n=== Completed weather real-world inference. ===")


if __name__ == "__main__":
    main()
