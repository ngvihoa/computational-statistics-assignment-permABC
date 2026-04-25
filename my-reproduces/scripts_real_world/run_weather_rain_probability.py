#!/usr/bin/env python3
"""
Apply permABC on real-world weather data for rain probability prediction.

This script follows the same high-level procedure as the real-world SIR runner:
1) Load and clean real data.
2) Build observations at multiple scales.
3) Run permABC-SMC and baselines on each scale.
4) Save lightweight outputs and a comparison table.

Modeling choice
---------------
We use BernoulliLogitWithCovariates to model binary rain probability (0/1) 
across multiple regions/provinces, using 5 normalized weather features as covariates:
- day.maxtemp_c
- day.maxwind_kph
- day.totalprecip_mm
- day.avghumidity
- day.uv

For each component (province/region/country), we treat the daily binary observations
as Bernoulli trials with logit-linear probabilities determined by component-specific
intercepts (alpha_k) and global feature coefficients (beta).
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
from permabc.models.bernoulli_logit_with_covariates import BernoulliLogitWithCovariates


# Weather feature columns to use as covariates
FEATURE_COLUMNS = [
    "day.maxtemp_c",
    "day.maxwind_kph",
    "day.totalprecip_mm",
    "day.avghumidity",
    "day.uv",
]

# Target variable for rain
TARGET_COLUMN = "day.daily_will_it_rain"


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


def load_weather_dataframe(csv_path: Path, country: str) -> pd.DataFrame:
    """Load weather panel data and validate required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Weather dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    required_cols = {"province", "region", "country", "date"} | set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Keep only needed columns
    cols_to_keep = ["province", "region", "country", "date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[cols_to_keep].copy()
    
    # Filter by country
    df = df[df["country"] == country].copy()

    # Clean dates and numeric columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with NaNs to avoid issues in pivot matrix
    df = df.dropna(subset=["date", "province", "region"] + FEATURE_COLUMNS + [TARGET_COLUMN])
    
    if len(df) == 0:
        raise ValueError("No rows available after cleaning/filtering weather dataset.")

    return df


def build_scale_data(
    df: pd.DataFrame,
    scale: str,
    max_days: int,
    max_components: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], dict]:
    """Build observation and covariate matrices at a given scale.
    
    Returns
    -------
    tuple
        (y_obs, X_cov, labels, dates, scaling_info)
        - y_obs: (K, n_obs) binary rain observations
        - X_cov: (K, n_obs, n_features) normalized weather features
        - labels: list of K component names
        - dates: list of dates
        - scaling_info: dict with scaler parameters for reproducibility
    """
    if scale == "national":
        # Aggregate to national level: average across all regions
        grouped = df.groupby("date", as_index=False).agg(
            {target: "mean" for target in [TARGET_COLUMN] + FEATURE_COLUMNS}
        ).sort_values("date")
        
        if max_days > 0 and len(grouped) > max_days:
            grouped = grouped.head(max_days)
        
        # Convert to matrices: (1, n_obs) for each
        dates = grouped["date"].dt.strftime("%Y-%m-%d").tolist()
        y_obs = grouped[TARGET_COLUMN].values.astype(np.float32)[np.newaxis, :]
        X_raw = grouped[FEATURE_COLUMNS].values.astype(np.float32)[np.newaxis, :, :]
        labels = ["national"]
        
    else:
        group_col = "region" if scale == "regional" else "province"
        
        # Aggregate by (group, date), then pivot to wide matrix
        agg_dict = {col: "mean" for col in [TARGET_COLUMN] + FEATURE_COLUMNS}
        grouped = df.groupby([group_col, "date"], as_index=False).agg(agg_dict)
        
        # Pivot to get (group, date) matrix for target
        pivot_target = grouped.pivot(index=group_col, columns="date", values=TARGET_COLUMN)
        pivot_target = pivot_target.sort_index(axis=0).sort_index(axis=1)
        
        # Pivot for each feature
        pivots_features = {}
        for feat in FEATURE_COLUMNS:
            pivot_feat = grouped.pivot(index=group_col, columns="date", values=feat)
            pivot_feat = pivot_feat.sort_index(axis=0).sort_index(axis=1)
            pivots_features[feat] = pivot_feat
        
        # Keep only dates and groups with complete data
        if max_days > 0 and pivot_target.shape[1] > max_days:
            pivot_target = pivot_target.iloc[:, :max_days]
            pivots_features = {
                f: pf.iloc[:, :max_days] for f, pf in pivots_features.items()
            }
        
        # Align: keep only rows/cols present in all dataframes
        all_dfs = [pivot_target] + list(pivots_features.values())
        common_rows = set(all_dfs[0].index)
        common_cols = set(all_dfs[0].columns)
        for pf in all_dfs[1:]:
            common_rows &= set(pf.index)
            common_cols &= set(pf.columns)
        
        pivot_target = pivot_target.loc[list(common_rows), list(common_cols)]
        pivots_features = {
            f: pf.loc[list(common_rows), list(common_cols)] 
            for f, pf in pivots_features.items()
        }
        
        # Drop any remaining rows with NaNs
        pivot_target = pivot_target.dropna(axis=0, how="any")
        for feat in FEATURE_COLUMNS:
            pivots_features[feat] = pivots_features[feat].loc[pivot_target.index]
        
        if max_components > 0 and pivot_target.shape[0] > max_components:
            idx = pivot_target.index[:max_components]
            pivot_target = pivot_target.loc[idx]
            pivots_features = {f: pf.loc[idx] for f, pf in pivots_features.items()}
        
        if pivot_target.shape[0] == 0 or pivot_target.shape[1] == 0:
            raise ValueError(f"Scale '{scale}' has no complete matrix after filtering.")
        
        y_obs = pivot_target.to_numpy(dtype=np.float32)
        dates = [str(d.date()) for d in pivot_target.columns]
        labels = [str(x) for x in pivot_target.index]
        
        # Stack features into (K, n_obs, n_features)
        K, n_obs = y_obs.shape
        X_raw = np.zeros((K, n_obs, len(FEATURE_COLUMNS)), dtype=np.float32)
        for i, feat in enumerate(FEATURE_COLUMNS):
            X_raw[:, :, i] = pivots_features[feat].to_numpy(dtype=np.float32)
    
    # Normalize covariates per feature (z-score across all observations)
    K, n_obs, n_features = X_raw.shape
    X_norm = np.zeros_like(X_raw)
    scaling_info = {}
    
    for i in range(n_features):
        feat_name = FEATURE_COLUMNS[i]
        X_feat = X_raw[:, :, i].flatten()  # Flatten to 1D for scaling
        
        # Manual z-score normalization
        mean_val = np.mean(X_feat)
        std_val = np.std(X_feat)
        if std_val < 1e-8:
            std_val = 1.0  # Avoid division by zero
        
        X_feat_scaled = (X_feat - mean_val) / std_val
        X_norm[:, :, i] = X_feat_scaled.reshape(K, n_obs)
        
        scaling_info[feat_name] = {
            "mean": float(mean_val),
            "std": float(std_val),
        }
    
    return y_obs, X_norm, labels, dates, scaling_info


def build_bernoulli_model_and_obs(
    y_obs: np.ndarray,
    X_cov: np.ndarray,
    mu_alpha: float,
    sigma_alpha: float,
    mu_beta: float,
    sigma_beta: float,
) -> Tuple[BernoulliLogitWithCovariates, np.ndarray]:
    """Instantiate Bernoulli-logit model and format observations for permABC.
    
    Parameters
    ----------
    y_obs : np.ndarray
        Binary rain observations of shape (K, n_obs).
    X_cov : np.ndarray
        Normalized weather feature covariates of shape (K, n_obs, n_features).
    mu_alpha, sigma_alpha : float
        Prior parameters for intercepts.
    mu_beta, sigma_beta : float
        Prior parameters for feature coefficients.
        
    Returns
    -------
    tuple
        (model, y_obs_formatted)
    """
    k_comp, n_obs, n_features = X_cov.shape
    
    model = BernoulliLogitWithCovariates(
        K=k_comp,
        n_obs=n_obs,
        n_features=n_features,
        mu_alpha=mu_alpha,
        sigma_alpha=sigma_alpha,
        mu_beta=mu_beta,
        sigma_beta=sigma_beta,
        X_cov=X_cov,
    )
    
    # Format observation: add batch dimension (1, K, n_obs)
    y_obs_formatted = y_obs[np.newaxis, :, :]
    
    return model, y_obs_formatted


def smc_result_to_lightweight(result: dict, method_name: str) -> dict:
    """Convert SMC outputs to a small and consistent storage format."""
    if result is None:
        return None

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
        # Under-matching should start as conservatively as possible on this
        # binary rain model to avoid early weight collapse.
        l0 = 1
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
    method_tag: str,
    metadata: dict,
    results_dir: Path,
) -> None:
    """Save one method result per (scale, seed)."""
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = metadata.get("seed", "unknown")
    
    path = results_dir / f"inference_weather_rain_{scale}_{method_tag}_seed_{seed}.pkl"
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
    csv_path = results_dir / f"comparison_weather_rain_seed_{seed}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"  Summary saved: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply permABC to real-world weather data for rain probability")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_particles", type=int, default=500)
    parser.add_argument("--final_iteration", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument("--country", type=str, default="Vietnam")
    parser.add_argument("--max_days", type=int, default=200)
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

    # Bernoulli-logit prior hyperparameters
    parser.add_argument("--mu_alpha", type=float, default=0.0)
    parser.add_argument("--sigma_alpha", type=float, default=2.0)
    parser.add_argument("--mu_beta", type=float, default=0.0)
    parser.add_argument("--sigma_beta", type=float, default=2.0)

    parser.add_argument(
        "--data_csv",
        type=str,
        default="my-reproduces/data/df_weather_clean.csv",
        help="Path to weather CSV file relative to repository root.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="my-reproduces/results/weather_rain_inference",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"=== Weather Rain Probability permABC | seed={args.seed} ===")

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

    # Load and validate data
    weather_df = load_weather_dataframe(
        csv_path=csv_path,
        country=args.country,
    )
    print(f"Loaded {len(weather_df)} valid weather records")

    # Pre-build scale data (observations + covariates) once
    scale_data = {}
    for scale in args.scales:
        y_obs, X_cov, labels, dates, scaling_info = build_scale_data(
            df=weather_df,
            scale=scale,
            max_days=args.max_days,
            max_components=args.max_components,
        )
        
        model, y_obs_formatted = build_bernoulli_model_and_obs(
            y_obs=y_obs,
            X_cov=X_cov,
            mu_alpha=args.mu_alpha,
            sigma_alpha=args.sigma_alpha,
            mu_beta=args.mu_beta,
            sigma_beta=args.sigma_beta,
        )
        
        scale_data[scale] = {
            "model": model,
            "y_obs": y_obs_formatted,
            "labels": labels,
            "dates": dates,
            "X_cov": X_cov,
            "scaling_info": scaling_info,
            "shape": y_obs.shape,
        }

    key = random.PRNGKey(args.seed)

    metadata = {
        "seed": args.seed,
        "country": args.country,
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "n_particles": args.n_particles,
        "final_iteration": args.final_iteration,
        "max_days": args.max_days,
        "max_components": args.max_components,
        "mu_alpha": args.mu_alpha,
        "sigma_alpha": args.sigma_alpha,
        "mu_beta": args.mu_beta,
        "sigma_beta": args.sigma_beta,
    }

    all_results: Dict[Tuple[str, str], dict] = {}

    for scale in args.scales:
        bundle = scale_data[scale]
        model = bundle["model"]
        y_obs = bundle["y_obs"]
        k_comp, n_obs = bundle["shape"]

        print("\n" + "=" * 72)
        print(f"Scale: {scale.upper()} | K={k_comp} | n_obs={n_obs} | n_features={len(FEATURE_COLUMNS)}")
        print("=" * 72)

        # Run permABC-SMC first to calibrate epsilon for OS/UM
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

                if result is None:
                    print(f"  Skipped: {method_name} returned no result")
                    continue

                print(
                    f"  Done in {elapsed:.1f}s | "
                    f"N_sim={result.get('total_n_sim', 0):,} | "
                    f"eps_final={result.get('final_epsilon', '?')}"
                )

                if method_name == "permABC-SMC":
                    epsilon_from_perm_smc = result.get("final_epsilon", None)
                    print(f"  Calibration epsilon for OS/UM: {epsilon_from_perm_smc}")

                # Attach scale-specific metadata
                result["meta"] = {
                    "labels": bundle["labels"],
                    "dates": bundle["dates"],
                    "scale": scale,
                    "features": FEATURE_COLUMNS,
                    "scaling_info": bundle["scaling_info"],
                }

                save_lightweight(
                    result=result,
                    scale=scale,
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

    print("\n=== Completed weather rain probability inference. ===")


if __name__ == "__main__":
    main()
