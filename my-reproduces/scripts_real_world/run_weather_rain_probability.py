#!/usr/bin/env python3
"""
Chạy permABC trên dữ liệu thời tiết thực để suy luận xác suất mưa.

Quy trình chính của script:
1) Nạp và làm sạch dữ liệu thực.
2) Tạo quan sát theo nhiều mức (toàn quốc/vùng/tỉnh).
3) Chạy permABC-SMC và các baseline cho từng mức.
4) Lưu kết quả dạng nhẹ (lightweight) và bảng so sánh.

Phần vẽ hình được tách riêng trong plot_weather_rain_probability.py,
giống workflow SIR real-world (inference và plotting tách riêng).

Lựa chọn mô hình
----------------
Dùng BernoulliLogitWithCovariates cho bài toán nhị phân mưa/không mưa (0/1)
với 5 biến thời tiết đã chuẩn hóa làm biến giải thích:
- day.maxtemp_c
- day.maxwind_kph
- day.totalprecip_mm
- day.avghumidity
- day.uv

Mỗi thành phần (tỉnh/vùng/quốc gia) được mô hình hóa bằng Bernoulli-logit,
trong đó intercept cục bộ là alpha_k và hệ số toàn cục là beta.

Lệnh chạy đầy đủ để so sánh phương pháp:
python3 my-reproduces/scripts_real_world/run_weather_rain_probability.py \
    --seed 42 --scales national regional provincial \
    --n_particles 1000 --final_iteration 100 --num_gibbs_blocks 3 \
    --gibbs_T 1000 --gibbs_M_loc 50 --gibbs_M_glob 100 --max_days 0
"""



from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time as _time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
from permabc.utils.functions import Theta

from abc_gibbs_weather import run_gibbs_sampler_weather


# Các cột đặc trưng thời tiết dùng làm biến giải thích
FEATURE_COLUMNS = [
    "day.maxtemp_c",
    "day.maxwind_kph",
    "day.totalprecip_mm",
    "day.avghumidity",
    "day.uv",
]

# Biến đích cho bài toán mưa
TARGET_COLUMN = "day.daily_will_it_rain"


_METHOD_REGISTRY_TEMPLATE: Dict[str, Dict[str, str]] = {
    "permABC-SMC": {"tag": "perm_smc", "type": "perm_smc"},
    "ABC-SMC": {"tag": "abc_smc", "type": "abc_smc"},
    "ABC-SMC (Gibbs {H}b)": {"tag": "abc_smc_g{H}", "type": "abc_smc_gibbs"},
    "permABC-SMC (Gibbs {H}b)": {"tag": "perm_smc_g{H}", "type": "perm_smc_gibbs"},
    "ABC-Gibbs": {"tag": "abc_gibbs_true", "type": "abc_gibbs_true"},
    "permABC-SMC-OS": {"tag": "perm_smc_os", "type": "perm_smc_os"},
    "permABC-SMC-UM": {"tag": "perm_smc_um", "type": "perm_smc_um"},
}


def expand_registry(num_gibbs_blocks: int) -> Dict[str, Dict[str, str]]:
    """Mở rộng tên method bằng cách thay {H} theo --num_gibbs_blocks."""
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
    """Tạo slug gọn, an toàn ASCII để đặt tên file."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def load_weather_dataframe(csv_path: Path, country: str) -> pd.DataFrame:
    """Nạp dữ liệu panel thời tiết và kiểm tra các cột bắt buộc."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Weather dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    required_cols = {"province", "region", "country", "date"} | set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Chỉ giữ các cột cần thiết
    cols_to_keep = ["province", "region", "country", "date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[cols_to_keep].copy()
    
    # Lọc theo quốc gia
    df = df[df["country"] == country].copy()

    # Chuẩn hóa kiểu dữ liệu cho cột ngày và cột số
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Loại bỏ dòng có NaN để tránh lỗi khi pivot
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
    """Tạo ma trận quan sát và ma trận biến giải thích theo từng mức.
    
    Returns
    -------
    tuple
        (y_obs, X_cov, labels, dates, scaling_info)
        - y_obs: (K, n_obs) quan sát mưa nhị phân
        - X_cov: (K, n_obs, n_features) đặc trưng thời tiết đã chuẩn hóa
        - labels: danh sách tên thành phần (K thành phần)
        - dates: danh sách ngày
        - scaling_info: tham số chuẩn hóa để tái lập
    """
    if scale == "national":
        # Gom về mức toàn quốc: lấy trung bình qua các vùng
        grouped = df.groupby("date", as_index=False).agg(
            {target: "mean" for target in [TARGET_COLUMN] + FEATURE_COLUMNS}
        ).sort_values("date")
        
        if max_days > 0 and len(grouped) > max_days:
            grouped = grouped.head(max_days)
        
        # Chuyển về ma trận: mỗi biến có dạng (1, n_obs)
        dates = grouped["date"].dt.strftime("%Y-%m-%d").tolist()
        y_obs = grouped[TARGET_COLUMN].values.astype(np.float32)[np.newaxis, :]
        X_raw = grouped[FEATURE_COLUMNS].values.astype(np.float32)[np.newaxis, :, :]
        labels = ["national"]
        
    else:
        group_col = "region" if scale == "regional" else "province"
        
        # Gom theo (nhóm, ngày), rồi pivot sang ma trận rộng
        agg_dict = {col: "mean" for col in [TARGET_COLUMN] + FEATURE_COLUMNS}
        grouped = df.groupby([group_col, "date"], as_index=False).agg(agg_dict)
        
        # Pivot để tạo ma trận (nhóm, ngày) cho biến đích
        pivot_target = grouped.pivot(index=group_col, columns="date", values=TARGET_COLUMN)
        pivot_target = pivot_target.sort_index(axis=0).sort_index(axis=1)
        
        # Pivot cho từng biến đặc trưng
        pivots_features = {}
        for feat in FEATURE_COLUMNS:
            pivot_feat = grouped.pivot(index=group_col, columns="date", values=feat)
            pivot_feat = pivot_feat.sort_index(axis=0).sort_index(axis=1)
            pivots_features[feat] = pivot_feat
        
        # Căn chỉnh: chỉ giữ hàng/cột có mặt ở tất cả DataFrame
        all_dfs = [pivot_target] + list(pivots_features.values())
        common_rows = set(all_dfs[0].index)
        common_cols = set(all_dfs[0].columns)
        for pf in all_dfs[1:]:
            common_rows &= set(pf.index)
            common_cols &= set(pf.columns)

        # Sắp xếp để đảm bảo kết quả xác định (deterministic)
        common_rows = sorted(common_rows)
        common_cols = sorted(common_cols)

        pivot_target = pivot_target.loc[common_rows, common_cols]
        pivots_features = {
            f: pf.loc[common_rows, common_cols]
            for f, pf in pivots_features.items()
        }

        # Chỉ giữ những ngày đầy đủ cho TOÀN BỘ thành phần và TOÀN BỘ biến.
        # Cách này giúp giữ đủ tập tỉnh/vùng (K) khi có thể.
        complete_dates_mask = pivot_target.notna().all(axis=0)
        for pf in pivots_features.values():
            complete_dates_mask &= pf.notna().all(axis=0)

        complete_dates = pivot_target.columns[complete_dates_mask]
        if max_days > 0 and len(complete_dates) > max_days:
            complete_dates = complete_dates[:max_days]

        pivot_target = pivot_target.loc[:, complete_dates]
        for feat in FEATURE_COLUMNS:
            pivots_features[feat] = pivots_features[feat].loc[pivot_target.index, complete_dates]
        
        if max_components > 0 and pivot_target.shape[0] > max_components:
            idx = pivot_target.index[:max_components]
            pivot_target = pivot_target.loc[idx]
            pivots_features = {f: pf.loc[idx] for f, pf in pivots_features.items()}
        
        if pivot_target.shape[0] == 0 or pivot_target.shape[1] == 0:
            raise ValueError(f"Scale '{scale}' has no complete matrix after filtering.")
        
        y_obs = pivot_target.to_numpy(dtype=np.float32)
        dates = [str(d.date()) for d in pivot_target.columns]
        labels = [str(x) for x in pivot_target.index]
        
        # Ghép đặc trưng thành tensor (K, n_obs, n_features)
        K, n_obs = y_obs.shape
        X_raw = np.zeros((K, n_obs, len(FEATURE_COLUMNS)), dtype=np.float32)
        for i, feat in enumerate(FEATURE_COLUMNS):
            X_raw[:, :, i] = pivots_features[feat].to_numpy(dtype=np.float32)
    
    # Chuẩn hóa từng đặc trưng (z-score trên toàn bộ quan sát)
    K, n_obs, n_features = X_raw.shape
    X_norm = np.zeros_like(X_raw)
    scaling_info = {}
    
    for i in range(n_features):
        feat_name = FEATURE_COLUMNS[i]
        X_feat = X_raw[:, :, i].flatten()  # Trải phẳng về 1D để chuẩn hóa
        
        # Chuẩn hóa z-score thủ công
        mean_val = np.mean(X_feat)
        std_val = np.std(X_feat)
        if std_val < 1e-8:
            std_val = 1.0  # Tránh chia cho 0
        
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
    """Khởi tạo mô hình Bernoulli-logit và định dạng quan sát cho permABC.
    
    Parameters
    ----------
    y_obs : np.ndarray
        Quan sát mưa nhị phân dạng (K, n_obs).
    X_cov : np.ndarray
        Biến giải thích thời tiết đã chuẩn hóa dạng (K, n_obs, n_features).
    mu_alpha, sigma_alpha : float
        Siêu tham số prior cho intercept.
    mu_beta, sigma_beta : float
        Siêu tham số prior cho hệ số đặc trưng.
        
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
    
    # Định dạng y_obs: thêm chiều batch thành (1, K, n_obs)
    y_obs_formatted = y_obs[np.newaxis, :, :]
    
    return model, y_obs_formatted


def smc_result_to_lightweight(result: dict, method_name: str) -> dict:
    """Chuyển output SMC sang format gọn và nhất quán để lưu."""
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


def gibbs_result_to_lightweight(
    alpha_chain: np.ndarray,
    beta_chain: np.ndarray,
    eps_loc: np.ndarray,
    eps_glob: np.ndarray,
    times: np.ndarray,
    n_sim_per_iter: int,
    method_name: str,
    n_particles_out: int,
) -> dict:
    """Chuyển output chain ABC-Gibbs về cùng format nhẹ như SMC."""
    total_iters = max(len(beta_chain) - 1, 0)
    burn_in = max(0, total_iters - n_particles_out)

    alpha_samples = alpha_chain[burn_in + 1 :]
    beta_samples = beta_chain[burn_in + 1 :]
    thetas_final = Theta(loc=alpha_samples, glob=beta_samples)

    return {
        "Thetas_final": thetas_final,
        "n_iterations": total_iters,
        "total_n_sim": int(total_iters * n_sim_per_iter),
        "time_final": float(np.sum(times)),
        "final_epsilon": float(eps_glob[-1]) if len(eps_glob) else np.inf,
        "Eps_values": eps_glob.tolist(),
        "N_sim": [int(n_sim_per_iter)] * total_iters,
        "eps_loc": eps_loc.tolist(),
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
    """Chạy một phương pháp và trả kết quả nhẹ đã chuẩn hóa."""
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

    if mtype == "abc_gibbs_true":
        k_comp = model.K
        y_obs_2d = np.asarray(y_obs).squeeze()
        if y_obs_2d.ndim == 1:
            y_obs_2d = y_obs_2d[None, :]

        total_iters = args.gibbs_T if args.gibbs_T > 0 else args.n_particles
        m_loc = args.gibbs_M_loc
        m_glob = args.gibbs_M_glob

        print(
            f"  ABC-Gibbs: T={total_iters}, M_loc={m_loc}, M_glob={m_glob}, "
            f"K={k_comp}, expected N_sim={total_iters * k_comp * (m_loc + m_glob):,}"
        )
        alpha_chain, beta_chain, eps_loc, eps_glob, times, n_sim_per_iter = run_gibbs_sampler_weather(
            key=key,
            model=model,
            y_obs_2d=y_obs_2d,
            T=total_iters,
            M_loc=m_loc,
            M_glob=m_glob,
        )
        return gibbs_result_to_lightweight(
            alpha_chain=alpha_chain,
            beta_chain=beta_chain,
            eps_loc=eps_loc,
            eps_glob=eps_glob,
            times=times,
            n_sim_per_iter=n_sim_per_iter,
            method_name=method_name,
            n_particles_out=args.n_particles,
        )

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
        # Under-matching nên bắt đầu thật bảo thủ trên dữ liệu nhị phân
        # để giảm nguy cơ sụp trọng số từ sớm.
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
    """Lưu kết quả cho từng phương pháp theo (scale, seed)."""
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = metadata.get("seed", "unknown")
    
    path = results_dir / f"inference_weather_rain_{scale}_{method_tag}_seed_{seed}.pkl"
    with path.open("wb") as f:
        pickle.dump(result, f)

    print(f"  Saved: {path}")


def save_summary(all_results: dict, metadata: dict, results_dir: Path) -> None:
    """Lưu CSV tổng hợp đa phương pháp để so sánh benchmark nhanh."""
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
    parser.add_argument("--gibbs_T", type=int, default=0)
    parser.add_argument("--gibbs_M_loc", type=int, default=50)
    parser.add_argument("--gibbs_M_glob", type=int, default=100)

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

    # Nạp dữ liệu và kiểm tra hợp lệ
    weather_df = load_weather_dataframe(
        csv_path=csv_path,
        country=args.country,
    )
    print(f"Loaded {len(weather_df)} valid weather records")

    # Tiền xử lý dữ liệu theo từng mức (y_obs + X_cov) một lần
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

        # Chạy permABC-SMC trước để lấy epsilon khởi tạo cho OS/UM
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

                # Gắn metadata theo từng mức để tái lập kết quả
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
