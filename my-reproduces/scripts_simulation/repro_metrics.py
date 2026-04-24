"""Tổng hợp các hàm hỗ trợ tính toán các chỉ số đánh giá độ ổn định và độ chính xác của permABC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from permabc.utils.functions import Theta


@dataclass
class RunResult:
    """Kết quả của một lần chạy (một method, một seed).

    Nhóm chỉ số chính:
    - Hiệu năng: runtime_sec, n_sim_total
    - Mức hội tụ ABC: eps_final
    - Độ chính xác ước lượng: mae_global, mae_local_mean
    """
    method: str  # Tên phương pháp suy luận: "perm" hoặc "smc".
    seed: int  # Seed ngẫu nhiên cho lần chạy hiện tại.
    runtime_sec: float  # Tổng thời gian chạy (giây).
    eps_final: float  # Epsilon cuối cùng của SMC; nhỏ hơn thường nghĩa là khớp dữ liệu tốt hơn.
    n_sim_total: float  # Tổng số lần mô phỏng được dùng trong toàn bộ vòng lặp SMC.
    mae_global: float  # Sai số tuyệt đối trung bình của tham số global: mean(|est_global - true_global|).
    mae_local_mean: float  # Sai số tuyệt đối của trung bình tham số local: |E[mean(local)] - true_mean(local)|.
    true_global: np.ndarray  # Giá trị ground-truth của vector tham số global.
    est_global: np.ndarray  # Giá trị ước lượng hậu nghiệm (trung bình có trọng số) của global.
    true_local_mean: float  # Trung bình thật của các tham số local giữa các compartment.
    est_local_mean: float  # Trung bình local hậu nghiệm (đã lấy trung bình có trọng số theo particle).


def _weighted_mean(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sum(values * w[:, None], axis=0)


def extract_metrics(true_theta: Theta, out: Dict, method: str, seed: int, runtime_sec: float) -> RunResult:
    thetas_last = out["Thetas"][-1]
    weights_last = np.asarray(out["Weights"][-1], dtype=np.float64)
    weights_last = weights_last / np.sum(weights_last)

    est_global = _weighted_mean(np.asarray(thetas_last.glob), weights_last)
    true_global = np.asarray(true_theta.glob[0])
    mae_global = float(np.mean(np.abs(est_global - true_global)))

    est_local_mean = float(np.sum(np.mean(np.asarray(thetas_last.loc)[:, :, 0], axis=1) * weights_last))
    true_local_mean = float(np.mean(np.asarray(true_theta.loc[0, :, 0])))
    mae_local_mean = float(abs(est_local_mean - true_local_mean))

    return RunResult(
        method=method,
        seed=seed,
        runtime_sec=runtime_sec,
        eps_final=float(out["Eps_values"][-1]),
        n_sim_total=float(np.sum(out["N_sim"])),
        mae_global=mae_global,
        mae_local_mean=mae_local_mean,
        true_global=true_global,
        est_global=est_global,
        true_local_mean=true_local_mean,
        est_local_mean=est_local_mean,
    )


def summarize(results: List[RunResult]) -> Dict:
    summary: Dict[str, Dict[str, float]] = {}
    methods = sorted({r.method for r in results})
    for method in methods:
        rows = [r for r in results if r.method == method]
        summary[method] = {
            "runs": len(rows),
            "mae_global_mean": float(np.mean([r.mae_global for r in rows])),
            "mae_global_std": float(np.std([r.mae_global for r in rows])),
            "mae_local_mean_mean": float(np.mean([r.mae_local_mean for r in rows])),
            "mae_local_mean_std": float(np.std([r.mae_local_mean for r in rows])),
            "runtime_sec_mean": float(np.mean([r.runtime_sec for r in rows])),
            "runtime_sec_std": float(np.std([r.runtime_sec for r in rows])),
            "n_sim_total_mean": float(np.mean([r.n_sim_total for r in rows])),
            "eps_final_mean": float(np.mean([r.eps_final for r in rows])),
        }
    return summary
