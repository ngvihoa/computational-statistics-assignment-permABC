#!/usr/bin/env python3
"""Chạy benchmark so sánh hiệu năng các phương pháp ABC trên Gaussian toy model.

Script này tham khảo luồng của experiments/scripts/run_performance_comparison.py,
nhưng được rút gọn để dùng nhanh trong my-reproduces/scripts_simulation.

Mục tiêu:
1) So sánh chất lượng hội tụ (epsilon) theo chi phí:
   - Số mô phỏng chuẩn hóa
   - Thời gian chuẩn hóa
2) So sánh theo nhóm phương pháp:
   - vanilla: ABC-Vanilla, permABC-Vanilla
   - smc: ABC-SMC, permABC-SMC
   - pmc: ABC-PMC
   - osum: permABC-SMC-OS, permABC-SMC-UM

Đầu ra:
- CSV long-format: mỗi dòng là một điểm trên đường hiệu năng
- JSON summary: thống kê tổng hợp theo method

Ví dụ chạy:
    # Full benchmark
    # Mặc định: n_points=1_000_000, n_particles=1000,
    # n_sim_budget=n_points*K, n_epsilon=10000 cho OS/UM calibration.
    python3 my-reproduces/scripts_simulation/run_performance_comparison_simulation.py \
        --methods all \
        --K 20 \
        --K-outliers 4 \
        --seed 42 \
        --prefix perf_compare_like_experiments_K20

    # Smoke test nhanh để kiểm tra pipeline CSV/JSON/figure.
    python3 my-reproduces/scripts_simulation/run_performance_comparison_simulation.py \
        --methods all \
        --K 8 \
        --K-outliers 2 \
        --seed 42 \
        --n-points 10000 \
        --n-particles 200 \
        --n-sim-budget 100000 \
        --n-epsilon 1000 \
        --m0-values 12 16 24 \
        --l0-values 2 4 6 8 \
        --prefix smoke_perf_compare_all_fast

    # không chạy OS/UM.
    python3 my-reproduces/scripts_simulation/run_performance_comparison_simulation.py \
        --methods vanilla smc pmc \
        --no-osum \
        --K 20 \
        --K-outliers 4 \
        --seed 42 \
        --prefix perf_compare_no_osum_K20

    # Debug riêng OS/UM với calibration nhẹ hơn.
    python3 my-reproduces/scripts_simulation/run_performance_comparison_simulation.py \
        --methods osum \
        --K 20 \
        --K-outliers 4 \
        --seed 42 \
        --n-particles 500 \
        --n-epsilon 2000 \
        --m0-values 30 40 100 \
        --l0-values 2 5 10 20 \
        --prefix debug_osum_K20
"""


from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
from jax import random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from permabc.algorithms.smc import abc_smc, perm_abc_smc
from permabc.algorithms.vanilla import abc_vanilla, perm_abc_vanilla
from permabc.core.distances import optimal_index_distance
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.utils.functions import Theta

try:
    from permabc.algorithms.pmc import abc_pmc
except Exception:  # pragma: no cover
    abc_pmc = None

try:
    from permabc.algorithms.over_sampling import perm_abc_smc_os
except Exception:  # pragma: no cover
    perm_abc_smc_os = None

try:
    from permabc.algorithms.under_matching import perm_abc_smc_um
except Exception:  # pragma: no cover
    perm_abc_smc_um = None


METHOD_ORDER = [
    "ABC-Vanilla",
    "permABC-Vanilla",
    "ABC-SMC",
    "ABC-PMC",
    "permABC-SMC",
    "permABC-SMC-OS",
    "permABC-SMC-UM",
]


@dataclass
class PerfRow:
    method: str
    epsilon: float
    n_sim: float
    time: float
    n_sim_raw: float
    time_raw: float
    abs_err_sigma2: float
    abs_err_mu_mean: float
    seed: int
    K: int
    K_outliers: int


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[:, None], axis=0)


def _unique_particle_fraction(theta: Theta, n_particles: int) -> float:
    """Return the fraction of unique particles, independent of algorithm verbosity."""
    if n_particles <= 0:
        return 0.0

    reshaped = np.asarray(theta.reshape_2d())
    if reshaped.size == 0:
        return 0.0

    return float(len(np.unique(reshaped, axis=0)) / float(n_particles))


def _unique_count(theta: Theta, reported_fraction: float, n_particles: int) -> float:
    """Use reported uniqueness when available, otherwise compute it from particles."""
    frac = float(reported_fraction) if np.isfinite(reported_fraction) else 0.0
    if frac <= 0.0:
        frac = _unique_particle_fraction(theta, n_particles)
    return max(frac * max(float(n_particles), 1.0), 1.0)


def _estimate_from_theta(theta: Theta, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    loc = np.asarray(theta.loc)[:, :, 0]
    glob = np.asarray(theta.glob).reshape(np.asarray(theta.glob).shape[0], -1)

    if weights is None:
        weights = np.ones(len(loc), dtype=np.float64) / max(len(loc), 1)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        s = np.sum(weights)
        weights = weights / s if s > 0 else np.ones(len(loc), dtype=np.float64) / max(len(loc), 1)

    mu_mean_particles = np.mean(loc, axis=1)
    mu_mean_est = float(np.sum(mu_mean_particles * weights))
    sigma2_est = float(_weighted_mean(glob, weights).ravel()[0])
    return {"mu_mean_est": mu_mean_est, "sigma2_est": sigma2_est}


def _build_error_fields(true_theta: Theta, est: Dict[str, float]) -> Dict[str, float]:
    sigma2_true = float(np.asarray(true_theta.glob[0, 0]))
    mu_mean_true = float(np.mean(np.asarray(true_theta.loc[0, :, 0])))
    return {
        "abs_err_sigma2": abs(float(est["sigma2_est"]) - sigma2_true),
        "abs_err_mu_mean": abs(float(est["mu_mean_est"]) - mu_mean_true),
    }


def _setup_experiment(K: int, K_outliers: int, seed: int, n_obs: int, mu_0: float, sigma_0: float, alpha: float, beta: float):
    key = random.PRNGKey(seed)
    key, key_true = random.split(key)

    model = GaussianWithNoSummaryStats(K=K, n_obs=n_obs, mu_0=mu_0, sigma_0=sigma_0, alpha=alpha, beta=beta)
    true_theta = model.prior_generator(key_true, n_particles=1)

    # Cố định sigma2 thật để so sánh nhất quán với benchmark paper.
    true_theta = Theta(loc=true_theta.loc, glob=np.array([[1.0]], dtype=np.float64))

    for i in range(min(K_outliers, K)):
        key, key_sign, key_val = random.split(key, 3)
        sign = int(np.asarray(random.choice(key_sign, a=np.array([-1, 1]), shape=(1,)))[0])
        if sign == 1:
            outlier_val = float(np.asarray(random.uniform(key_val, shape=(1,), minval=-3 * sigma_0, maxval=-2 * sigma_0))[0])
        else:
            outlier_val = float(np.asarray(random.uniform(key_val, shape=(1,), minval=2 * sigma_0, maxval=3 * sigma_0))[0])
        loc = np.array(true_theta.loc, copy=True)
        loc[0, i, 0] = outlier_val
        true_theta = Theta(loc=loc, glob=true_theta.glob)

    key, key_obs = random.split(key)
    y_obs = model.data_generator(key_obs, true_theta)
    return key, model, y_obs, true_theta


def _extract_smc_rows(
    out: Optional[Dict],
    method: str,
    true_theta: Theta,
    K: int,
    seed: int,
    K_outliers: int,
    n_particles: int,
    n_sample: int = 1000,
) -> List[PerfRow]:
    if out is None:
        return []

    needed = ["N_sim", "Eps_values", "Time", "Thetas", "Weights", "unique_part"]
    if any(key not in out for key in needed):
        return []

    n_sim_arr = np.asarray(out["N_sim"], dtype=np.float64)
    eps_arr = np.asarray(out["Eps_values"], dtype=np.float64)
    time_arr = np.asarray(out["Time"], dtype=np.float64)
    uniq_arr = np.asarray(out["unique_part"], dtype=np.float64)

    if len(n_sim_arr) == 0 or len(eps_arr) == 0 or len(time_arr) == 0 or len(uniq_arr) == 0:
        return []

    # Một số cấu hình chỉ trả về một snapshot cuối; vẫn giữ lại để plot không bị rỗng.
    if len(n_sim_arr) < 2 or len(out["Thetas"]) < 2 or len(out["Weights"]) < 2:
        theta_last = out["Thetas"][-1]
        weights_last = np.asarray(out["Weights"][-1], dtype=np.float64)
        est = _estimate_from_theta(theta_last, weights_last)
        err = _build_error_fields(true_theta, est)
        unique_count = _unique_count(theta_last, float(uniq_arr[-1]), n_particles)
        n_sim_norm = (float(np.sum(n_sim_arr)) / (max(K, 1) * unique_count)) * n_sample
        time_norm = (float(np.sum(time_arr)) / unique_count) * n_sample
        return [
            PerfRow(
                method=method,
                epsilon=float(eps_arr[-1]),
                n_sim=n_sim_norm,
                time=time_norm,
                n_sim_raw=float(np.sum(n_sim_arr)),
                time_raw=float(np.sum(time_arr)),
                abs_err_sigma2=float(err["abs_err_sigma2"]),
                abs_err_mu_mean=float(err["abs_err_mu_mean"]),
                seed=seed,
                K=K,
                K_outliers=K_outliers,
            )
        ]

    n_sim_cum = np.cumsum(n_sim_arr[1:])
    time_cum = np.cumsum(time_arr[1:])
    eps_vals = eps_arr[1:]
    uniq_vals = uniq_arr[1:]

    max_steps = min(len(n_sim_cum), len(time_cum), len(eps_vals), len(uniq_vals), max(0, len(out["Thetas"]) - 1))
    rows: List[PerfRow] = []

    for i in range(max_steps):
        theta_i = out["Thetas"][i + 1]
        weights_i = np.asarray(out["Weights"][i + 1], dtype=np.float64)
        est = _estimate_from_theta(theta_i, weights_i)
        err = _build_error_fields(true_theta, est)

        unique_count = _unique_count(theta_i, float(uniq_vals[i]), n_particles)
        n_sim_norm = (float(n_sim_cum[i]) / (max(K, 1) * unique_count)) * n_sample
        time_norm = (float(time_cum[i]) / unique_count) * n_sample

        rows.append(
            PerfRow(
                method=method,
                epsilon=float(eps_vals[i]),
                n_sim=n_sim_norm,
                time=time_norm,
                n_sim_raw=float(n_sim_cum[i]),
                time_raw=float(time_cum[i]),
                abs_err_sigma2=float(err["abs_err_sigma2"]),
                abs_err_mu_mean=float(err["abs_err_mu_mean"]),
                seed=seed,
                K=K,
                K_outliers=K_outliers,
            )
        )

    return rows


def _run_vanilla_rows(key, model, y_obs, true_theta, n_points: int, seed: int, K: int, K_outliers: int) -> List[PerfRow]:
    rows: List[PerfRow] = []
    alphas = np.logspace(0, -3, 10)

    # ABC-Vanilla
    t0 = time.time()
    _, thetas, dists, n_sim = abc_vanilla(
        key=key,
        model=model,
        n_points=n_points,
        epsilon=np.inf,
        y_obs=y_obs,
    )
    dists = np.asarray(dists, dtype=np.float64).reshape(-1)
    total_time = time.time() - t0
    per_sim = total_time / max(float(n_sim), 1.0)

    for alpha in alphas:
        eps = float(np.quantile(dists, alpha))
        accepted = np.asarray(dists) <= eps
        theta_acc = thetas[accepted]
        est = _estimate_from_theta(theta_acc, None)
        err = _build_error_fields(true_theta, est)

        n_sim_norm = (1.0 / alpha) * 1000.0
        rows.append(
            PerfRow(
                method="ABC-Vanilla",
                epsilon=eps,
                n_sim=n_sim_norm,
                time=n_sim_norm * per_sim,
                n_sim_raw=n_sim_norm * K,
                time_raw=n_sim_norm * per_sim,
                abs_err_sigma2=float(err["abs_err_sigma2"]),
                abs_err_mu_mean=float(err["abs_err_mu_mean"]),
                seed=seed,
                K=K,
                K_outliers=K_outliers,
            )
        )

    # permABC-Vanilla
    t0 = time.time()
    _, thetas_p, dists_p, _, _, n_sim_p = perm_abc_vanilla(
        key=random.PRNGKey(seed + 1111),
        model=model,
        n_points=n_points,
        epsilon=np.inf,
        y_obs=y_obs,
        try_swaps=True,
        try_lsa=True,
    )
    dists_p = np.asarray(dists_p, dtype=np.float64).reshape(-1)
    total_time_p = time.time() - t0
    per_sim_p = total_time_p / max(float(n_sim_p), 1.0)

    for alpha in alphas:
        eps = float(np.quantile(dists_p, alpha))
        accepted = np.asarray(dists_p) <= eps
        theta_acc = thetas_p[accepted]
        est = _estimate_from_theta(theta_acc, None)
        err = _build_error_fields(true_theta, est)

        n_sim_norm = (1.0 / alpha) * 1000.0
        rows.append(
            PerfRow(
                method="permABC-Vanilla",
                epsilon=eps,
                n_sim=n_sim_norm,
                time=n_sim_norm * per_sim_p,
                n_sim_raw=n_sim_norm * K,
                time_raw=n_sim_norm * per_sim_p,
                abs_err_sigma2=float(err["abs_err_sigma2"]),
                abs_err_mu_mean=float(err["abs_err_mu_mean"]),
                seed=seed,
                K=K,
                K_outliers=K_outliers,
            )
        )

    return rows


def _run_smc_pmc_rows(key, model, y_obs, true_theta, n_particles: int, n_sim_budget: float, seed: int, K: int, K_outliers: int, methods: Sequence[str]) -> List[PerfRow]:
    rows: List[PerfRow] = []
    run_all = "all" in methods
    run_smc = run_all or ("smc" in methods)
    run_pmc = run_all or ("pmc" in methods)

    if run_smc:
        out_smc = abc_smc(
            key=random.PRNGKey(seed + 2000),
            model=model,
            n_particles=n_particles,
            epsilon_target=0,
            y_obs=y_obs,
            kernel=KernelTruncatedRW,
            verbose=0,
            Final_iteration=0,
            update_weights_distance=False,
            stopping_accept_rate=0.0,
            N_sim_max=n_sim_budget,
        )
        rows.extend(_extract_smc_rows(out_smc, "ABC-SMC", true_theta, K, seed, K_outliers, n_particles))

        out_perm = perm_abc_smc(
            key=random.PRNGKey(seed + 3000),
            model=model,
            n_particles=n_particles,
            epsilon_target=0,
            y_obs=y_obs,
            kernel=KernelTruncatedRW,
            verbose=0,
            Final_iteration=0,
            update_weights_distance=False,
            stopping_accept_rate=0.0,
            N_sim_max=n_sim_budget,
        )
        rows.extend(_extract_smc_rows(out_perm, "permABC-SMC", true_theta, K, seed, K_outliers, n_particles))

    if run_pmc and abc_pmc is not None:
        out_pmc = abc_pmc(
            key=random.PRNGKey(seed + 4000),
            model=model,
            n_particles=n_particles,
            epsilon_target=0,
            y_obs=y_obs,
            alpha=0.95,
            verbose=0,
            update_weights_distance=False,
            stopping_accept_rate=0.0,
            N_sim_max=n_sim_budget,
        )
        rows.extend(_extract_smc_rows(out_pmc, "ABC-PMC", true_theta, K, seed, K_outliers, n_particles))

    return rows


def _calibrate_perm_epsilon(key, model, y_obs, n_epsilon: int, alpha_epsilon: float, M: int = 0, L: int = 0) -> float:
    key_theta, key_data = random.split(key)
    if M > 0:
        thetas = model.prior_generator(key_theta, n_epsilon, M)
    else:
        thetas = model.prior_generator(key_theta, n_epsilon)
    zs = model.data_generator(key_data, thetas)
    dists_perm, _, _, _ = optimal_index_distance(
        model=model,
        zs=zs,
        y_obs=y_obs,
        epsilon=0,
        verbose=0,
        M=M,
        L=L,
    )
    return float(np.quantile(np.asarray(dists_perm, dtype=np.float64), alpha_epsilon))


def _run_osum_rows(
    key,
    model,
    y_obs,
    true_theta,
    n_particles: int,
    seed: int,
    K: int,
    K_outliers: int,
    m0_values: Sequence[int],
    l0_values: Sequence[int],
    n_epsilon: int,
    alpha_epsilon: float,
) -> List[PerfRow]:
    rows: List[PerfRow] = []

    if perm_abc_smc_os is None or perm_abc_smc_um is None:
        return rows

    for m0 in m0_values:
        if m0 <= K:
            continue
        key_eps = random.PRNGKey(seed + 4500 + int(m0))
        epsilon = _calibrate_perm_epsilon(key_eps, model, y_obs, n_epsilon, alpha_epsilon, M=int(m0))
        out_os = perm_abc_smc_os(
            key=random.PRNGKey(seed + 5000 + int(m0)),
            model=model,
            n_particles=n_particles,
            y_obs=y_obs,
            kernel=KernelTruncatedRW,
            M_0=int(m0),
            epsilon=epsilon,
            alpha_epsilon=alpha_epsilon,
            alpha_M=0.9,
            Final_iteration=0,
            alpha_resample=0.5,
            update_weights_distance=False,
            verbose=0,
            duplicate=True,
        )
        # Chỉ lấy điểm cuối để so sánh OS giữa các M0 gọn hơn.
        os_rows = _extract_smc_rows(out_os, "permABC-SMC-OS", true_theta, K, seed, K_outliers, n_particles)
        if os_rows:
            rows.append(os_rows[-1])

    for l0 in l0_values:
        l0 = int(l0)
        if l0 > K:
            continue
        key_eps = random.PRNGKey(seed + 6500 + l0)
        epsilon = _calibrate_perm_epsilon(key_eps, model, y_obs, n_epsilon, alpha_epsilon, L=l0)
        out_um = perm_abc_smc_um(
            key=random.PRNGKey(seed + 7000 + l0),
            model=model,
            n_particles=n_particles,
            y_obs=y_obs,
            kernel=KernelTruncatedRW,
            L_0=l0,
            epsilon=epsilon,
            alpha_epsilon=alpha_epsilon,
            alpha_L=0.9,
            Final_iteration=0,
            alpha_resample=0.5,
            update_weights_distance=False,
            verbose=0,
            stopping_acc_rate=0.0,
        )
        um_rows = _extract_smc_rows(out_um, "permABC-SMC-UM", true_theta, K, seed, K_outliers, n_particles)
        if um_rows:
            rows.append(um_rows[-1])

    return rows


def _summarize(rows: List[PerfRow]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[PerfRow]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for method, subset in grouped.items():
        eps = np.asarray([r.epsilon for r in subset], dtype=np.float64)
        n_sim = np.asarray([r.n_sim for r in subset], dtype=np.float64)
        tm = np.asarray([r.time for r in subset], dtype=np.float64)
        err_s2 = np.asarray([r.abs_err_sigma2 for r in subset], dtype=np.float64)
        err_mu = np.asarray([r.abs_err_mu_mean for r in subset], dtype=np.float64)
        summary[method] = {
            "rows": len(subset),
            "epsilon_mean": float(np.nanmean(eps)),
            "n_sim_mean": float(np.nanmean(n_sim)),
            "time_mean": float(np.nanmean(tm)),
            "abs_err_sigma2_mean": float(np.nanmean(err_s2)),
            "abs_err_mu_mean_mean": float(np.nanmean(err_mu)),
        }
    return summary


def _save(rows: List[PerfRow], summary: Dict[str, Dict[str, float]], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "epsilon",
                "n_sim",
                "time",
                "n_sim_raw",
                "time_raw",
                "abs_err_sigma2",
                "abs_err_mu_mean",
                "seed",
                "K",
                "K_outliers",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    json_path = output_dir / f"{prefix}_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "method_order": METHOD_ORDER}, file, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def _rows_by_method(rows: List[PerfRow]) -> Dict[str, List[PerfRow]]:
    grouped: Dict[str, List[PerfRow]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row)
    for method in grouped:
        grouped[method] = sorted(grouped[method], key=lambda r: r.n_sim)
    return grouped


def _style_for_method(method: str, index: int):
    cmap = plt.get_cmap("tab10")
    color = cmap(index % 10)
    if method == "permABC-SMC":
        return dict(color="#111111", linewidth=3.0, alpha=1.0, marker="o")
    if method.startswith("permABC-SMC"):
        return dict(color=color, linewidth=2.2, alpha=0.95, marker="o")
    if method == "ABC-SMC":
        return dict(color=color, linewidth=2.4, alpha=0.9, marker="o")
    if method == "ABC-PMC":
        return dict(color=color, linewidth=2.0, alpha=0.9, marker="o")
    return dict(color=color, linewidth=1.7, alpha=0.85, marker="o")


def _plot_context(rows: List[PerfRow]) -> Dict[str, str]:
    methods = {row.method for row in rows}
    first = rows[0]

    if methods == set(METHOD_ORDER):
        method_label = "all methods"
        method_tag = "all_methods"
    elif methods and methods <= {"permABC-SMC-OS", "permABC-SMC-UM"}:
        method_label = "OS/UM variants"
        method_tag = "osum"
    elif methods and not (methods & {"permABC-SMC-OS", "permABC-SMC-UM"}):
        method_label = "without OS/UM"
        method_tag = "no_osum"
    else:
        method_label = "selected methods"
        method_tag = "selected_methods"

    dataset = f"K={first.K}, outliers={first.K_outliers}, seed={first.seed}"
    stem = f"gaussian_abc_{method_tag}_K{first.K}_out{first.K_outliers}_seed{first.seed}"
    return {"method_label": method_label, "dataset": dataset, "stem": stem}


def _plot_epsilon_vs_cost(rows: List[PerfRow], figures_dir: Path) -> None:
    grouped = _rows_by_method(rows)
    context = _plot_context(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Gaussian ABC performance ({context['method_label']}; {context['dataset']})",
        fontsize=14,
        fontweight="bold",
    )

    for i, method in enumerate(METHOD_ORDER):
        if method not in grouped:
            continue
        subset = grouped[method]
        eps = np.asarray([r.epsilon for r in subset], dtype=np.float64)
        n_sim = np.asarray([r.n_sim for r in subset], dtype=np.float64)
        tm = np.asarray([r.time for r in subset], dtype=np.float64)
        style = _style_for_method(method, i)

        mask_sim = np.isfinite(eps) & np.isfinite(n_sim) & (eps > 0) & (n_sim > 0)
        if np.any(mask_sim):
            axes[0].plot(
                n_sim[mask_sim],
                eps[mask_sim],
                marker=style["marker"],
                linewidth=style["linewidth"],
                color=style["color"],
                alpha=style["alpha"],
                label=method,
            )

        mask_time = np.isfinite(eps) & np.isfinite(tm) & (eps > 0) & (tm > 0)
        if np.any(mask_time):
            axes[1].plot(
                tm[mask_time],
                eps[mask_time],
                marker=style["marker"],
                linewidth=style["linewidth"],
                color=style["color"],
                alpha=style["alpha"],
                label=method,
            )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("n_sim (chuẩn hóa)")
    axes[0].set_ylabel("epsilon")
    axes[0].set_title("Epsilon theo số mô phỏng")
    axes[0].grid(alpha=0.25)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time (chuẩn hóa, giây)")
    axes[1].set_ylabel("epsilon")
    axes[1].set_title("Epsilon theo thời gian")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig_path = figures_dir / f"{context['stem']}_epsilon_vs_cost.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


def _plot_error_vs_cost(rows: List[PerfRow], figures_dir: Path) -> None:
    grouped = _rows_by_method(rows)
    context = _plot_context(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Parameter error vs simulation cost ({context['method_label']}; {context['dataset']})",
        fontsize=14,
        fontweight="bold",
    )

    for i, method in enumerate(METHOD_ORDER):
        if method not in grouped:
            continue
        subset = grouped[method]
        x = np.asarray([r.n_sim for r in subset], dtype=np.float64)
        err_s2 = np.asarray([r.abs_err_sigma2 for r in subset], dtype=np.float64)
        err_mu = np.asarray([r.abs_err_mu_mean for r in subset], dtype=np.float64)
        style = _style_for_method(method, i)

        mask1 = np.isfinite(x) & np.isfinite(err_s2) & (x > 0)
        if np.any(mask1):
            axes[0].plot(
                x[mask1],
                err_s2[mask1],
                marker=style["marker"],
                linewidth=style["linewidth"],
                color=style["color"],
                alpha=style["alpha"],
                label=method,
            )

        mask2 = np.isfinite(x) & np.isfinite(err_mu) & (x > 0)
        if np.any(mask2):
            axes[1].plot(
                x[mask2],
                err_mu[mask2],
                marker=style["marker"],
                linewidth=style["linewidth"],
                color=style["color"],
                alpha=style["alpha"],
                label=method,
            )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("n_sim (chuẩn hóa)")
    axes[0].set_ylabel("|sigma2_est - sigma2_true|")
    axes[0].set_title("Sai số sigma2 theo n_sim")
    axes[0].grid(alpha=0.25)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("n_sim (chuẩn hóa)")
    axes[1].set_ylabel("|mu_mean_est - mu_mean_true|")
    axes[1].set_title("Sai số mu_mean theo n_sim")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig_path = figures_dir / f"{context['stem']}_error_vs_cost.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


def _plot_after_run(rows: List[PerfRow], output_dir: Path, figures_dir: Optional[Path]) -> None:
    target_dir = figures_dir or (output_dir / "figures_performance")
    target_dir.mkdir(parents=True, exist_ok=True)
    _plot_epsilon_vs_cost(rows, target_dir)
    _plot_error_vs_cost(rows, target_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark so sánh hiệu năng ABC methods (simulation)")

    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--K-outliers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--mu-0", type=float, default=0.0)
    parser.add_argument("--sigma-0", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=5.0)

    parser.add_argument("--n-points", type=int, default=1000000)
    parser.add_argument("--n-particles", type=int, default=1000)
    parser.add_argument(
        "--n-sim-budget",
        type=float,
        default=None,
        help="Budget in component-simulations for SMC/PMC. Default: n_points * K, matching experiments/script.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["all", "vanilla", "smc", "pmc", "osum"],
        default=["all"],
        help="Chọn nhóm phương pháp cần chạy. Dùng 'all' để chạy đầy đủ.",
    )

    parser.add_argument("--osum", dest="include_osum", action="store_true")
    parser.add_argument("--no-osum", dest="include_osum", action="store_false")
    parser.set_defaults(include_osum=True)

    parser.add_argument("--m0-values", nargs="+", type=int, default=None)
    parser.add_argument("--l0-values", nargs="+", type=int, default=None)
    parser.add_argument("--n-epsilon", type=int, default=10000)
    parser.add_argument("--alpha-epsilon", type=float, default=0.95)

    parser.add_argument("--output-dir", type=str, default="my-reproduces/results/simulation")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--figures-dir", type=str, default="")
    parser.add_argument("--plot-after-run", dest="plot_after_run", action="store_true")
    parser.add_argument("--no-plot-after-run", dest="plot_after_run", action="store_false")
    parser.set_defaults(plot_after_run=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_all = "all" in args.methods
    run_vanilla = run_all or ("vanilla" in args.methods)
    run_smc = run_all or ("smc" in args.methods)
    run_pmc = run_all or ("pmc" in args.methods)
    run_osum = (run_all or ("osum" in args.methods)) and args.include_osum
    n_sim_budget = float(args.n_sim_budget) if args.n_sim_budget is not None else float(args.n_points * args.K)
    m0_values = args.m0_values or list(np.asarray([1.5 * args.K, 2 * args.K, 5 * args.K, 7 * args.K, 10 * args.K, 15 * args.K, 20 * args.K, 25 * args.K], dtype=int))
    l0_values = args.l0_values or list(np.asarray(np.linspace(2, args.K, args.K), dtype=int))

    print("Performance comparison (simulation)")
    print(f"K={args.K}, K_outliers={args.K_outliers}, seed={args.seed}")
    print(f"n_points={args.n_points}, n_particles={args.n_particles}, n_sim_budget={n_sim_budget:.0f}")
    print(f"methods={args.methods}, include_osum={run_osum}")

    key, model, y_obs, true_theta = _setup_experiment(
        K=args.K,
        K_outliers=args.K_outliers,
        seed=args.seed,
        n_obs=args.n_obs,
        mu_0=args.mu_0,
        sigma_0=args.sigma_0,
        alpha=args.alpha,
        beta=args.beta,
    )

    rows: List[PerfRow] = []

    if run_vanilla:
        key, sub = random.split(key)
        print("Running vanilla group...")
        rows.extend(
            _run_vanilla_rows(
                key=sub,
                model=model,
                y_obs=y_obs,
                true_theta=true_theta,
                n_points=args.n_points,
                seed=args.seed,
                K=args.K,
                K_outliers=args.K_outliers,
            )
        )

    if run_smc or run_pmc:
        print("Running smc/pmc group...")
        rows.extend(
            _run_smc_pmc_rows(
                key=key,
                model=model,
                y_obs=y_obs,
                true_theta=true_theta,
                n_particles=args.n_particles,
                n_sim_budget=n_sim_budget,
                seed=args.seed,
                K=args.K,
                K_outliers=args.K_outliers,
                methods=args.methods,
            )
        )

    if run_osum:
        print("Running osum group...")
        rows.extend(
            _run_osum_rows(
                key=key,
                model=model,
                y_obs=y_obs,
                true_theta=true_theta,
                n_particles=args.n_particles,
                seed=args.seed,
                K=args.K,
                K_outliers=args.K_outliers,
                m0_values=m0_values,
                l0_values=l0_values,
                n_epsilon=args.n_epsilon,
                alpha_epsilon=args.alpha_epsilon,
            )
        )

    if not rows:
        raise RuntimeError("No results were produced. Check --methods and optional dependencies (pmc/osum).")

    rows.sort(key=lambda r: (METHOD_ORDER.index(r.method) if r.method in METHOD_ORDER else 999, r.epsilon))
    summary = _summarize(rows)

    prefix = args.prefix or f"performance_compare_K{args.K}_out{args.K_outliers}_seed{args.seed}"
    output_dir = Path(args.output_dir)
    _save(rows, summary, output_dir, prefix)

    if args.plot_after_run:
        target_figures_dir = Path(args.figures_dir) if args.figures_dir else None
        _plot_after_run(rows, output_dir, target_figures_dir)


if __name__ == "__main__":
    main()
