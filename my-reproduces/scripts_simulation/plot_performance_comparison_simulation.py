#!/usr/bin/env python3
"""Vẽ lại biểu đồ benchmark từ file CSV đã có trong my-reproduces/results/simulation.

Script này KHÔNG chạy lại mô phỏng. Nó chỉ đọc CSV output từ:
- my-reproduces/scripts_simulation/run_performance_comparison_simulation.py

Biểu đồ tạo ra:
1) Epsilon theo chi phí (n_sim chuẩn hóa, time chuẩn hóa)
2) Sai số tham số theo chi phí (sigma2, mu_mean)

Ví dụ:
    # Tự động lấy file CSV mới nhất trong thư mục kết quả mặc định
    python3 my-reproduces/scripts_simulation/plot_performance_comparison_simulation.py

    # Chỉ định prefix đã dùng khi chạy benchmark
    python3 my-reproduces/scripts_simulation/plot_performance_comparison_simulation.py \
        --prefix perf_compare_like_experiments_K20

    # Chỉ định trực tiếp file CSV
    python3 my-reproduces/scripts_simulation/plot_performance_comparison_simulation.py \
        --csv my-reproduces/results/simulation/perf_compare_like_experiments_K20.csv

    # Lưu hình vào thư mục khác
    python3 my-reproduces/scripts_simulation/plot_performance_comparison_simulation.py \
        --prefix perf_compare_like_experiments_K20 \
        --figures-dir my-reproduces/results/simulation/figures_rerendered
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def _load_rows(csv_path: Path) -> List[PerfRow]:
    rows: List[PerfRow] = []
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
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
        }
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CSV thiếu cột bắt buộc: {sorted(missing)}")

        for r in reader:
            rows.append(
                PerfRow(
                    method=str(r["method"]),
                    epsilon=float(r["epsilon"]),
                    n_sim=float(r["n_sim"]),
                    time=float(r["time"]),
                    n_sim_raw=float(r["n_sim_raw"]),
                    time_raw=float(r["time_raw"]),
                    abs_err_sigma2=float(r["abs_err_sigma2"]),
                    abs_err_mu_mean=float(r["abs_err_mu_mean"]),
                    seed=int(float(r["seed"])),
                    K=int(float(r["K"])),
                    K_outliers=int(float(r["K_outliers"])),
                )
            )

    if not rows:
        raise ValueError(f"CSV rỗng: {csv_path}")
    return rows


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


def _plot_epsilon_vs_cost(rows: List[PerfRow], figures_dir: Path) -> Path:
    grouped = _rows_by_method(rows)
    context = _plot_context(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
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
        fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.10))

    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig_path = figures_dir / f"{context['stem']}_epsilon_vs_cost.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _plot_error_vs_cost(rows: List[PerfRow], figures_dir: Path) -> Path:
    grouped = _rows_by_method(rows)
    context = _plot_context(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
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
        fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.10))

    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig_path = figures_dir / f"{context['stem']}_error_vs_cost.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _resolve_csv_path(results_dir: Path, csv_arg: str, prefix: str) -> Path:
    if csv_arg:
        csv_path = Path(csv_arg)
        if not csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")
        return csv_path

    if prefix:
        candidate = results_dir / f"{prefix}.csv"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Không tìm thấy file theo prefix: {candidate}")

    csv_files = sorted(results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy CSV trong {results_dir}")
    return csv_files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vẽ lại benchmark ABC từ CSV đã lưu")
    parser.add_argument("--results-dir", type=str, default="my-reproduces/results/simulation")
    parser.add_argument("--csv", type=str, default="", help="Đường dẫn CSV cụ thể. Ưu tiên cao nhất.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix output khi chạy benchmark (vd: perf_compare_like_experiments_K20)")
    parser.add_argument("--figures-dir", type=str, default="", help="Thư mục lưu hình. Mặc định: <results-dir>/figures_performance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    csv_path = _resolve_csv_path(results_dir, args.csv, args.prefix)

    rows = _load_rows(csv_path)

    figures_dir = Path(args.figures_dir) if args.figures_dir else (results_dir / "figures_performance")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig1 = _plot_epsilon_vs_cost(rows, figures_dir)
    fig2 = _plot_error_vs_cost(rows, figures_dir)

    print(f"Loaded CSV: {csv_path}")
    print(f"Saved figure: {fig1}")
    print(f"Saved figure: {fig2}")


if __name__ == "__main__":
    main()
