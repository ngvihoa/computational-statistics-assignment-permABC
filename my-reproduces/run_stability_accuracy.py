#!/usr/bin/env python3
"""CLI entrypoint for stability/accuracy synthetic benchmark runs."""

# -----------------------------------------------------------------------------
# HƯỚNG DẪN CHẠY NHANH 
#
# 1) Chạy mô hình G-and-K (so sánh permABC-SMC và ABC-SMC thường):
#  python3 my-reproduces/run_stability_accuracy.py --model gk --methods perm smc --seeds 0 1 2 --K 6 --n-obs 20 --n-particles 250 --n-iter-max 4
#
# 2) Chạy mô hình Lotka-Volterra:
#  python3 my-reproduces/run_stability_accuracy.py --model lotka --methods perm smc --seeds 0 1 2 --K 6 --n-obs 30 --n-particles 250 --n-iter-max 4
#
# Ý nghĩa các tham số chính:
# - --model: chọn bài toán mô phỏng (gk hoặc lotka)
# - --methods: phương pháp cần so sánh
#   + perm: permutation-enhanced ABC-SMC
#   + smc: ABC-SMC tiêu chuẩn (không hoán vị)
# - --seeds: các seed lặp lại để đánh giá độ ổn định
# - --K: số compartment/nhóm trao đổi được
# - --n-obs: số điểm quan sát mỗi compartment
# - --n-particles: số particle trong SMC (lớn hơn thì ổn định hơn nhưng chậm hơn)
# - --n-iter-max: số vòng lặp SMC tối đa
#
# Đầu ra:
# - File CSV theo từng seed: my-reproduces/results/*_per_seed.csv
# - File tổng hợp mean/std: my-reproduces/results/*_summary.json
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse

from repro_runner import BenchmarkConfig, run_benchmark


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Stability/accuracy simulation for permABC")
    parser.add_argument("--model", choices=["gk", "lotka"], default="gk")
    parser.add_argument("--methods", nargs="+", choices=["perm", "smc"], default=["perm", "smc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--true-seed", type=int, default=2026)
    parser.add_argument("--obs-seed", type=int, default=2027)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n-obs", type=int, default=25)
    parser.add_argument("--n-particles", type=int, default=600)
    parser.add_argument("--n-iter-max", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="my-reproduces/results")
    args = parser.parse_args()

    return BenchmarkConfig(
        model=args.model,
        methods=args.methods,
        seeds=args.seeds,
        true_seed=args.true_seed,
        obs_seed=args.obs_seed,
        K=args.K,
        n_obs=args.n_obs,
        n_particles=args.n_particles,
        n_iter_max=args.n_iter_max,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    run_benchmark(config)


if __name__ == "__main__":
    main()
