"""Chạy benchmark để đánh giá độ ổn định và độ chính xác của permABC so với ABC-SMC trên các mô hình mô phỏng."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from jax import random

from permabc.algorithms.smc import abc_smc, perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW

from repro_io import save_outputs
from repro_metrics import RunResult, extract_metrics, summarize
from repro_models_gk import GAndKModelForReproduce
from repro_models_lotka import LotkaVolterraModelForReproduce


@dataclass
class BenchmarkConfig:
    model: str
    methods: List[str]
    seeds: List[int]
    true_seed: int
    obs_seed: int
    K: int
    n_obs: int
    n_particles: int
    n_iter_max: int
    output_dir: str


def build_model(model_name: str, K: int, n_obs: int):
    if model_name == "gk":
        return GAndKModelForReproduce(K=K, n_obs=n_obs)
    if model_name == "lotka":
        return LotkaVolterraModelForReproduce(K=K, n_obs=n_obs)
    raise ValueError(f"Unsupported model: {model_name}")


def _run_method(method: str, model, seed: int, y_obs, n_particles: int, n_iter_max: int):
    key = random.PRNGKey(seed)
    start = time.time()

    if method == "perm":
        out = perm_abc_smc(
            key=key,
            model=model,
            n_particles=n_particles,
            epsilon_target=0.0,
            y_obs=y_obs,
            kernel=KernelTruncatedRW,
            verbose=0,
            N_iteration_max=n_iter_max,
            Final_iteration=0,
            try_identity=True,
            try_swaps=True,
            try_lsa=True,
        )
    elif method == "smc":
        out = abc_smc(
            key=key,
            model=model,
            n_particles=n_particles,
            epsilon_target=0.0,
            y_obs=y_obs,
            kernel=KernelTruncatedRW,
            verbose=0,
            N_iteration_max=n_iter_max,
            Final_iteration=0,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime = time.time() - start
    return out, runtime


def run_benchmark(config: BenchmarkConfig) -> Dict:
    model = build_model(config.model, K=config.K, n_obs=config.n_obs)

    key_true = random.PRNGKey(config.true_seed)
    true_theta = model.prior_generator(key_true, n_particles=1)

    key_obs = random.PRNGKey(config.obs_seed)
    y_obs = model.data_generator(key_obs, true_theta)

    print(f"Model: {config.model}")
    print(
        f"K={config.K}, n_obs={config.n_obs}, "
        f"n_particles={config.n_particles}, n_iter_max={config.n_iter_max}"
    )
    print(f"Methods: {config.methods}")
    print(f"Seeds: {config.seeds}")

    results: List[RunResult] = []
    for method in config.methods:
        for seed in config.seeds:
            print(f"Running method={method}, seed={seed} ...")
            out, runtime = _run_method(
                method=method,
                model=model,
                seed=seed,
                y_obs=y_obs,
                n_particles=config.n_particles,
                n_iter_max=config.n_iter_max,
            )
            metrics = extract_metrics(
                true_theta=true_theta,
                out=out,
                method=method,
                seed=seed,
                runtime_sec=runtime,
            )
            results.append(metrics)
            print(
                f"  mae_global={metrics.mae_global:.5f}, "
                f"mae_local_mean={metrics.mae_local_mean:.5f}, "
                f"runtime={metrics.runtime_sec:.2f}s"
            )

    summary = summarize(results)
    print("\nSummary:")
    import json

    print(json.dumps(summary, indent=2))

    from pathlib import Path

    prefix = f"{config.model}_K{config.K}_N{config.n_particles}"
    save_outputs(results=results, summary=summary, out_dir=Path(config.output_dir), prefix=prefix)
    print(f"\nSaved outputs in: {config.output_dir}")

    return {
        "summary": summary,
        "n_runs": len(results),
    }
