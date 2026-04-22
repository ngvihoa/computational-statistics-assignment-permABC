#!/usr/bin/env python3
"""Chay mo phong so sanh cac phuong phap ABC tren Gaussian toy model.

Muc tieu:
1) So sanh nhom phuong phap:
   - ABC-Vanilla
   - permABC-Vanilla
   - ABC-SMC
   - ABC-PMC
   - permABC-SMC
2) So sanh rieng do on dinh:
   - permABC-SMC vs ABC-Gibbs

Ghi chu:
- Script nay dung model GaussianWithNoSummaryStats de tuong thich ABC-Gibbs.
- Ket qua duoc luu thanh CSV + JSON trong my-reproduces/results.

Vi du chay:
    python3 my-reproduces/run_method_comparisons.py --task all
    python3 my-reproduces/run_method_comparisons.py --task gibbs
    python3 my-reproduces/run_method_comparisons.py --task all --seeds 0 1 2 3 --K 8 --n-obs 40 --n-particles 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from jax import random

from permabc.algorithms.pmc import abc_pmc
from permabc.algorithms.smc import abc_smc, perm_abc_smc
from permabc.algorithms.vanilla import abc_vanilla, perm_abc_vanilla
from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.utils.functions import Theta


@dataclass
class MethodRecord:
    group: str
    method: str
    seed: int
    runtime_sec: float
    n_sim_total: float
    eps_final: float
    sigma2_true: float
    sigma2_est: float
    abs_err_sigma2: float
    mu_mean_true: float
    mu_mean_est: float
    abs_err_mu_mean: float
    status: str
    note: str


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[:, None], axis=0)


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


def _extract_result_from_smc_like(result: Dict) -> Dict[str, float]:
    theta_last = result["Thetas"][-1]
    weights_last = np.asarray(result["Weights"][-1], dtype=np.float64)
    est = _estimate_from_theta(theta_last, weights=weights_last)
    return {
        **est,
        "n_sim_total": float(np.sum(result.get("N_sim", [np.nan]))),
        "eps_final": float(result.get("Eps_values", [np.nan])[-1]),
    }


def _extract_result_from_vanilla(thetas: Theta, n_sim_total: float, eps: float) -> Dict[str, float]:
    est = _estimate_from_theta(thetas, weights=None)
    return {
        **est,
        "n_sim_total": float(n_sim_total),
        "eps_final": float(eps),
    }


def _run_abc_gibbs(seed: int, model: GaussianWithNoSummaryStats, y_obs: np.ndarray, T: int, M_mu: int, M_sigma2: int):
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from abc_gibbs_gaussian import run_gibbs_sampler

    y2d = np.asarray(y_obs)
    if y2d.ndim == 3 and y2d.shape[0] == 1:
        y2d = y2d[0]

    key = random.PRNGKey(seed)
    t0 = time.time()
    mus, sigma2s, _, eps_sigma2, times, n_sim_per_iter = run_gibbs_sampler(
        key=key,
        model=model,
        y_obs_2d=y2d,
        T=T,
        M_mu=M_mu,
        M_sigma2=M_sigma2,
    )
    runtime_sec = time.time() - t0

    chain_mu = np.asarray(mus[1:])
    chain_s2 = np.asarray(sigma2s[1:])
    theta_chain = Theta(
        loc=chain_mu[:, :, None],
        glob=chain_s2[:, None],
    )
    est = _estimate_from_theta(theta_chain, weights=None)

    return {
        **est,
        "runtime_sec": float(runtime_sec if runtime_sec > 0 else np.sum(times)),
        "n_sim_total": float(T * n_sim_per_iter),
        "eps_final": float(np.mean(np.sqrt(np.asarray(eps_sigma2)))) if len(eps_sigma2) else np.nan,
    }


def _compute_calibrated_epsilon(model: GaussianWithNoSummaryStats, y_obs: np.ndarray, key, n_calib: int, q: float) -> float:
    key_t, key_z = random.split(key)
    thetas = model.prior_generator(key_t, n_calib)
    zs = model.data_generator(key_z, thetas)
    dists = np.asarray(model.distance(zs, y_obs), dtype=np.float64)
    return float(np.quantile(dists, q))


def _build_record(
    group: str,
    method: str,
    seed: int,
    sigma2_true: float,
    mu_mean_true: float,
    runtime_sec: float,
    out: Dict[str, float],
    status: str = "ok",
    note: str = "",
) -> MethodRecord:
    sigma2_est = float(out.get("sigma2_est", np.nan))
    mu_mean_est = float(out.get("mu_mean_est", np.nan))
    return MethodRecord(
        group=group,
        method=method,
        seed=seed,
        runtime_sec=float(runtime_sec),
        n_sim_total=float(out.get("n_sim_total", np.nan)),
        eps_final=float(out.get("eps_final", np.nan)),
        sigma2_true=float(sigma2_true),
        sigma2_est=sigma2_est,
        abs_err_sigma2=float(abs(sigma2_est - sigma2_true)) if np.isfinite(sigma2_est) else np.nan,
        mu_mean_true=float(mu_mean_true),
        mu_mean_est=mu_mean_est,
        abs_err_mu_mean=float(abs(mu_mean_est - mu_mean_true)) if np.isfinite(mu_mean_est) else np.nan,
        status=status,
        note=note,
    )


def _save_records(records: List[MethodRecord], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    grouped: Dict[str, List[MethodRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.method, []).append(rec)

    summary: Dict[str, Dict[str, float]] = {}
    for method, rows in grouped.items():
        sigma_err = np.asarray([r.abs_err_sigma2 for r in rows], dtype=np.float64)
        mu_err = np.asarray([r.abs_err_mu_mean for r in rows], dtype=np.float64)
        runtime = np.asarray([r.runtime_sec for r in rows], dtype=np.float64)
        nsim = np.asarray([r.n_sim_total for r in rows], dtype=np.float64)
        eps = np.asarray([r.eps_final for r in rows], dtype=np.float64)

        summary[method] = {
            "runs": len(rows),
            "abs_err_sigma2_mean": float(np.nanmean(sigma_err)),
            "abs_err_sigma2_std": float(np.nanstd(sigma_err)),
            "abs_err_mu_mean_mean": float(np.nanmean(mu_err)),
            "abs_err_mu_mean_std": float(np.nanstd(mu_err)),
            "runtime_sec_mean": float(np.nanmean(runtime)),
            "runtime_sec_std": float(np.nanstd(runtime)),
            "n_sim_total_mean": float(np.nanmean(nsim)),
            "n_sim_total_std": float(np.nanstd(nsim)),
            "eps_final_mean": float(np.nanmean(eps)),
            "eps_final_std": float(np.nanstd(eps)),
            "ok_runs": int(sum(1 for r in rows if r.status == "ok")),
        }

    json_path = output_dir / f"{prefix}_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary}, file, indent=2)


def run_all_methods(args, model, y_obs, true_theta) -> List[MethodRecord]:
    records: List[MethodRecord] = []

    sigma2_true = float(np.asarray(true_theta.glob[0, 0]))
    mu_mean_true = float(np.mean(np.asarray(true_theta.loc[0, :, 0])))

    eps = _compute_calibrated_epsilon(
        model=model,
        y_obs=y_obs,
        key=random.PRNGKey(args.eps_seed),
        n_calib=args.n_calib,
        q=args.eps_quantile,
    )
    print(f"[all] Calibrated epsilon for vanilla methods: {eps:.6f}")

    for seed in args.seeds:
        print(f"[all] seed={seed}")

        # ABC-Vanilla
        t0 = time.time()
        try:
            _, thetas, _, n_sim = abc_vanilla(
                key=random.PRNGKey(seed),
                model=model,
                n_points=args.n_points_vanilla,
                epsilon=eps,
                y_obs=y_obs,
            )
            out = _extract_result_from_vanilla(thetas=thetas, n_sim_total=n_sim, eps=eps)
            records.append(_build_record("all", "ABC-Vanilla", seed, sigma2_true, mu_mean_true, time.time() - t0, out))
        except Exception as exc:
            records.append(_build_record("all", "ABC-Vanilla", seed, sigma2_true, mu_mean_true, time.time() - t0, {}, "failed", str(exc)))

        # permABC-Vanilla
        t0 = time.time()
        try:
            _, thetas_p, _, _, _, n_sim_p = perm_abc_vanilla(
                key=random.PRNGKey(seed + 10_000),
                model=model,
                n_points=args.n_points_vanilla,
                epsilon=eps,
                y_obs=y_obs,
                try_swaps=True,
                try_lsa=True,
            )
            out = _extract_result_from_vanilla(thetas=thetas_p, n_sim_total=n_sim_p, eps=eps)
            records.append(_build_record("all", "permABC-Vanilla", seed, sigma2_true, mu_mean_true, time.time() - t0, out))
        except Exception as exc:
            records.append(_build_record("all", "permABC-Vanilla", seed, sigma2_true, mu_mean_true, time.time() - t0, {}, "failed", str(exc)))

        # ABC-SMC
        t0 = time.time()
        try:
            res_smc = abc_smc(
                key=random.PRNGKey(seed + 20_000),
                model=model,
                n_particles=args.n_particles,
                epsilon_target=0.0,
                y_obs=y_obs,
                kernel=KernelTruncatedRW,
                verbose=0,
                N_iteration_max=args.n_iter_max,
                Final_iteration=0,
            )
            out = _extract_result_from_smc_like(res_smc)
            records.append(_build_record("all", "ABC-SMC", seed, sigma2_true, mu_mean_true, time.time() - t0, out))
        except Exception as exc:
            records.append(_build_record("all", "ABC-SMC", seed, sigma2_true, mu_mean_true, time.time() - t0, {}, "failed", str(exc)))

        # ABC-PMC
        t0 = time.time()
        try:
            res_pmc = abc_pmc(
                key=random.PRNGKey(seed + 30_000),
                model=model,
                n_particles=args.n_particles,
                epsilon_target=0.0,
                alpha=args.pmc_alpha,
                y_obs=y_obs,
                verbose=0,
                N_sim_max=args.n_sim_max,
            )
            out = _extract_result_from_smc_like(res_pmc)
            records.append(_build_record("all", "ABC-PMC", seed, sigma2_true, mu_mean_true, time.time() - t0, out))
        except Exception as exc:
            records.append(_build_record("all", "ABC-PMC", seed, sigma2_true, mu_mean_true, time.time() - t0, {}, "failed", str(exc)))

        # permABC-SMC
        t0 = time.time()
        try:
            res_perm = perm_abc_smc(
                key=random.PRNGKey(seed + 40_000),
                model=model,
                n_particles=args.n_particles,
                epsilon_target=0.0,
                y_obs=y_obs,
                kernel=KernelTruncatedRW,
                verbose=0,
                N_iteration_max=args.n_iter_max,
                Final_iteration=0,
                try_identity=True,
                try_swaps=True,
                try_lsa=True,
            )
            out = _extract_result_from_smc_like(res_perm)
            records.append(_build_record("all", "permABC-SMC", seed, sigma2_true, mu_mean_true, time.time() - t0, out))
        except Exception as exc:
            records.append(_build_record("all", "permABC-SMC", seed, sigma2_true, mu_mean_true, time.time() - t0, {}, "failed", str(exc)))

    return records


def run_perm_vs_gibbs(args, model, y_obs, true_theta) -> List[MethodRecord]:
    records: List[MethodRecord] = []

    sigma2_true = float(np.asarray(true_theta.glob[0, 0]))
    mu_mean_true = float(np.mean(np.asarray(true_theta.loc[0, :, 0])))

    for seed in args.seeds:
        print(f"[gibbs] seed={seed}")

        # permABC-SMC
        t0 = time.time()
        try:
            res_perm = perm_abc_smc(
                key=random.PRNGKey(seed + 50_000),
                model=model,
                n_particles=args.n_particles,
                epsilon_target=0.0,
                y_obs=y_obs,
                kernel=KernelTruncatedRW,
                verbose=0,
                N_iteration_max=args.n_iter_max,
                Final_iteration=0,
                try_identity=True,
                try_swaps=True,
                try_lsa=True,
            )
            out_perm = _extract_result_from_smc_like(res_perm)
            records.append(_build_record("gibbs", "permABC-SMC", seed, sigma2_true, mu_mean_true, time.time() - t0, out_perm))
        except Exception as exc:
            records.append(_build_record("gibbs", "permABC-SMC", seed, sigma2_true, mu_mean_true, time.time() - t0, {}, "failed", str(exc)))

        # ABC-Gibbs
        t0 = time.time()
        try:
            out_gibbs = _run_abc_gibbs(
                seed=seed + 60_000,
                model=model,
                y_obs=y_obs,
                T=args.gibbs_T,
                M_mu=args.gibbs_M_mu,
                M_sigma2=args.gibbs_M_sigma2,
            )
            records.append(_build_record("gibbs", "ABC-Gibbs", seed, sigma2_true, mu_mean_true, time.time() - t0, out_gibbs))
        except Exception as exc:
            records.append(_build_record("gibbs", "ABC-Gibbs", seed, sigma2_true, mu_mean_true, time.time() - t0, {}, "failed", str(exc)))

    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ABC methods on Gaussian toy model")
    parser.add_argument("--task", choices=["all", "gibbs", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n-obs", type=int, default=30)
    parser.add_argument("--mu-0", type=float, default=0.0)
    parser.add_argument("--sigma-0", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=5.0)

    parser.add_argument("--true-seed", type=int, default=2026)
    parser.add_argument("--obs-seed", type=int, default=2027)

    parser.add_argument("--n-particles", type=int, default=600)
    parser.add_argument("--n-iter-max", type=int, default=7)
    parser.add_argument("--n-points-vanilla", type=int, default=300)

    parser.add_argument("--n-calib", type=int, default=3000)
    parser.add_argument("--eps-quantile", type=float, default=0.20)
    parser.add_argument("--eps-seed", type=int, default=2028)

    parser.add_argument("--pmc-alpha", type=float, default=0.95)
    parser.add_argument("--n-sim-max", type=float, default=2e6)

    parser.add_argument("--gibbs-T", type=int, default=300)
    parser.add_argument("--gibbs-M-mu", type=int, default=40)
    parser.add_argument("--gibbs-M-sigma2", type=int, default=80)

    parser.add_argument("--output-dir", type=str, default="my-reproduces/results")
    return parser.parse_args()


def main():
    args = parse_args()

    model = GaussianWithNoSummaryStats(
        K=args.K,
        n_obs=args.n_obs,
        mu_0=args.mu_0,
        sigma_0=args.sigma_0,
        alpha=args.alpha,
        beta=args.beta,
    )

    key_true = random.PRNGKey(args.true_seed)
    true_theta = model.prior_generator(key_true, n_particles=1)

    key_obs = random.PRNGKey(args.obs_seed)
    y_obs = model.data_generator(key_obs, true_theta)

    output_dir = Path(args.output_dir)

    if args.task in ("all", "both"):
        all_records = run_all_methods(args, model, y_obs, true_theta)
        _save_records(all_records, output_dir, prefix=f"compare_all_K{args.K}_N{args.n_particles}")
        print(f"Saved all-method comparison to {output_dir}")

    if args.task in ("gibbs", "both"):
        gibbs_records = run_perm_vs_gibbs(args, model, y_obs, true_theta)
        _save_records(gibbs_records, output_dir, prefix=f"compare_gibbs_K{args.K}_N{args.n_particles}")
        print(f"Saved perm-vs-gibbs comparison to {output_dir}")


if __name__ == "__main__":
    main()
