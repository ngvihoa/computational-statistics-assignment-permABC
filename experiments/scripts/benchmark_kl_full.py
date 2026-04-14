#!/usr/bin/env python3
"""
Benchmark: full KL(q || p*) in dimension K+1 using k-NN estimator.

1. Sample from the true posterior p*(mu, sigma2 | y) exactly (conjugate model)
2. Sanity check: KL(p* || p*) ≈ 0
3. Run ABC-vanilla and permABC-vanilla at several epsilon levels
4. Compute full KL(q_ABC || p*) via k-NN and compare with score diagnostic

Usage:
    python benchmark_kl_full.py --K 5 --seed 42
    python benchmark_kl_full.py --K 20 --seed 42
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import invgamma
from scipy.spatial import cKDTree

# Project setup
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = None
for _p in _THIS.parents:
    if (_p / "pyproject.toml").exists():
        _PROJECT_ROOT = _p
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import random

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta
from diagnostics import (
    _log_posterior_sigma2_unnorm,
    expected_neg_log_joint_true,
    build_sigma2_reference_bins,
    empirical_kl_sigma2,
)


# ── Exact posterior sampler ──────────────────────────────────────────────────

def sample_true_posterior(model, y_obs, n_samples, rng, grid_size=5000):
    """Sample (mu_1,...,mu_K, sigma2) from the exact posterior.

    Step 1: sample sigma2 ~ p(sigma2 | y) via grid inverse-CDF
    Step 2: sample mu_k | sigma2, y ~ N(m_k, v_k)  (conjugate)
    """
    y2d = np.asarray(y_obs, dtype=float)
    if y2d.ndim == 3 and y2d.shape[0] == 1:
        y2d = y2d[0]
    K, n_obs = y2d.shape
    mu0, sigma0 = float(model.mu_0), float(model.sigma_0)
    var0 = sigma0 ** 2

    # Build CDF for sigma2
    lo = invgamma.ppf(1e-5, a=model.alpha, scale=model.beta)
    hi = invgamma.ppf(1 - 1e-5, a=model.alpha, scale=model.beta)
    grid = np.exp(np.linspace(np.log(max(lo, 1e-10)), np.log(hi), grid_size))
    logp = np.array([_log_posterior_sigma2_unnorm(model, y2d, float(s)) for s in grid])
    logp -= np.max(logp)
    dens = np.exp(logp)
    widths = np.diff(grid)
    mids_dens = 0.5 * (dens[:-1] + dens[1:])
    mass = mids_dens * widths
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    cdf /= cdf[-1]

    # Sample sigma2 via inverse CDF
    u = rng.uniform(0, 1, size=n_samples)
    sigma2_samples = np.interp(u, cdf, grid)

    # Sample mu_k | sigma2, y (conjugate Normal)
    y_bar = np.mean(y2d, axis=1)  # (K,)
    mu_samples = np.zeros((n_samples, K))
    for i in range(n_samples):
        s2 = sigma2_samples[i]
        v_k = 1.0 / (1.0 / var0 + n_obs / s2)   # posterior variance
        m_k = v_k * (mu0 / var0 + n_obs * y_bar / s2)  # posterior mean
        mu_samples[i] = rng.normal(m_k, np.sqrt(v_k))

    return mu_samples, sigma2_samples


def samples_to_array(mu, sigma2):
    """Stack (mu_1,...,mu_K, sigma2) into (N, K+1) array."""
    return np.column_stack([mu, sigma2[:, None]])


# ── k-NN KL estimator ───────────────────────────────────────────────────────

def kl_knn(samples_q, samples_p, k=5):
    """Estimate KL(q || p) using k-NN (Wang-Kulback-Jones 2009).

    KL(q||p) ≈ (d/N) sum_i [log(nu_k(x_i) / rho_k(x_i))] + log(M / (N-1))

    where rho_k = distance to k-th NN in q-samples (leave-one-out)
          nu_k  = distance to k-th NN in p-samples
          N = |q|, M = |p|, d = dimension
    """
    N = samples_q.shape[0]
    M = samples_p.shape[0]
    d = samples_q.shape[1]

    tree_q = cKDTree(samples_q)
    tree_p = cKDTree(samples_p)

    # k+1 because query includes point itself for q
    rho, _ = tree_q.query(samples_q, k=k + 1)
    rho_k = rho[:, -1]  # k-th NN distance (leave-one-out)

    nu, _ = tree_p.query(samples_q, k=k)
    nu_k = nu[:, -1]  # k-th NN distance

    # Avoid log(0)
    rho_k = np.maximum(rho_k, 1e-300)
    nu_k = np.maximum(nu_k, 1e-300)

    kl = (d / N) * np.sum(np.log(nu_k / rho_k)) + np.log(M / (N - 1))
    return float(kl)


# ── ABC vanilla runners ─────────────────────────────────────────────────────

def run_abc_vanilla(key, model, y_obs, N_points, perm=False):
    """Run ABC/permABC vanilla with epsilon=inf, return all simulations."""
    from permabc.algorithms.vanilla import abc_vanilla, perm_abc_vanilla

    key, subkey = random.split(key)
    if perm:
        # Returns: datas, thetas, dists, ys_index, zs_index, n_sim
        # loc is already permuted by perm_abc_vanilla
        datas, thetas, dists, ys_idx, zs_idx, n_sim = perm_abc_vanilla(
            subkey, model, N_points, np.inf, y_obs)
    else:
        # Returns: datas, thetas, dists, n_sim
        datas, thetas, dists, n_sim = abc_vanilla(
            subkey, model, N_points, np.inf, y_obs)

    # Sort by distance
    order = np.argsort(np.asarray(dists))
    loc = np.asarray(thetas.loc)[order]
    glob = np.asarray(thetas.glob)[order]
    dists_sorted = np.asarray(dists)[order]

    return {"loc": loc, "glob": glob, "distances": dists_sorted}


def extract_abc_samples(out, n_accept):
    """Extract the n_accept best particles as (N, K+1) array."""
    loc = np.asarray(out["loc"][:n_accept], dtype=float)
    glob = np.asarray(out["glob"][:n_accept], dtype=float)
    dists = np.asarray(out["distances"][:n_accept], dtype=float)

    mu = loc.reshape(n_accept, -1)        # (n, K) — flatten d_loc
    sigma2 = glob.reshape(n_accept, -1)[:, 0]  # (n,)

    return np.column_stack([mu, sigma2]), dists


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark full KL in dim K+1")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_obs", type=int, default=10)
    parser.add_argument("--N_points", type=int, default=1_000_000)
    parser.add_argument("--n_posterior", type=int, default=5000,
                        help="Number of true posterior samples")
    parser.add_argument("--k_nn", type=int, default=5,
                        help="k for k-NN KL estimator")
    return parser.parse_args()


def main():
    args = parse_arguments()
    K = args.K
    rng = np.random.default_rng(args.seed)
    key = random.PRNGKey(args.seed)

    figures_dir = _PROJECT_ROOT / "experiments" / "figures" / "kl_benchmark"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup model and data ─────────────────────────────────────────────
    model = GaussianWithNoSummaryStats(K=K, n_obs=args.n_obs, sigma_0=10, alpha=5, beta=5)
    key, sk = random.split(key)
    true_theta = model.prior_generator(sk, 1)
    true_theta = Theta(loc=true_theta.loc, glob=np.array([1.0])[None, :])
    key, sk = random.split(key)
    y_obs = model.data_generator(sk, true_theta)
    y2d = np.asarray(y_obs)
    if y2d.ndim == 3:
        y2d = y2d[0]

    print(f"K={K}, n_obs={args.n_obs}, dim={K+1}")
    print(f"True sigma2 = 1.0")

    # ── Sample true posterior ────────────────────────────────────────────
    print(f"\nSampling {args.n_posterior} from true posterior...")
    mu_true, s2_true = sample_true_posterior(model, y2d, args.n_posterior, rng)
    X_true = samples_to_array(mu_true, s2_true)
    print(f"  sigma2 range: [{s2_true.min():.3f}, {s2_true.max():.3f}], "
          f"mean={s2_true.mean():.3f}")

    # ── Sanity check: KL(p* || p*) ──────────────────────────────────────
    print(f"\nSanity check: KL(p* || p*) with two independent draws...")
    mu_true2, s2_true2 = sample_true_posterior(model, y2d, args.n_posterior, rng)
    X_true2 = samples_to_array(mu_true2, s2_true2)
    kl_self = kl_knn(X_true, X_true2, k=args.k_nn)
    print(f"  KL(p* || p*) = {kl_self:.4f}  (should be ≈ 0)")

    # ── Run ABC vanilla and permABC vanilla ──────────────────────────────
    print(f"\nRunning ABC-vanilla ({args.N_points:,} points)...")
    key, sk = random.split(key)
    out_abc = run_abc_vanilla(sk, model, y_obs, args.N_points, perm=False)

    print(f"Running permABC-vanilla ({args.N_points:,} points)...")
    key, sk = random.split(key)
    out_perm = run_abc_vanilla(sk, model, y_obs, args.N_points, perm=True)

    # ── Compute KL at various epsilon levels (= acceptance thresholds) ──
    accept_sizes = np.unique(np.geomspace(50, min(5000, args.N_points // 10), 12).astype(int))
    sigma2_edges = build_sigma2_reference_bins(model, y_obs, nbins=80)

    results = {"ABC-Vanilla": [], "permABC-Vanilla": []}

    for n_acc in accept_sizes:
        for name, out, is_perm in [
            ("ABC-Vanilla", out_abc, False),
            ("permABC-Vanilla", out_perm, True),
        ]:
            X_abc, dists = extract_abc_samples(out, n_acc)
            eps = float(dists[-1])  # epsilon = largest accepted distance

            # Full KL via k-NN
            kl_full = kl_knn(X_abc, X_true, k=args.k_nn)

            # Score diagnostic
            thetas_abc = Theta(
                loc=X_abc[:, :K][:, :, None],
                glob=X_abc[:, K:]
            )
            w = np.ones(len(X_abc)) / len(X_abc)
            score = expected_neg_log_joint_true(model, y_obs, thetas_abc, weights=w, perm=None)

            # Marginal KL on sigma2
            kl_s2 = empirical_kl_sigma2(model, y_obs, thetas_abc, weights=w,
                                         edges=sigma2_edges, direction="q_vs_p")

            # Entropy estimate via Kozachenko-Leonenko
            from scipy.special import digamma, gammaln
            tree_q = cKDTree(X_abc)
            rho, _ = tree_q.query(X_abc, k=args.k_nn + 1)
            rho_k = np.maximum(rho[:, -1], 1e-300)
            d = X_abc.shape[1]
            N_q = len(X_abc)
            # H(q) ≈ (d/N) sum log(rho_k) + log(N-1) - digamma(k) + log(V_d)
            # where V_d = pi^(d/2) / Gamma(d/2+1) is the volume of unit d-ball
            log_Vd = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)
            entropy_kl = (d / N_q) * np.sum(np.log(rho_k)) + \
                         np.log(N_q - 1) - digamma(args.k_nn) + log_Vd

            # KL(q||p*) = -H(q) + E_q[-log p*(θ|y)] = -entropy + score
            # (score is already -E_q[log p*(θ|y)])
            backward_kl_approx = -entropy_kl + score

            results[name].append({
                "n_accept": n_acc,
                "epsilon": eps,
                "kl_full_knn": kl_full,
                "score": score,
                "kl_sigma2": kl_s2,
                "entropy_kl": entropy_kl,
                "backward_kl": backward_kl_approx,
            })
            print(f"  {name:>20s}  n={n_acc:>5d}  eps={eps:>7.2f}  "
                  f"KL_full={kl_full:>8.3f}  score={score:>10.1f}  "
                  f"KL_s2={kl_s2:>8.4f}  H={entropy_kl:>8.1f}  "
                  f"KL_back={backward_kl_approx:>10.1f}")

    # ── Figures ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for name, color, marker in [
        ("ABC-Vanilla", "#d62728", "s"),
        ("permABC-Vanilla", "#2ca02c", "s"),
    ]:
        res = results[name]
        eps = [r["epsilon"] for r in res]
        kl_full = [r["kl_full_knn"] for r in res]
        scores = [r["score"] for r in res]
        kl_s2 = [r["kl_sigma2"] for r in res]
        n_acc = [r["n_accept"] for r in res]
        bkl = [r["backward_kl"] for r in res]

        # KL_full vs epsilon
        axes[0, 0].plot(eps, kl_full, label=name, color=color, marker=marker, linewidth=2, markersize=5)
        # Score vs epsilon
        axes[0, 1].plot(eps, scores, label=name, color=color, marker=marker, linewidth=2, markersize=5)
        # KL_sigma2 vs epsilon
        axes[0, 2].plot(eps, kl_s2, label=name, color=color, marker=marker, linewidth=2, markersize=5)
        # KL_full vs score
        axes[1, 0].plot(scores, kl_full, label=name, color=color, marker=marker, linewidth=2, markersize=5)
        # KL_full vs KL_sigma2
        axes[1, 1].plot(kl_s2, kl_full, label=name, color=color, marker=marker, linewidth=2, markersize=5)
        # Backward KL vs KL_full
        axes[1, 2].plot(kl_full, bkl, label=name, color=color, marker=marker, linewidth=2, markersize=5)

    axes[0, 0].set_xlabel(r"$\varepsilon$"); axes[0, 0].set_ylabel("KL_full (k-NN)")
    axes[0, 0].set_title(r"Full KL vs $\varepsilon$")
    axes[0, 1].set_xlabel(r"$\varepsilon$"); axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title(r"Score vs $\varepsilon$")
    axes[0, 2].set_xlabel(r"$\varepsilon$"); axes[0, 2].set_ylabel(r"$\mathrm{KL}_{\sigma^2}$")
    axes[0, 2].set_title(r"$\mathrm{KL}_{\sigma^2}$ vs $\varepsilon$")
    axes[1, 0].set_xlabel("Score"); axes[1, 0].set_ylabel("KL_full (k-NN)")
    axes[1, 0].set_title("KL_full vs Score")
    axes[1, 1].set_xlabel(r"$\mathrm{KL}_{\sigma^2}$"); axes[1, 1].set_ylabel("KL_full (k-NN)")
    axes[1, 1].set_title(r"KL_full vs $\mathrm{KL}_{\sigma^2}$")
    axes[1, 2].set_xlabel("KL_full (k-NN)"); axes[1, 2].set_ylabel(r"$-H(q) + \mathrm{score}$")
    axes[1, 2].set_title("Backward KL approx vs KL_full")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"KL benchmark — K={K}, dim={K+1}, seed={args.seed}", fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = figures_dir / f"kl_benchmark_K_{K}_seed_{args.seed}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nSanity: KL(p*||p*) = {kl_self:.4f}")
    print("\nCorrelation KL_full vs diagnostics:")
    for name in results:
        kl_f = np.array([r["kl_full_knn"] for r in results[name]])
        sc = np.array([r["score"] for r in results[name]])
        kl_s = np.array([r["kl_sigma2"] for r in results[name]])
        bk = np.array([r["backward_kl"] for r in results[name]])
        mask = np.isfinite(kl_f) & np.isfinite(sc) & np.isfinite(kl_s) & np.isfinite(bk)
        if mask.sum() > 3:
            print(f"  {name}:")
            print(f"    corr(KL_full, score)       = {np.corrcoef(kl_f[mask], sc[mask])[0,1]:.4f}")
            print(f"    corr(KL_full, KL_sigma2)   = {np.corrcoef(kl_f[mask], kl_s[mask])[0,1]:.4f}")
            print(f"    corr(KL_full, backward_KL) = {np.corrcoef(kl_f[mask], bk[mask])[0,1]:.4f}")


if __name__ == "__main__":
    main()
