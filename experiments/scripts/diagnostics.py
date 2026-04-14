"""
Diagnostic functions for ABC posterior quality assessment.

Provides binned KL divergence and log-posterior scoring for Gaussian mixture
models with shared variance (sigma2).  Used by run_performance_comparison.py
and figure scripts (e.g. fig9).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _y_obs_to_2d(y_obs):
    """Return observed data as (K, n_obs)."""
    y = np.asarray(y_obs, dtype=float)
    if y.ndim == 3 and y.shape[0] == 1:
        y = y[0]
    elif y.ndim == 1:
        y = y.reshape(1, -1)
    return y


def _extract_loc_glob_arrays(thetas):
    """Extract loc, glob arrays from Theta-like object or dict."""
    if isinstance(thetas, dict):
        loc = np.asarray(thetas["loc"])
        glob = np.asarray(thetas["glob"])
    else:
        loc = np.asarray(getattr(thetas, "loc"))
        glob = np.asarray(getattr(thetas, "glob"))
    return loc, glob


def _apply_permutation_to_loc(loc, perm):
    """
    Apply per-particle permutation to loc.
    loc shape: (N, K, d_loc), perm shape: (N, K).
    """
    loc_arr = np.asarray(loc)
    perm_arr = np.asarray(perm, dtype=int)
    idx = np.arange(loc_arr.shape[0])[:, None]
    return loc_arr[idx, perm_arr]


# ---------------------------------------------------------------------------
# Sigma2 marginal: log-likelihood and posterior
# ---------------------------------------------------------------------------

def _log_marginal_lik_group_sigma2(y_group, sigma2, mu0, sigma0):
    """
    Log p(y_group | sigma2) after integrating out mu ~ N(mu0, sigma0^2),
    with likelihood y_i | mu, sigma2 ~ N(mu, sigma2).

    Uses matrix determinant lemma + Sherman-Morrison.
    """
    y = np.asarray(y_group, dtype=float).reshape(-1)
    n = y.shape[0]
    if sigma2 <= 0:
        return -np.inf

    a = float(sigma2)
    b = float(sigma0) ** 2
    logdet = (n - 1) * np.log(a) + np.log(a + n * b)

    yc = y - float(mu0)
    sum_yc = float(np.sum(yc))
    quad = float(np.dot(yc, yc))
    quad_form = (quad / a) - (b / (a * (a + n * b))) * (sum_yc ** 2)

    return -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad_form)


def _log_posterior_sigma2_unnorm(model, y_obs, sigma2):
    """Unnormalized log posterior for sigma2 given y_obs, integrating out mus."""
    from scipy.stats import invgamma

    lp = float(invgamma.logpdf(sigma2, a=float(model.alpha), scale=float(model.beta)))

    y = np.asarray(y_obs, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    K = y.shape[0]
    for k in range(K):
        lp += _log_marginal_lik_group_sigma2(
            y_group=y[k],
            sigma2=sigma2,
            mu0=float(model.mu_0),
            sigma0=float(model.sigma_0),
        )
    return lp


# ---------------------------------------------------------------------------
# Reference histogram bins for sigma2
# ---------------------------------------------------------------------------

def build_sigma2_reference_bins(model, y_obs, nbins=80, grid_size=4000, tail_prob=1e-4):
    """
    Fixed histogram edges for sigma2 covering both prior and posterior.

    Bins span from the prior's lower tail to the prior's upper tail,
    ensuring that both prior-like (early ABC) and posterior-like (late ABC)
    particles land inside the bins.

    Same ``edges`` must be reused for all methods / epsilon so KL_sigma2 is comparable.
    """
    from scipy.stats import invgamma

    # Prior range
    lo_prior = float(invgamma.ppf(tail_prob, a=float(model.alpha), scale=float(model.beta)))
    hi_prior = float(invgamma.ppf(1.0 - tail_prob, a=float(model.alpha), scale=float(model.beta)))

    lo_prior = max(lo_prior, 1e-8)
    hi_prior = max(hi_prior, lo_prior * 1.01)

    # Posterior mode range (for ensuring coverage)
    grid = np.exp(np.linspace(np.log(lo_prior), np.log(hi_prior), int(grid_size)))
    logp = np.array(
        [_log_posterior_sigma2_unnorm(model, y_obs, float(s)) for s in grid],
        dtype=float,
    )
    logp -= np.max(logp)
    dens = np.exp(logp)

    widths = np.diff(grid)
    dens_mids = 0.5 * (dens[:-1] + dens[1:])
    mass = dens_mids * widths
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    cdf /= max(cdf[-1], 1e-300)

    hi_post = float(np.interp(1.0 - tail_prob, cdf, grid))

    # Bins span from prior lower tail to max(prior upper, posterior upper)
    lo = lo_prior
    hi = max(hi_prior, hi_post)

    edges = np.exp(np.linspace(np.log(lo), np.log(hi), nbins + 1))
    return edges


# ---------------------------------------------------------------------------
# Sigma2 KL divergence (binned)
# ---------------------------------------------------------------------------

def _sigma2_histogram_masses(model, y_obs, thetas, weights=None, edges=None, eps=1e-12):
    """Normalized p_true and q_ABC histogram masses on fixed ``edges``."""
    if edges is None:
        raise ValueError("edges must be provided (use build_sigma2_reference_bins).")

    if isinstance(thetas, dict) and "glob" in thetas:
        sigma2_full = np.asarray(thetas["glob"]).reshape(-1)
    else:
        sigma2_full = np.asarray(getattr(thetas, "glob")).reshape(-1)

    if weights is not None:
        w_full = np.asarray(weights, dtype=float).reshape(-1)
        n = min(sigma2_full.size, w_full.size)
        sigma2_full = sigma2_full[:n]
        w_full = w_full[:n]
    else:
        w_full = None

    finite_pos = np.isfinite(sigma2_full) & (sigma2_full > 0)
    sigma2_s = sigma2_full[finite_pos]
    if sigma2_s.size < 5:
        return None, None

    if w_full is None:
        w = np.ones_like(sigma2_s, dtype=float)
    else:
        w = w_full[finite_pos]

    w = np.clip(w, 0.0, np.inf)
    if np.sum(w) <= 0:
        return None, None
    w = w / np.sum(w)

    q_mass, _ = np.histogram(sigma2_s, bins=edges, weights=w)
    q_mass = q_mass / max(np.sum(q_mass), eps)

    mids = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    logp = np.array(
        [_log_posterior_sigma2_unnorm(model, y_obs, float(s)) for s in mids],
        dtype=float,
    )
    logp -= np.max(logp)
    p_unn = np.exp(logp) * widths
    p_mass = p_unn / max(np.sum(p_unn), eps)

    q = np.clip(q_mass, eps, 1.0)
    p = np.clip(p_mass, eps, 1.0)
    return p, q


def empirical_kl_sigma2(model, y_obs, thetas, weights=None, edges=None,
                        nbins=80, eps=1e-12, direction="q_vs_p"):
    """
    Binned KL on sigma2 with fixed bins.

    ``direction``
        ``"q_vs_p"``  : KL( q_ABC || p_true ) = sum_b q_b [log q_b - log p_b]
        ``"p_vs_q"``  : KL( p_true || q_ABC ) = sum_b p_b [log p_b - log q_b]
    """
    if edges is None:
        edges = build_sigma2_reference_bins(model, y_obs, nbins=nbins)
    p, q = _sigma2_histogram_masses(
        model, y_obs, thetas, weights=weights, edges=edges, eps=eps
    )
    if p is None:
        return np.nan
    if direction == "q_vs_p":
        return float(np.sum(q * (np.log(q) - np.log(p))))
    else:
        return float(np.sum(p * (np.log(p) - np.log(q))))


# Backward-compatible aliases
def empirical_kl_sigma2_abc_vs_true(model, y_obs, thetas, weights=None, edges=None, nbins=80, eps=1e-12):
    return empirical_kl_sigma2(model, y_obs, thetas, weights=weights, edges=edges,
                               nbins=nbins, eps=eps, direction="q_vs_p")

def empirical_kl_sigma2_true_vs_abc(model, y_obs, thetas, weights=None, edges=None, nbins=80, eps=1e-12):
    return empirical_kl_sigma2(model, y_obs, thetas, weights=weights, edges=edges,
                               nbins=nbins, eps=eps, direction="p_vs_q")


# ---------------------------------------------------------------------------
# Joint posterior diagnostics
# ---------------------------------------------------------------------------

def _log_joint_posterior_unnorm(model, y2d, mu_vec, sigma2):
    """
    Unnormalized log posterior for one particle:
      log p(mu, sigma2 | y) = log p(sigma2) + sum_k log p(mu_k)
                             + sum_{k,i} log p(y_{k,i}|mu_k,sigma2)
    """
    from scipy.stats import invgamma

    if sigma2 <= 0 or not np.isfinite(sigma2):
        return -np.inf

    mu = np.asarray(mu_vec, dtype=float).reshape(-1)
    y = np.asarray(y2d, dtype=float)
    K = y.shape[0]
    if mu.shape[0] != K:
        return -np.inf

    mu0 = float(model.mu_0)
    var0 = float(model.sigma_0) ** 2

    lp = float(invgamma.logpdf(sigma2, a=float(model.alpha), scale=float(model.beta)))
    lp += np.sum(-0.5 * np.log(2.0 * np.pi * var0) - 0.5 * ((mu - mu0) ** 2) / var0)
    lp += np.sum(-0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * ((y - mu[:, None]) ** 2) / sigma2)
    return float(lp)


def _prepare_joint_particles(model, y_obs, thetas, weights=None, perm=None):
    """
    Shared preparation for joint-posterior diagnostics.

    Returns (mu_mat, sigma2, q, y2d) with invalid particles filtered out,
    or None if fewer than 5 valid particles remain.
    """
    y2d = _y_obs_to_2d(y_obs)
    loc, glob = _extract_loc_glob_arrays(thetas)

    if loc.ndim == 2:
        loc = loc[:, :, None]
    if glob.ndim == 2 and glob.shape[1] == 1:
        sigma2 = glob[:, 0]
    else:
        sigma2 = glob.reshape(-1)

    if perm is not None:
        loc = _apply_permutation_to_loc(loc, perm)

    mu_mat = loc[..., 0]
    n_part = mu_mat.shape[0]

    if weights is not None:
        w_arr = np.asarray(weights, dtype=float).reshape(-1)
        n_use = min(n_part, w_arr.size)
        mu_mat = mu_mat[:n_use]
        sigma2 = sigma2[:n_use]
        w_arr = w_arr[:n_use]
    else:
        w_arr = None
        n_use = n_part

    if n_use < 5:
        return None

    valid = np.isfinite(sigma2) & (sigma2 > 0) & np.all(np.isfinite(mu_mat), axis=1)
    mu_mat = mu_mat[valid]
    sigma2 = sigma2[valid]
    if mu_mat.shape[0] < 5:
        return None

    if w_arr is None:
        q = np.ones(mu_mat.shape[0], dtype=float)
    else:
        q = w_arr[valid]
    q = np.clip(q, 0.0, np.inf)
    if np.sum(q) <= 0:
        return None
    q = q / np.sum(q)

    return mu_mat, sigma2, q, y2d


def empirical_kl_joint(model, y_obs, thetas, weights=None, perm=None,
                       eps=1e-12, direction="q_vs_p"):
    """
    Discrete divergence on particle support (not a true KL vs continuous p).

    ``direction``
        ``"q_vs_p"`` : sum_i q_i [log q_i - log p_i],  p_i renormed on particles.
        ``"p_vs_q"`` : sum_i p_i [log p_i - log q_i]
    """
    prep = _prepare_joint_particles(model, y_obs, thetas, weights=weights, perm=perm)
    if prep is None:
        return np.nan
    mu_mat, sigma2, q, y2d = prep

    logp = np.array(
        [_log_joint_posterior_unnorm(model, y2d, mu_mat[i], float(sigma2[i]))
         for i in range(mu_mat.shape[0])],
        dtype=float,
    )
    finite = np.isfinite(logp) & np.isfinite(q)
    if np.sum(finite) < 5:
        return np.nan
    logp = logp[finite]
    q = q[finite]
    q = q / np.sum(q)

    logp = logp - np.max(logp)
    p = np.exp(logp)
    p = p / max(np.sum(p), eps)

    q = np.clip(q, eps, 1.0)
    p = np.clip(p, eps, 1.0)

    if direction == "q_vs_p":
        return float(np.sum(q * (np.log(q) - np.log(p))))
    else:
        return float(np.sum(p * (np.log(p) - np.log(q))))


# Backward-compatible aliases
def empirical_kl_joint_abc_vs_true(model, y_obs, thetas, weights=None, perm=None, eps=1e-12):
    return empirical_kl_joint(model, y_obs, thetas, weights=weights, perm=perm,
                              eps=eps, direction="q_vs_p")

def empirical_kl_joint_true_vs_abc(model, y_obs, thetas, weights=None, perm=None, eps=1e-12):
    return empirical_kl_joint(model, y_obs, thetas, weights=weights, perm=perm,
                              eps=eps, direction="p_vs_q")


def expected_neg_log_joint_true(model, y_obs, thetas, weights=None, perm=None):
    """
    Joint diagnostic:  - E_q [ log p*(theta | y) ]

    Lower score means q places more mass on high-posterior regions.
    Defined up to an additive constant (unnormalized log posterior).
    """
    prep = _prepare_joint_particles(model, y_obs, thetas, weights=weights, perm=perm)
    if prep is None:
        return np.nan
    mu_mat, sigma2, q, y2d = prep

    logp = np.array(
        [_log_joint_posterior_unnorm(model, y2d, mu_mat[i], float(sigma2[i]))
         for i in range(mu_mat.shape[0])],
        dtype=float,
    )

    finite = np.isfinite(logp)
    if np.sum(finite) < 5:
        return np.nan

    q = q[finite]
    q = q / np.sum(q)
    logp = logp[finite]

    return float(-np.sum(q * logp))


# ---------------------------------------------------------------------------
# Sigma2 posterior grid (shared helper for mu_k integration)
# ---------------------------------------------------------------------------

def _sigma2_posterior_grid(model, y_obs, grid_size=500, tail_prob=1e-4):
    """Normalised ``(sigma2_grid, weights)`` for numerical integration over sigma2."""
    from scipy.stats import invgamma

    lo = float(invgamma.ppf(tail_prob, a=float(model.alpha), scale=float(model.beta)))
    hi = float(invgamma.ppf(1 - tail_prob, a=float(model.alpha), scale=float(model.beta)))
    lo = max(lo, 1e-8)
    hi = max(hi, lo * 1.01)

    grid = np.exp(np.linspace(np.log(lo), np.log(hi), int(grid_size)))
    logp = np.array(
        [_log_posterior_sigma2_unnorm(model, y_obs, float(s)) for s in grid],
        dtype=float,
    )
    logp -= np.max(logp)
    w = np.exp(logp)

    # trapezoidal quadrature weights
    dg = np.diff(grid)
    trap = np.zeros(len(grid))
    trap[0] = dg[0] / 2
    trap[-1] = dg[-1] / 2
    trap[1:-1] = (dg[:-1] + dg[1:]) / 2
    w *= trap
    total = np.sum(w)
    if total <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= total

    return grid, w


# ---------------------------------------------------------------------------
# Mu_k marginals: analytical posterior and KL
# ---------------------------------------------------------------------------

def _mu_k_posterior_params(y_k, sigma2, mu0, sigma0):
    """Conjugate Normal posterior params for mu_k | y_k, sigma2."""
    n = len(y_k)
    prec0 = 1.0 / (sigma0 ** 2)
    prec_lik = n / sigma2
    prec_post = prec0 + prec_lik
    var_post = 1.0 / prec_post
    mean_post = var_post * (mu0 * prec0 + float(np.sum(y_k)) / sigma2)
    return mean_post, var_post


def _mu_k_marginal_density_on_grid(mu_grid, y_k, sigma2_grid, sigma2_weights, mu0, sigma0):
    """p(mu_k | y) evaluated on *mu_grid* by integrating out sigma2."""
    density = np.zeros(len(mu_grid), dtype=float)
    for j, s2 in enumerate(sigma2_grid):
        m, v = _mu_k_posterior_params(y_k, float(s2), mu0, sigma0)
        sd = np.sqrt(v)
        density += sigma2_weights[j] * np.exp(-0.5 * ((mu_grid - m) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))
    return density


def build_mu_reference_bins(model, y_obs, k, sigma2_grid, sigma2_weights,
                            nbins=60, tail_nsigma=5):
    """Fixed histogram edges for mu_k from the marginal posterior p(mu_k | y)."""
    y2d = _y_obs_to_2d(y_obs)
    y_k = y2d[k].ravel()
    mu0, sigma0 = float(model.mu_0), float(model.sigma_0)

    # centre on posterior at E[sigma2|y]
    s2_mean = float(np.sum(sigma2_grid * sigma2_weights))
    m, v = _mu_k_posterior_params(y_k, s2_mean, mu0, sigma0)
    sd = np.sqrt(v) * 1.5          # slightly wider
    lo = m - tail_nsigma * sd
    hi = m + tail_nsigma * sd
    return np.linspace(lo, hi, nbins + 1)


def empirical_kl_mu_avg(model, y_obs, thetas, weights=None, perm=None,
                        nbins=60, eps=1e-12, sigma2_grid_size=500):
    """
    Average binned KL(q_ABC || p_true) over all K mu_k marginals.

    The reference p(mu_k | y) is obtained by numerically integrating out sigma2.
    """
    y2d = _y_obs_to_2d(y_obs)
    K = y2d.shape[0]
    loc, glob = _extract_loc_glob_arrays(thetas)

    if loc.ndim == 2:
        loc = loc[:, :, None]
    if perm is not None:
        loc = _apply_permutation_to_loc(loc, perm)

    mu_mat = loc[..., 0]   # (N, K)
    N = mu_mat.shape[0]

    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()[:N]
        w = np.clip(w, 0.0, np.inf)
        if np.sum(w) <= 0:
            return np.nan
        w = w / np.sum(w)
    else:
        w = np.ones(N, dtype=float) / N

    sigma2_grid, sigma2_weights = _sigma2_posterior_grid(model, y_obs, grid_size=sigma2_grid_size)
    mu0, sigma0 = float(model.mu_0), float(model.sigma_0)

    kl_values = []
    for k in range(K):
        y_k = y2d[k].ravel()
        edges = build_mu_reference_bins(model, y_obs, k, sigma2_grid, sigma2_weights, nbins=nbins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        # true marginal density -> bin masses
        dens = _mu_k_marginal_density_on_grid(mids, y_k, sigma2_grid, sigma2_weights, mu0, sigma0)
        p_mass = dens * widths
        p_total = np.sum(p_mass)
        if p_total <= 0:
            continue
        p_mass = p_mass / p_total

        # ABC histogram
        q_mass, _ = np.histogram(mu_mat[:, k], bins=edges, weights=w)
        q_total = np.sum(q_mass)
        if q_total <= 0:
            continue
        q_mass = q_mass / q_total

        p = np.clip(p_mass, eps, 1.0)
        q = np.clip(q_mass, eps, 1.0)
        kl_values.append(float(np.sum(q * (np.log(q) - np.log(p)))))

    if not kl_values:
        return np.nan
    return float(np.mean(kl_values))


# ---------------------------------------------------------------------------
# Wasserstein-2 distances
# ---------------------------------------------------------------------------

def _w2_1d_particles_vs_density(particles, particle_weights,
                                density_grid, density_values,
                                n_quantiles=1000):
    """
    1-D Wasserstein-2 between weighted particles and a density on a grid.

    Uses quantile-function comparison:
        W_2^2 = int_0^1 |Q_abc(t) - Q_true(t)|^2 dt
    """
    # ABC quantile function
    order = np.argsort(particles)
    sp = particles[order]
    sw = particle_weights[order]
    cdf_abc = np.concatenate([[0.0], np.cumsum(sw)])
    sp_ext = np.concatenate([[sp[0]], sp])

    # true CDF from density (trapezoidal)
    dg = np.diff(density_grid)
    dens_mid = 0.5 * (density_values[:-1] + density_values[1:])
    mass = dens_mid * dg
    cdf_true = np.concatenate([[0.0], np.cumsum(mass)])
    cdf_total = cdf_true[-1]
    if cdf_total <= 0:
        return np.nan
    cdf_true = cdf_true / cdf_total

    t = np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1]
    q_abc = np.interp(t, cdf_abc, sp_ext)
    q_true = np.interp(t, cdf_true, density_grid)

    return float(np.sqrt(np.mean((q_abc - q_true) ** 2)))


def empirical_w2_sigma2(model, y_obs, thetas, weights=None,
                        grid_size=2000, n_quantiles=1000):
    """Wasserstein-2 distance on the sigma2 marginal."""
    from scipy.stats import invgamma

    if isinstance(thetas, dict) and "glob" in thetas:
        sigma2_full = np.asarray(thetas["glob"]).reshape(-1)
    else:
        sigma2_full = np.asarray(getattr(thetas, "glob")).reshape(-1)

    if weights is not None:
        w_full = np.asarray(weights, dtype=float).ravel()
        n = min(sigma2_full.size, w_full.size)
        sigma2_full = sigma2_full[:n]
        w_full = w_full[:n]
    else:
        w_full = None

    valid = np.isfinite(sigma2_full) & (sigma2_full > 0)
    sigma2_s = sigma2_full[valid]
    if sigma2_s.size < 5:
        return np.nan

    if w_full is None:
        w_s = np.ones(sigma2_s.size, dtype=float) / sigma2_s.size
    else:
        w_s = w_full[valid]
        w_s = np.clip(w_s, 0.0, np.inf)
        if np.sum(w_s) <= 0:
            return np.nan
        w_s = w_s / np.sum(w_s)

    # true posterior density on a log-spaced grid
    tail_prob = 1e-4
    lo = float(invgamma.ppf(tail_prob, a=float(model.alpha), scale=float(model.beta)))
    hi = float(invgamma.ppf(1 - tail_prob, a=float(model.alpha), scale=float(model.beta)))
    lo = max(lo, 1e-8)
    hi = max(hi, lo * 1.01)

    grid = np.exp(np.linspace(np.log(lo), np.log(hi), int(grid_size)))
    logp = np.array(
        [_log_posterior_sigma2_unnorm(model, y_obs, float(s)) for s in grid],
        dtype=float,
    )
    logp -= np.max(logp)
    dens = np.exp(logp)

    return _w2_1d_particles_vs_density(sigma2_s, w_s, grid, dens, n_quantiles)


def empirical_w2_mu_avg(model, y_obs, thetas, weights=None, perm=None,
                        sigma2_grid_size=500, density_grid_size=1000,
                        n_quantiles=1000):
    """Average Wasserstein-2 over all K mu_k marginals."""
    y2d = _y_obs_to_2d(y_obs)
    K = y2d.shape[0]
    loc, glob = _extract_loc_glob_arrays(thetas)

    if loc.ndim == 2:
        loc = loc[:, :, None]
    if perm is not None:
        loc = _apply_permutation_to_loc(loc, perm)

    mu_mat = loc[..., 0]   # (N, K)
    N = mu_mat.shape[0]

    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()[:N]
        w = np.clip(w, 0.0, np.inf)
        if np.sum(w) <= 0:
            return np.nan
        w = w / np.sum(w)
    else:
        w = np.ones(N, dtype=float) / N

    sigma2_grid, sigma2_weights = _sigma2_posterior_grid(model, y_obs, grid_size=sigma2_grid_size)
    mu0, sigma0 = float(model.mu_0), float(model.sigma_0)

    w2_values = []
    for k in range(K):
        y_k = y2d[k].ravel()

        # density grid for mu_k centred on posterior
        s2_mean = float(np.sum(sigma2_grid * sigma2_weights))
        m, v = _mu_k_posterior_params(y_k, s2_mean, mu0, sigma0)
        sd = np.sqrt(v) * 1.5
        mu_grid = np.linspace(m - 6 * sd, m + 6 * sd, int(density_grid_size))

        dens = _mu_k_marginal_density_on_grid(
            mu_grid, y_k, sigma2_grid, sigma2_weights, mu0, sigma0,
        )

        mu_particles = mu_mat[:, k]
        valid = np.isfinite(mu_particles)
        n_valid = int(np.sum(valid))
        if n_valid < 5:
            continue

        w_k = w[valid]
        w_k = w_k / np.sum(w_k)
        w2_k = _w2_1d_particles_vs_density(
            mu_particles[valid], w_k, mu_grid, dens, n_quantiles,
        )
        w2_values.append(w2_k)

    if not w2_values:
        return np.nan
    return float(np.mean(w2_values))


# ---------------------------------------------------------------------------
# True posterior sampler (Gaussian conjugate model)
# ---------------------------------------------------------------------------

def sample_true_posterior(model, y_obs, n_samples, rng=None):
    """
    Draw exact samples from the true joint posterior p(mu_1,...,mu_K, sigma2 | y).

    For the Gaussian conjugate model:
        sigma2 ~ InverseGamma(alpha', beta')   (marginal, integrating out mu)
        mu_k | sigma2, y_k ~ Normal(m_k, s_k^2)

    Returns (mu_samples, sigma2_samples) with shapes (n_samples, K) and (n_samples,).
    """
    from scipy.stats import invgamma

    if rng is None:
        rng = np.random.default_rng(42)

    y2d = _y_obs_to_2d(y_obs)
    K = y2d.shape[0]
    mu0, sigma0 = float(model.mu_0), float(model.sigma_0)
    alpha0, beta0 = float(model.alpha), float(model.beta)

    # --- Step 1: sample sigma2 from its marginal via rejection/grid sampling ---
    # Build CDF on a fine grid and invert
    s2_grid, s2_weights = _sigma2_posterior_grid(model, y_obs, grid_size=5000)

    # cumulative distribution
    cdf = np.cumsum(s2_weights)
    cdf = cdf / cdf[-1]
    # sample via inverse CDF
    u = rng.uniform(size=n_samples)
    sigma2_samples = np.interp(u, cdf, s2_grid)

    # --- Step 2: sample mu_k | sigma2, y_k ---
    mu_samples = np.empty((n_samples, K), dtype=float)
    for k in range(K):
        y_k = y2d[k].ravel()
        for i in range(n_samples):
            m, v = _mu_k_posterior_params(y_k, sigma2_samples[i], mu0, sigma0)
            mu_samples[i, k] = rng.normal(m, np.sqrt(v))

    return mu_samples, sigma2_samples


# ---------------------------------------------------------------------------
# Sliced Wasserstein-2 (joint)
# ---------------------------------------------------------------------------

def _w2_1d_sorted(a, b):
    """W2 between two 1-D arrays of equal length (unweighted)."""
    return float(np.sqrt(np.mean((np.sort(a) - np.sort(b)) ** 2)))


def _w2_1d_weighted_vs_unweighted(particles, weights, ref_sorted, n_quantiles=2000):
    """W2 between weighted empirical and sorted reference samples via quantiles."""
    order = np.argsort(particles)
    sp = particles[order]
    sw = weights[order]
    cdf_q = np.concatenate([[0.0], np.cumsum(sw)])
    sp_ext = np.concatenate([[sp[0]], sp])

    t = np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1]
    q_abc = np.interp(t, cdf_q, sp_ext)

    # reference quantiles (unweighted, already sorted)
    n_ref = len(ref_sorted)
    cdf_ref = np.linspace(0.0, 1.0, n_ref)
    q_ref = np.interp(t, cdf_ref, ref_sorted)

    return float(np.sqrt(np.mean((q_abc - q_ref) ** 2)))


def sliced_w2_joint(model, y_obs, thetas, weights=None, perm=None,
                    n_projections=200, n_ref_samples=5000, seed=0):
    """
    Sliced Wasserstein-2 distance between ABC particles and the true joint posterior.

    SW_2(q, p) = ( E_omega [ W_2^2( omega^T # q, omega^T # p ) ] )^{1/2}

    Works in the full (mu_1,...,mu_K, sigma2) space.
    """
    y2d = _y_obs_to_2d(y_obs)
    K = y2d.shape[0]
    loc, glob = _extract_loc_glob_arrays(thetas)

    if loc.ndim == 2:
        loc = loc[:, :, None]
    if perm is not None:
        loc = _apply_permutation_to_loc(loc, perm)

    mu_mat = loc[..., 0]   # (N, K)
    if glob.ndim == 2 and glob.shape[1] == 1:
        sigma2 = glob[:, 0]
    else:
        sigma2 = glob.reshape(-1)

    N = mu_mat.shape[0]

    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()[:N]
        w = np.clip(w, 0.0, np.inf)
        if np.sum(w) <= 0:
            return np.nan
        w = w / np.sum(w)
    else:
        w = np.ones(N, dtype=float) / N

    # filter invalid particles
    valid = np.isfinite(sigma2) & (sigma2 > 0) & np.all(np.isfinite(mu_mat), axis=1)
    if np.sum(valid) < 10:
        return np.nan
    mu_mat = mu_mat[valid]
    sigma2 = sigma2[valid]
    w = w[valid]
    w = w / np.sum(w)

    # ABC particles in joint space: (N, K+1)
    abc_joint = np.column_stack([mu_mat, sigma2])
    dim = abc_joint.shape[1]

    # true posterior samples
    rng = np.random.default_rng(seed)
    mu_ref, s2_ref = sample_true_posterior(model, y_obs, n_ref_samples, rng=rng)
    ref_joint = np.column_stack([mu_ref, s2_ref])

    # random projections
    rng_proj = np.random.default_rng(seed + 1)
    directions = rng_proj.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    sw2_sq = 0.0
    for omega in directions:
        proj_abc = abc_joint @ omega      # (N,)
        proj_ref = ref_joint @ omega      # (n_ref,)
        proj_ref_sorted = np.sort(proj_ref)

        w2 = _w2_1d_weighted_vs_unweighted(proj_abc, w, proj_ref_sorted)
        sw2_sq += w2 ** 2

    sw2_sq /= n_projections
    return float(np.sqrt(sw2_sq))
