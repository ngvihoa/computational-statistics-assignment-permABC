"""
Proposal kernels for ABC-SMC algorithms.

This module implements various kernel classes for parameter proposals in ABC-SMC,
including Random Walk kernels with adaptive variance and truncated proposals
for bounded parameter spaces.
"""

import jax.numpy as jnp
from jax import random
from ..utils.functions import Theta
import numpy as np
from scipy.stats import truncnorm as _scipy_truncnorm
try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

MIN_SAMPLES_FOR_VARIANCE = 25

class Kernel:
    """
    Base class for all proposal kernels in ABC-SMC.
    """

    def __init__(
        self,
        model,
        thetas: 'Theta',
        weights: np.ndarray,
        ys_index: Optional[np.ndarray],
        zs_index: Optional[np.ndarray],
        tau_loc_glob: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([])),
        L: int = 0,
        M: int = 0,
        verbose: int = 0
    ) -> None:
        self.model = model
        self.K = model.K
        self.thetas = thetas
        self.weights = np.asarray(weights)
        self.verbose = verbose

        self.L = L if L != 0 else self.K
        self.M = M if M != 0 else self.K

        self.ys_index = np.repeat([np.arange(model.K)], thetas.loc.shape[0], axis=0) if ys_index is None else np.asarray(ys_index, dtype=np.int32)
        self.zs_index = np.repeat([np.arange(model.K)], thetas.loc.shape[0], axis=0) if zs_index is None else np.asarray(zs_index, dtype=np.int32)

        if tau_loc_glob[0].size == 0 and tau_loc_glob[1].size == 0:
            self.tau_loc, self.tau_glob = self.get_rw_variance()
        else:
            self.tau_loc, self.tau_glob = np.asarray(tau_loc_glob[0]), np.asarray(tau_loc_glob[1])

        self.tau = self.set_rw_variance_by_particle()

    def get_rw_variance(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.L < self.K:
            tau_loc = self.get_tau_loc_under_matching()
        else:
            if self.K > 1:
                loc_np = np.asarray(self.thetas.loc)
                idx = np.arange(loc_np.shape[0])[:, None]
                permuted_loc = loc_np[idx, self.zs_index]
            else:
                permuted_loc = np.asarray(self.thetas.loc)
            tau_loc = np.sqrt(2 * np.var(permuted_loc, axis=0))

        tau_glob = np.sqrt(2 * np.var(np.asarray(self.thetas.glob), axis=0))
        return tau_loc, tau_glob

    def get_tau_loc_glob(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.tau_loc, self.tau_glob

    def get_tau_loc_under_matching(self) -> np.ndarray:
        n_particles = self.thetas.loc.shape[0]
        loc_np = np.asarray(self.thetas.loc)

        # Gather all matched parameters at once: (N, L, dim_loc)
        pidx = np.arange(n_particles)[:, None]
        params = loc_np[pidx, self.zs_index[:, :self.L]]
        targets = self.ys_index[:, :self.L]  # (N, L) — component indices

        tau_loc = np.zeros_like(loc_np[0])
        for k in range(self.K):
            mask = (targets == k)  # (N, L) bool
            count = mask.sum()
            if count > MIN_SAMPLES_FOR_VARIANCE:
                tau_loc[k] = np.sqrt(2 * np.var(params[mask], axis=0))
        return tau_loc

    def set_rw_variance_by_particle(self) -> 'Theta':
        n_particles = self.thetas.loc.shape[0]
        tau_loc_np = np.asarray(self.tau_loc)
        tau_loc_not_match = np.max(tau_loc_np, axis=0)

        if self.K > 1:
            out_loc = np.zeros(np.asarray(self.thetas.loc).shape, dtype=np.float64)
            out_loc[np.arange(n_particles)[:, None], self.zs_index[:, :self.L]] = tau_loc_np[self.ys_index]
            out_loc = np.where(out_loc == 0., tau_loc_not_match, out_loc)
        else:
            out_loc = np.repeat([tau_loc_np], n_particles, axis=0)

        out_glob = np.repeat([np.asarray(self.tau_glob)], n_particles, axis=0)
        return Theta(loc=out_loc, glob=out_glob)


    def sample(self, key, thetas: 'Theta') -> 'Theta':
        raise NotImplementedError("Kernel subclasses must implement `sample()` method.")

    def logpdf(self, thetas_prop: 'Theta') -> np.ndarray:
        raise NotImplementedError("Kernel subclasses must implement `logpdf()` method.")


class KernelRW(Kernel):
    """Random Walk Kernel — Gaussian proposals."""

    def sample(self, key) -> 'Theta':
        loc_np = np.asarray(self.thetas.loc)
        glob_np = np.asarray(self.thetas.glob)
        tau_loc_np = np.asarray(self.tau.loc)
        tau_glob_np = np.asarray(self.tau.glob)

        rng = np.random.default_rng(int(key[0]))
        proposed_loc = loc_np + rng.standard_normal(loc_np.shape) * tau_loc_np
        proposed_glob = glob_np + rng.standard_normal(glob_np.shape) * tau_glob_np

        return Theta(loc=proposed_loc, glob=proposed_glob)

    def logpdf(self, thetas_prop: 'Theta') -> np.ndarray:
        tau_loc = np.asarray(self.tau.loc)
        tau_glob = np.asarray(self.tau.glob)
        EPS = 1e-30
        tau_loc_safe = np.where(tau_loc > EPS, tau_loc, EPS)
        tau_glob_safe = np.where(tau_glob > EPS, tau_glob, EPS)
        logpdf_loc = -0.5 * ((np.asarray(thetas_prop.loc) - np.asarray(self.thetas.loc)) / tau_loc_safe) ** 2
        logpdf_glob = -0.5 * ((np.asarray(thetas_prop.glob) - np.asarray(self.thetas.glob)) / tau_glob_safe) ** 2
        return np.sum(logpdf_loc, axis=(1, 2)) + np.sum(logpdf_glob, axis=1)


class KernelTruncatedRW(Kernel):
    """Truncated Random Walk Kernel — uses scipy truncnorm for bounded parameters."""

    def sample(self, key) -> 'Theta':
        loc_min = np.asarray(self.model.support_par_loc[:, 0])
        loc_max = np.asarray(self.model.support_par_loc[:, 1])
        glob_min = np.asarray(self.model.support_par_glob[:, 0])
        glob_max = np.asarray(self.model.support_par_glob[:, 1])

        mu_loc = np.asarray(self.thetas.loc)
        mu_glob = np.asarray(self.thetas.glob)
        s_loc = np.asarray(self.tau.loc)
        s_glob = np.asarray(self.tau.glob)

        EPS = 1e-30
        s_loc_safe = np.where(s_loc > EPS, s_loc, EPS)
        s_glob_safe = np.where(s_glob > EPS, s_glob, EPS)

        a_loc = (loc_min - mu_loc) / s_loc_safe
        b_loc = (loc_max - mu_loc) / s_loc_safe
        a_glob = (glob_min - mu_glob) / s_glob_safe
        b_glob = (glob_max - mu_glob) / s_glob_safe

        proposed_loc = _scipy_truncnorm.rvs(
            a=a_loc, b=b_loc, loc=mu_loc, scale=s_loc_safe,
        )
        proposed_glob = _scipy_truncnorm.rvs(
            a=a_glob, b=b_glob, loc=mu_glob, scale=s_glob_safe,
        )

        return Theta(loc=proposed_loc, glob=proposed_glob)

    def logpdf(self, thetas_prop: 'Theta') -> np.ndarray:
        mu_loc = np.asarray(self.thetas.loc)
        mu_glob = np.asarray(self.thetas.glob)
        s_loc = np.asarray(self.tau.loc)
        s_glob = np.asarray(self.tau.glob)

        EPS = 1e-30
        s_loc_safe = np.where(s_loc > EPS, s_loc, EPS)
        s_glob_safe = np.where(s_glob > EPS, s_glob, EPS)

        a_loc = (np.asarray(self.model.support_par_loc[:, 0]) - mu_loc) / s_loc_safe
        b_loc = (np.asarray(self.model.support_par_loc[:, 1]) - mu_loc) / s_loc_safe
        a_glob = (np.asarray(self.model.support_par_glob[:, 0]) - mu_glob) / s_glob_safe
        b_glob = (np.asarray(self.model.support_par_glob[:, 1]) - mu_glob) / s_glob_safe

        logpdf_loc = _scipy_truncnorm.logpdf(
            np.asarray(thetas_prop.loc),
            a=a_loc, b=b_loc, loc=mu_loc, scale=s_loc_safe,
        )
        logpdf_glob = _scipy_truncnorm.logpdf(
            np.asarray(thetas_prop.glob),
            a=a_glob, b=b_glob, loc=mu_glob, scale=s_glob_safe,
        )
        return np.nansum(logpdf_loc, axis=(1, 2)) + np.nansum(logpdf_glob, axis=1)
