"""Mô hình mô phỏng dữ liệu G-and-k."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp
from scipy.stats import norm

from permabc.models import ModelBase
from permabc.utils.functions import Theta


class GAndKModelForReproduce(ModelBase):
    def __init__(
        self,
        K: int,
        n_obs: int = 25,
        mu0: float = 0.0,
        sigma0: float = 2.0,
        b_bounds: Tuple[float, float] = (0.5, 2.0),
        g_bounds: Tuple[float, float] = (0.0, 4.0),
        k_bounds: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(K)
        self.n_obs = n_obs
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.b_bounds = b_bounds
        self.g_bounds = g_bounds
        self.k_bounds = k_bounds

        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])
        self.support_par_glob = jnp.array(
            [
                [b_bounds[0], b_bounds[1]],
                [g_bounds[0], g_bounds[1]],
                [k_bounds[0], k_bounds[1]],
            ]
        )

        self.dim_loc = 1
        self.dim_glob = 3
        self.loc_name = ["$\\theta_{"]
        self.glob_name = ["$b$", "$g$", "$k$"]

    @staticmethod
    def _gk_transform(z: np.ndarray, b: np.ndarray, g: np.ndarray, k: np.ndarray) -> np.ndarray:
        c = 0.8
        skew = 1.0 + c * np.tanh(0.5 * g * z)
        tail = (1.0 + z * z) ** k
        return b * skew * tail * z

    def prior_generator(self, key, n_particles, n_silos=0):
        if n_silos == 0:
            n_silos = self.K
        rng = np.random.default_rng(int(key[0]))

        loc = rng.standard_normal((n_particles, n_silos, 1)) * self.sigma0 + self.mu0
        b = rng.uniform(*self.b_bounds, size=(n_particles, 1))
        g = rng.uniform(*self.g_bounds, size=(n_particles, 1))
        k = rng.uniform(*self.k_bounds, size=(n_particles, 1))
        glob = np.hstack([b, g, k])
        return Theta(loc=loc, glob=glob)

    def prior_logpdf(self, thetas):
        loc_np = np.asarray(thetas.loc)
        glob_np = np.asarray(thetas.glob)

        lp_loc = np.sum(norm.logpdf(loc_np, loc=self.mu0, scale=self.sigma0), axis=(1, 2))

        in_support = (
            (glob_np[:, 0] >= self.b_bounds[0])
            & (glob_np[:, 0] <= self.b_bounds[1])
            & (glob_np[:, 1] >= self.g_bounds[0])
            & (glob_np[:, 1] <= self.g_bounds[1])
            & (glob_np[:, 2] >= self.k_bounds[0])
            & (glob_np[:, 2] <= self.k_bounds[1])
        )

        log_vol = np.log(self.b_bounds[1] - self.b_bounds[0])
        log_vol += np.log(self.g_bounds[1] - self.g_bounds[0])
        log_vol += np.log(self.k_bounds[1] - self.k_bounds[0])
        lp_glob = np.where(in_support, -log_vol, -np.inf)
        return lp_loc + lp_glob

    def data_generator(self, key, thetas):
        n_particles, n_silos = thetas.loc.shape[:2]
        rng = np.random.default_rng(int(key[0]))

        z = rng.standard_normal((n_particles, n_silos, self.n_obs))
        loc = np.asarray(thetas.loc)
        glob = np.asarray(thetas.glob)

        b = glob[:, 0][:, None, None]
        g = glob[:, 1][:, None, None]
        k = glob[:, 2][:, None, None]

        x = loc + self._gk_transform(z, b, g, k)
        x.sort(axis=2)
        return x
