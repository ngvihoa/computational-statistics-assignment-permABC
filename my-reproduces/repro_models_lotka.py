"""Mô hình mô phỏng dữ liệu Lotka-Volterra."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp

from permabc.models import ModelBase
from permabc.utils.functions import Theta


class LotkaVolterraModelForReproduce(ModelBase):
    def __init__(
        self,
        K: int,
        n_obs: int = 40,
        dt: float = 0.1,
        prey0_bounds: Tuple[float, float] = (8.0, 20.0),
        pred0: float = 6.0,
        alpha_bounds: Tuple[float, float] = (0.6, 1.4),
        beta_bounds: Tuple[float, float] = (0.03, 0.12),
        delta_bounds: Tuple[float, float] = (0.02, 0.10),
        gamma_bounds: Tuple[float, float] = (0.5, 1.2),
        obs_noise: float = 0.4,
    ):
        super().__init__(K)
        self.n_obs = n_obs
        self.dt = dt
        self.pred0 = pred0
        self.prey0_bounds = prey0_bounds
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.delta_bounds = delta_bounds
        self.gamma_bounds = gamma_bounds
        self.obs_noise = obs_noise

        self.support_par_loc = jnp.array([[prey0_bounds[0], prey0_bounds[1]]])
        self.support_par_glob = jnp.array(
            [
                [alpha_bounds[0], alpha_bounds[1]],
                [beta_bounds[0], beta_bounds[1]],
                [delta_bounds[0], delta_bounds[1]],
                [gamma_bounds[0], gamma_bounds[1]],
            ]
        )

        self.dim_loc = 1
        self.dim_glob = 4
        self.loc_name = ["$x_0{"]
        self.glob_name = ["$\\alpha$", "$\\beta$", "$\\delta$", "$\\gamma$"]

    @staticmethod
    def _simulate_single(prey0, pred0, alpha, beta, delta, gamma, n_obs, dt):
        prey = np.empty(n_obs, dtype=np.float64)
        pred = np.empty(n_obs, dtype=np.float64)
        x = max(prey0, 1e-3)
        y = max(pred0, 1e-3)
        for t in range(n_obs):
            prey[t] = x
            pred[t] = y
            dx = alpha * x - beta * x * y
            dy = delta * x * y - gamma * y
            x = max(x + dt * dx, 1e-3)
            y = max(y + dt * dy, 1e-3)
        return prey, pred

    def prior_generator(self, key, n_particles, n_silos=0):
        if n_silos == 0:
            n_silos = self.K
        rng = np.random.default_rng(int(key[0]))

        prey0 = rng.uniform(*self.prey0_bounds, size=(n_particles, n_silos, 1))
        alpha = rng.uniform(*self.alpha_bounds, size=(n_particles, 1))
        beta = rng.uniform(*self.beta_bounds, size=(n_particles, 1))
        delta = rng.uniform(*self.delta_bounds, size=(n_particles, 1))
        gamma = rng.uniform(*self.gamma_bounds, size=(n_particles, 1))
        glob = np.hstack([alpha, beta, delta, gamma])
        return Theta(loc=prey0, glob=glob)

    def prior_logpdf(self, thetas):
        loc = np.asarray(thetas.loc)
        glob = np.asarray(thetas.glob)

        in_loc = (loc[:, :, 0] >= self.prey0_bounds[0]) & (loc[:, :, 0] <= self.prey0_bounds[1])
        in_glob = (
            (glob[:, 0] >= self.alpha_bounds[0]) & (glob[:, 0] <= self.alpha_bounds[1])
            & (glob[:, 1] >= self.beta_bounds[0]) & (glob[:, 1] <= self.beta_bounds[1])
            & (glob[:, 2] >= self.delta_bounds[0]) & (glob[:, 2] <= self.delta_bounds[1])
            & (glob[:, 3] >= self.gamma_bounds[0]) & (glob[:, 3] <= self.gamma_bounds[1])
        )

        log_loc = -self.K * np.log(self.prey0_bounds[1] - self.prey0_bounds[0])
        log_glob = 0.0
        log_glob -= np.log(self.alpha_bounds[1] - self.alpha_bounds[0])
        log_glob -= np.log(self.beta_bounds[1] - self.beta_bounds[0])
        log_glob -= np.log(self.delta_bounds[1] - self.delta_bounds[0])
        log_glob -= np.log(self.gamma_bounds[1] - self.gamma_bounds[0])

        in_all = np.all(in_loc, axis=1) & in_glob
        return np.where(in_all, log_loc + log_glob, -np.inf)

    def data_generator(self, key, thetas):
        n_particles, n_silos = thetas.loc.shape[:2]
        loc = np.asarray(thetas.loc)[:, :, 0]
        glob = np.asarray(thetas.glob)

        out = np.empty((n_particles, n_silos, 2 * self.n_obs), dtype=np.float64)
        for i in range(n_particles):
            alpha, beta, delta, gamma = glob[i]
            for k in range(n_silos):
                prey, pred = self._simulate_single(
                    prey0=loc[i, k],
                    pred0=self.pred0,
                    alpha=alpha,
                    beta=beta,
                    delta=delta,
                    gamma=gamma,
                    n_obs=self.n_obs,
                    dt=self.dt,
                )
                out[i, k, : self.n_obs] = prey
                out[i, k, self.n_obs :] = pred

        rng = np.random.default_rng(int(key[0]) + 17)
        out += rng.normal(loc=0.0, scale=self.obs_noise, size=out.shape)
        return out
