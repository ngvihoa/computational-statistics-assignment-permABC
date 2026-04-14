#!/usr/bin/env python3
"""
Benchmark JAX vs Numba for SIR simulation.
Tests with N_particles=1000, K=94 departments, n_obs=120 days.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

jax.config.update("jax_enable_x64", True)

# Try numba
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available, skipping numba benchmark")

# ── Parameters ──────────────────────────────────────────────────────────────

N_PARTICLES = 1000
K = 94
N_OBS = 120
N_POP = 100000.0
SIGMA = 0.05
STEPS_PER_DAY = 10
DT = 1.0 / STEPS_PER_DAY

# ── JAX version (current code) ─────────────────────────────────────────────

@partial(jax.jit, static_argnames=['n_obs', 'steps_per_day', 'sigma'])
def simulate_sir_jax(S0, I0, R0, beta, gamma, n_pop, n_obs, dt=0.1,
                     noise_key=None, sigma=0.05, steps_per_day=10):
    n_particles, n_regions = S0.shape
    dtype = S0.dtype

    if noise_key is not None and sigma > 0:
        noise_shape = (n_particles, n_regions, n_obs, steps_per_day)
        noise = jnp.exp(sigma * random.normal(noise_key, noise_shape, dtype=dtype))
    else:
        noise = jnp.ones((n_particles, n_regions, n_obs, steps_per_day), dtype=dtype)

    def simulate_particle(particle_idx):
        def simulate_day(state, day_idx):
            S, I, R = state
            def substep(carry, step_idx):
                S_curr, I_curr, R_curr = carry
                S_safe = jnp.maximum(S_curr, 0.0)
                I_safe = jnp.maximum(I_curr, 0.0)
                inf_rate = beta[particle_idx] * S_safe * I_safe / n_pop * dt
                noise_factor = noise[particle_idx, :, day_idx, step_idx]
                new_infections = jnp.minimum(inf_rate * noise_factor, S_safe)
                recovery_rate = gamma[particle_idx] * I_safe * dt
                new_recoveries = jnp.minimum(recovery_rate, I_safe)
                I_new = jnp.maximum(0.0, I_safe + new_infections - new_recoveries)
                R_new = jnp.maximum(0.0, R_curr + new_recoveries)
                S_new = jnp.maximum(0.0, n_pop - I_new - R_new)
                return (S_new, I_new, R_new), I_new
            final_state, _ = jax.lax.scan(substep, (S, I, R), jnp.arange(steps_per_day))
            return final_state, final_state[1]
        initial_state = (S0[particle_idx], I0[particle_idx], R0[particle_idx])
        _, I_trajectory = jax.lax.scan(simulate_day, initial_state, jnp.arange(n_obs))
        return I_trajectory

    I_trajectories = jax.vmap(simulate_particle)(jnp.arange(n_particles))
    return jnp.transpose(I_trajectories, (0, 2, 1))


# ── Numba version ───────────────────────────────────────────────────────────

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def simulate_sir_numba(S0, I0, R0, beta, gamma, n_pop, n_obs, dt,
                           noise, steps_per_day):
        """
        S0, I0, R0: (N, K)
        beta, gamma: (N, K)
        noise: (N, K, n_obs, steps_per_day)
        Returns I_traj: (N, K, n_obs)
        """
        N, K_ = S0.shape
        I_traj = np.empty((N, K_, n_obs), dtype=np.float64)

        for p in prange(N):
            for k in range(K_):
                S = S0[p, k]
                I = I0[p, k]
                R = R0[p, k]
                for t in range(n_obs):
                    for s in range(steps_per_day):
                        S_safe = max(S, 0.0)
                        I_safe = max(I, 0.0)
                        inf_rate = beta[p, k] * S_safe * I_safe / n_pop * dt
                        new_inf = min(inf_rate * noise[p, k, t, s], S_safe)
                        rec_rate = gamma[p, k] * I_safe * dt
                        new_rec = min(rec_rate, I_safe)
                        I = max(0.0, I_safe + new_inf - new_rec)
                        R = max(0.0, R + new_rec)
                        S = max(0.0, n_pop - I - R)
                    I_traj[p, k, t] = I
        return I_traj


# ── Generate random parameters ──────────────────────────────────────────────

def make_params(key):
    k1, k2, k3, k4, k5 = random.split(key, 5)
    I0 = random.uniform(k1, (N_PARTICLES, K), minval=0.1, maxval=100.0)
    R0_init = random.uniform(k2, (N_PARTICLES, K), minval=0.1, maxval=50.0)
    S0 = N_POP - I0 - R0_init
    gamma = random.uniform(k3, (N_PARTICLES, K), minval=0.05, maxval=0.5)
    r0_glob = random.uniform(k4, (N_PARTICLES, 1), minval=0.5, maxval=4.0)
    beta = r0_glob * gamma
    noise_shape = (N_PARTICLES, K, N_OBS, STEPS_PER_DAY)
    noise = np.array(jnp.exp(SIGMA * random.normal(k5, noise_shape)))
    return S0, I0, R0_init, beta, gamma, noise


# ── Benchmark ────────────────────────────────────────────────────────────────

def main():
    print(f"Benchmark SIR: N={N_PARTICLES}, K={K}, T={N_OBS}, steps/day={STEPS_PER_DAY}")
    print(f"Total substeps per particle: {N_OBS * STEPS_PER_DAY * K:,}")
    print()

    key = random.PRNGKey(0)
    S0, I0, R0_init, beta, gamma, noise_np = make_params(key)

    # ── JAX ──
    print("JAX (jit + vmap):")
    # Warmup (compilation)
    key_noise = random.PRNGKey(1)
    t0 = time.perf_counter()
    out_jax = simulate_sir_jax(S0, I0, R0_init, beta, gamma,
                                n_pop=N_POP, n_obs=N_OBS, dt=DT,
                                noise_key=key_noise, sigma=SIGMA,
                                steps_per_day=STEPS_PER_DAY)
    out_jax.block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"  Compile + first run: {t_compile:.2f}s")

    # Timed runs
    times_jax = []
    for i in range(5):
        key_noise = random.PRNGKey(10 + i)
        t0 = time.perf_counter()
        out = simulate_sir_jax(S0, I0, R0_init, beta, gamma,
                                n_pop=N_POP, n_obs=N_OBS, dt=DT,
                                noise_key=key_noise, sigma=SIGMA,
                                steps_per_day=STEPS_PER_DAY)
        out.block_until_ready()
        times_jax.append(time.perf_counter() - t0)
    print(f"  After compilation: {np.mean(times_jax):.3f}s ± {np.std(times_jax):.3f}s (5 runs)")
    print(f"  Output shape: {out_jax.shape}")

    # ── Numba ──
    if HAS_NUMBA:
        S0_np = np.asarray(S0)
        I0_np = np.asarray(I0)
        R0_np = np.asarray(R0_init)
        beta_np = np.asarray(beta)
        gamma_np = np.asarray(gamma)

        print(f"\nNumba (parallel prange, {numba.config.NUMBA_NUM_THREADS} threads):")
        # Warmup
        t0 = time.perf_counter()
        out_numba = simulate_sir_numba(S0_np, I0_np, R0_np, beta_np, gamma_np,
                                        N_POP, N_OBS, DT, noise_np, STEPS_PER_DAY)
        t_compile_nb = time.perf_counter() - t0
        print(f"  Compile + first run: {t_compile_nb:.2f}s")

        times_nb = []
        for i in range(5):
            # Fresh noise each time (like JAX)
            noise_i = np.array(jnp.exp(SIGMA * random.normal(random.PRNGKey(10 + i),
                                       (N_PARTICLES, K, N_OBS, STEPS_PER_DAY))))
            t0 = time.perf_counter()
            out = simulate_sir_numba(S0_np, I0_np, R0_np, beta_np, gamma_np,
                                      N_POP, N_OBS, DT, noise_i, STEPS_PER_DAY)
            times_nb.append(time.perf_counter() - t0)
        print(f"  After compilation: {np.mean(times_nb):.3f}s ± {np.std(times_nb):.3f}s (5 runs)")
        print(f"  Output shape: {out_numba.shape}")

        # ── Comparison ──
        diff = np.abs(np.asarray(out_jax) - out_numba)
        print(f"\n  Max abs diff (same noise, first run): {diff.max():.2e}")
        print(f"  Speedup Numba/JAX: {np.mean(times_jax) / np.mean(times_nb):.2f}x")
    else:
        print("\nNumba not available, install with: pip install numba")


if __name__ == "__main__":
    main()
