import jax
import jax.numpy as jnp
import numpy as np
from permabc import KernelTruncatedRW
from permabc.utils.functions import Theta
from collections import namedtuple

def test_kernel_truncated_rw_respects_bounds():
    """
    Tests that the Truncated Random Walk kernel's proposals stay within the defined bounds.
    """
    key = jax.random.PRNGKey(42)
    n_particles = 100
    
    # 1. Create a mock model with strict bounds
    bounds = jnp.array([[-1.0, 1.0]]) # All parameters must be between -1 and 1
    MockModel = namedtuple('Model', ['K', 'support_par_loc', 'support_par_glob'])
    mock_model = MockModel(
        K=1,
        support_par_loc=bounds,
        support_par_glob=bounds
    )

    # 2. Create initial particles (some are already at the bounds)
    initial_loc = jnp.linspace(-1.0, 1.0, n_particles).reshape(n_particles, 1, 1)
    initial_glob = jnp.linspace(-1.0, 1.0, n_particles).reshape(n_particles, 1)
    initial_thetas = Theta(loc=initial_loc, glob=initial_glob)
    weights = jnp.ones(n_particles) / n_particles

    # 3. Initialize the kernel
    # Use a large variance to force proposals to go outside the bounds if not truncated
    tau_loc = jnp.full_like(initial_loc, 2.0)
    tau_glob = jnp.full_like(initial_glob, 2.0)
    
    kernel = KernelTruncatedRW(
        model=mock_model,
        thetas=initial_thetas,
        weights=weights,
        ys_index=None,
        zs_index=None,
        tau_loc_glob=(tau_loc[0], tau_glob[0]) # Provide pre-computed variance
    )

    # 4. Generate new samples
    proposed_thetas = kernel.sample(key)

    # 5. Assert that all proposed parameters are within the [-1, 1] bounds
    assert np.all(proposed_thetas.loc >= -1.0)
    assert np.all(proposed_thetas.loc <= 1.0)
    assert np.all(proposed_thetas.glob >= -1.0)
    assert np.all(proposed_thetas.glob <= 1.0)