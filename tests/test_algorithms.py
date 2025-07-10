import jax
import numpy as np
from permabc import perm_abc_smc, KernelTruncatedRW
from permabc.models import GaussianWithNoSummaryStats

def test_perm_abc_smc_smoke_test():
    """
    Tests that the main perm_abc_smc algorithm runs without crashing on a simple model.
    """
    key = jax.random.PRNGKey(0)
    model = GaussianWithNoSummaryStats(K=2, n_obs=10)

    # Generate dummy observed data
    key, key_data = jax.random.split(key)
    true_thetas = model.prior_generator(key, n_particles=1)
    y_obs = model.data_generator(key_data, true_thetas)

    # Run the algorithm with minimal settings for speed
    results = perm_abc_smc(
        key=key,
        model=model,
        n_particles=10,        # Very few particles
        epsilon_target=50.0,
        y_obs=y_obs,
        kernel=KernelTruncatedRW,
        Final_iteration=0,     # Stop as soon as epsilon is reached
        verbose=0              # No console output during test
    )

    # Assert that the output is a dictionary and contains essential keys
    assert isinstance(results, dict)
    assert "Thetas" in results
    assert "Weights" in results
    assert "Eps_values" in results
    # Check that the final Thetas object has the correct number of particles
    assert results['Thetas'][-1].loc.shape[0] == 10