"""
Tests for permABC models using pytest conventions.
"""

import pytest
from permabc.models import GaussianWithNoSummaryStats, Uniform_known
from jax import random
import jax.numpy as jnp

# --- Fixtures pour réutiliser le modèle dans plusieurs tests ---

@pytest.fixture
def gaussian_model():
    """Provides a default GaussianWithNoSummaryStats model instance."""
    return GaussianWithNoSummaryStats(K=3, n_obs=5)

@pytest.fixture
def uniform_model():
    """Provides a default Uniform_known model instance."""
    return Uniform_known(K=2, n_obs=3)

# --- Tests ---

def test_model_creation(gaussian_model, uniform_model):
    """Tests that model instances are created correctly."""
    assert gaussian_model.K == 3
    assert gaussian_model.n_obs == 5
    assert uniform_model.K == 2
    assert uniform_model.n_obs == 3

def test_prior_generation(gaussian_model):
    """Tests prior sample generation produces outputs with correct shapes."""
    key = random.PRNGKey(42)
    n_particles = 10
    
    thetas = gaussian_model.prior_generator(key, n_particles=n_particles)
    
    assert thetas.loc.shape == (n_particles, gaussian_model.K, 1)
    assert thetas.glob.shape == (n_particles, 1)
    
    log_pdf = gaussian_model.prior_logpdf(thetas)
    assert log_pdf.shape == (n_particles,)

def test_data_generation(gaussian_model):
    """Tests data generation and distance computation."""
    key = random.PRNGKey(42)
    n_particles = 10

    key, subkey = random.split(key)
    thetas = gaussian_model.prior_generator(subkey, n_particles=n_particles)
    
    key, subkey = random.split(key)
    data = gaussian_model.data_generator(subkey, thetas)
    
    assert data.shape == (n_particles, gaussian_model.K, gaussian_model.n_obs)
    
    # Test that sorting is applied
    assert jnp.all(data[:, :, :-1] <= data[:, :, 1:])

    # Test distance computation
    key, subkey = random.split(key)
    y_obs = gaussian_model.data_generator(subkey, gaussian_model.prior_generator(key, 1))
    distances = gaussian_model.distance(data, y_obs)
    assert distances.shape == (n_particles,)