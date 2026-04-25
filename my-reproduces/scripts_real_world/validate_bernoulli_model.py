#!/usr/bin/env python3
"""
Quick validation script for BernoulliLogitWithCovariates model.

Tests:
1. Prior sampling generates correct shapes
2. Data generation produces binary outputs
3. Distance computation works without NaNs
"""

import sys
from pathlib import Path
import numpy as np
from jax import random

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from permabc.models.bernoulli_logit_with_covariates import BernoulliLogitWithCovariates


def test_model_creation():
    """Test that model can be created with valid parameters."""
    print("Test 1: Model creation...")
    model = BernoulliLogitWithCovariates(
        K=3,
        n_obs=10,
        n_features=5,
        mu_alpha=0.0,
        sigma_alpha=2.0,
        mu_beta=0.0,
        sigma_beta=2.0,
    )
    print(f"  ✓ Model created: K={model.K}, n_obs={model.n_obs}, n_features={model.n_features}")
    return model


def test_prior_sampling(model, n_particles=50):
    """Test that prior sampling returns correct shapes."""
    print("\nTest 2: Prior sampling...")
    key = random.PRNGKey(42)
    thetas = model.prior_generator(key, n_particles=n_particles)
    
    print(f"  Theta.loc shape: {thetas.loc.shape} (expected: ({n_particles}, {model.K}, 1))")
    print(f"  Theta.glob shape: {thetas.glob.shape} (expected: ({n_particles}, {model.n_features}))")
    
    assert thetas.loc.shape == (n_particles, model.K, 1), f"Wrong loc shape: {thetas.loc.shape}"
    assert thetas.glob.shape == (n_particles, model.n_features), f"Wrong glob shape: {thetas.glob.shape}"
    print(f"  ✓ Prior sampling OK")
    
    return thetas


def test_prior_logpdf(model, thetas):
    """Test that prior logpdf computation works."""
    print("\nTest 3: Prior log-pdf...")
    log_pdf_vals = model.prior_logpdf(thetas)
    print(f"  Log-pdf shape: {log_pdf_vals.shape} (expected: ({thetas.loc.shape[0]},))")
    print(f"  Log-pdf range: [{log_pdf_vals.min():.3f}, {log_pdf_vals.max():.3f}]")
    print(f"  Finite values: {np.isfinite(log_pdf_vals).sum()}/{len(log_pdf_vals)}")
    
    assert np.isfinite(log_pdf_vals).all(), "Some log-pdf values are not finite"
    print(f"  ✓ Prior log-pdf OK")


def test_data_generation(model, thetas):
    """Test that data generation produces binary outputs with correct shape."""
    print("\nTest 4: Data generation...")
    key = random.PRNGKey(123)
    
    zs = model.data_generator(key, thetas)
    print(f"  Simulated data shape: {zs.shape} (expected: ({thetas.loc.shape[0]}, {model.K}, {model.n_obs}))")
    print(f"  Data type: {zs.dtype}")
    print(f"  Unique values in data: {np.unique(zs)}")
    print(f"  Mean (should be ~0.5): {zs.mean():.3f}")
    
    assert zs.shape == (thetas.loc.shape[0], model.K, model.n_obs), f"Wrong data shape: {zs.shape}"
    assert set(np.unique(zs)) <= {0, 1}, "Data should be binary (0 or 1)"
    print(f"  ✓ Data generation OK")
    
    return zs


def test_distance_computation(model, zs):
    """Test that distance computation works."""
    print("\nTest 5: Distance computation...")
    n_particles = zs.shape[0]
    
    # Create synthetic observation data
    y_obs = np.zeros((1, model.K, model.n_obs), dtype=np.float32)
    y_obs[0, 0, :5] = 1  # First region has rain in first 5 days
    
    distances = model.distance(zs, y_obs)
    print(f"  Distance shape: {distances.shape} (expected: ({n_particles},))")
    print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"  Finite distances: {np.isfinite(distances).sum()}/{len(distances)}")
    
    assert distances.shape == (n_particles,), f"Wrong distance shape: {distances.shape}"
    assert np.isfinite(distances).all(), "Some distances are not finite"
    print(f"  ✓ Distance computation OK")


def test_distance_matrices(model, zs):
    """Test that distance matrices computation works."""
    print("\nTest 6: Distance matrices (for assignment)...")
    n_particles = zs.shape[0]
    
    # Create synthetic observation data
    y_obs = np.zeros((1, model.K, model.n_obs), dtype=np.float32)
    y_obs[0, 0, :5] = 1
    
    dist_matrices = model.distance_matrices_loc(zs, y_obs, M=model.K, L=model.K)
    print(f"  Distance matrices shape: {dist_matrices.shape}")
    print(f"  Expected shape: ({n_particles}, {model.K}, {model.K})")
    print(f"  Distance range: [{dist_matrices.min():.1f}, {dist_matrices.max():.1f}]")
    print(f"  Finite values: {np.isfinite(dist_matrices).sum()}/{dist_matrices.size}")
    
    assert dist_matrices.shape == (n_particles, model.K, model.K), f"Wrong matrix shape: {dist_matrices.shape}"
    assert np.isfinite(dist_matrices).all(), "Some distances are not finite"
    print(f"  ✓ Distance matrices OK")


def test_set_covariates(model):
    """Test that covariates can be set and used."""
    print("\nTest 7: Setting covariates...")
    X_cov = np.random.randn(model.K, model.n_obs, model.n_features).astype(np.float32)
    model.set_X_cov(X_cov)
    print(f"  X_cov shape set to: {model.X_cov.shape}")
    print(f"  X_cov range: [{model.X_cov.min():.3f}, {model.X_cov.max():.3f}]")
    assert np.allclose(model.X_cov, X_cov), "Covariates not set correctly"
    print(f"  ✓ Covariate setting OK")


def main():
    """Run all tests."""
    print("=" * 70)
    print("BernoulliLogitWithCovariates Model Validation")
    print("=" * 70)
    
    try:
        model = test_model_creation()
        thetas = test_prior_sampling(model, n_particles=50)
        test_prior_logpdf(model, thetas)
        zs = test_data_generation(model, thetas)
        test_distance_computation(model, zs)
        test_distance_matrices(model, zs)
        test_set_covariates(model)
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
    except Exception as exc:
        print("\n" + "=" * 70)
        print(f"✗ Test failed with error:")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
