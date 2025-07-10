"""
Statistical models for permutation-enhanced ABC.

This module provides a collection of statistical models that are compatible
with permutation-enhanced ABC algorithms. Each model implements the standard
interface defined by ModelBase.

Available Models
---------------
- ModelBase: Abstract base class for all models
- GaussianWithNoSummaryStats: Gaussian mixture with raw observations
- GaussianWithCorrelatedParams: Hierarchical Gaussian with parameter correlation
- Uniform_known: Simple uniform model for demonstrating permutation effects
- SIRWithKnownInit: SIR epidemic model with known initial conditions
- SIRWithUnknownInit: SIR epidemic model with unknown initial conditions

Examples
--------
>>> from permabc.models import GaussianWithNoSummaryStats
>>> model = GaussianWithNoSummaryStats(K=3, n_obs=10)
>>> 
>>> # Generate prior samples
>>> key = jax.random.PRNGKey(42)
>>> thetas = model.prior_generator(key, n_particles=100)
>>> 
>>> # Simulate data
>>> key, subkey = jax.random.split(key)
>>> data = model.data_generator(subkey, thetas)
"""

import jax.numpy as jnp
import numpy as np
from jax import vmap, jit
from functools import partial


# ==============================================================================
# BASE MODEL CLASS AND UTILITIES
# ==============================================================================

@partial(jit, static_argnums=(3, 4))  # L and K are static arguments
def _compute_distance_matrices_jit_static(zs_slice, y_obs_slice, weights_slice, L, K):
    """
    JIT-compiled distance matrix computation with static L and K.
    
    This function computes distance matrices efficiently using JAX compilation
    with L and K as static arguments for optimal performance.
    
    Parameters
    ----------
    zs_slice : jnp.ndarray
        Sliced simulated data.
    y_obs_slice : jnp.ndarray
        Sliced observed data.
    weights_slice : jnp.ndarray
        Distance weights.
    L : int, static
        Number of components to match.
    K : int, static
        Total number of components.
        
    Returns
    -------
    jnp.ndarray
        Distance matrices for batch processing.
    """
    def distance_matrix_no_summary(z, y, w, L, K):
        """Compute pairwise distance matrix for a single particle."""
        M = z.shape[0]
        # Compute all pairwise distances
        dist_matrix = jnp.sum(((y[:, None, :] - z[None, :, :]) * w[:, None, None]) ** 2, axis=2)
        # Create padded matrix for assignment problem
        matrix = jnp.zeros((2 * K - L, K + M - L))
        matrix = matrix.at[:K, :M].set(dist_matrix)
        return matrix
    
    return vmap(distance_matrix_no_summary, in_axes=(0, None, None, None, None))(
        zs_slice, y_obs_slice, weights_slice, L, K
    )


class ModelBase:
    """
    Base class for all statistical models in permABC.
    
    This abstract base class defines the interface that all models must implement
    to be compatible with permutation-enhanced ABC algorithms. It provides common
    functionality for distance computation and weight management.
    
    Attributes
    ----------
    K : int
        Number of components in the model.
    weights_distance : np.ndarray
        Weights for distance computation between components.
    """
    
    def __init__(self, K, weights_distance=None):
        """
        Initialize the base model.
        
        Parameters
        ----------
        K : int
            Number of components/groups in the model.
        weights_distance : array-like, optional
            Weights for distance computation. If None, uniform weights are used.
        """
        self.K = K
        if weights_distance is None:
            self.weights_distance = np.ones(K) / K
        else:
            weights = np.array(weights_distance)
            self.weights_distance = weights / np.sum(weights)

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate samples from the prior distribution.
        
        This method must be implemented by subclasses to define how to sample
        from the model's prior distribution.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        n_particles : int
            Number of particles to generate.
        n_silos : int, default=0
            Number of components (if 0, defaults to K).
            
        Returns
        -------
        Theta
            Parameter samples from the prior.
            
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Each model must implement its own prior generator.")

    def prior_logpdf(self, thetas):
        """
        Compute log probability density of the prior distribution.
        
        Parameters
        ----------
        thetas : Theta
            Parameter values to evaluate.
            
        Returns
        -------
        np.ndarray
            Log probability densities.
            
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Each model must implement its own prior logpdf.")

    def data_generator(self, key, thetas):
        """
        Generate simulated data given model parameters.
        
        This method simulates data from the model given parameter values.
        The output should be structured to match the observed data format.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        thetas : Theta
            Parameter values for simulation.
            
        Returns
        -------
        np.ndarray
            Simulated data with shape (n_particles, K, n_obs).
            
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Each model must implement its own data generator.")

    def distance(self, zs, y_obs):
        """
        Compute distance between simulated and observed data.
        
        This is the primary distance metric used in ABC to compare simulated
        data with observations. Uses weighted Euclidean distance by default.
        
        Parameters
        ----------
        zs : np.ndarray
            Simulated data of shape (n_particles, K, n_obs).
        y_obs : np.ndarray
            Observed data of shape (1, K, n_obs).
            
        Returns
        -------
        np.ndarray
            Distance values for each particle.
        """
        if zs.shape[0] == 1: 
            # Single particle case - return scalar
            diff = y_obs[0] - zs
            weighted_diff = diff * self.weights_distance[None, :, None]
            return jnp.sqrt(jnp.sum(weighted_diff ** 2, axis=(1, 2))).astype(np.float32)
        
        # Multiple particles case
        diff = y_obs[0] - zs
        weighted_diff = diff * self.weights_distance[None, :, None]
        return np.array(jnp.sqrt(jnp.sum(weighted_diff ** 2, axis=(1, 2))))

    def distance_component(self, z_k, y_k):
        """
        Compute distance between single components.
        
        Helper method for computing distances between individual components,
        used in permutation assignment algorithms.
        
        Parameters
        ----------
        z_k : np.ndarray
            Simulated data for component k.
        y_k : np.ndarray
            Observed data for component k.
            
        Returns
        -------
        float
            Distance between the components.
        """
        diff = y_k - z_k
        return np.sum(diff ** 2)

    def distance_global(self, zs, y_obs):
        """
        Compute global distance for parameters beyond K components.
        
        This method handles cases where models have global parameters
        or components beyond the main K components.
        
        Parameters
        ----------
        zs : np.ndarray
            Simulated data.
        y_obs : np.ndarray
            Observed data.
            
        Returns
        -------
        np.ndarray
            Global distance values.
        """
        if y_obs.shape[1] > self.K:
            # Handle extra global components
            global_weights = self.weights_distance[self.K:]
            global_diff = y_obs[0, self.K:] - zs[:, self.K:]
            return jnp.sum((global_weights * global_diff) ** 2, axis=1)
        return np.zeros(zs.shape[0])

    def distance_matrices_loc(self, zs, y_obs, M=0, L=0):
        """
        Compute local pairwise distance matrices efficiently.
        
        This method computes distance matrices between all pairs of simulated
        and observed components, optimized with JAX compilation.
        
        Parameters
        ----------
        zs : np.ndarray
            Simulated data.
        y_obs : np.ndarray
            Observed data.
        M : int, default=0
            Number of simulated components (defaults to K).
        L : int, default=0
            Number of components to match (defaults to K).
            
        Returns
        -------
        np.ndarray
            Distance matrices for assignment problems.
        """
        if M == 0: 
            M = self.K
        if L == 0: 
            L = self.K
        
        # Slice data before JIT compilation for efficiency
        zs_slice = zs[:, :M]
        y_obs_slice = y_obs[0, :self.K]
        weights_slice = self.weights_distance[:self.K]
        
        return _compute_distance_matrices_jit_static(
            zs_slice, y_obs_slice, weights_slice, L, self.K
        )

    def update_weights_distance(self, zs, verbose=0):
        """
        Update distance weights adaptively based on data variability.
        
        This method dynamically adjusts the weights used in distance computation
        based on the median absolute deviation (MAD) of each component.
        Components with higher variability get lower weights.
        
        Parameters
        ----------
        zs : np.ndarray or list
            Simulated data or list of component data.
        verbose : int, default=0
            Verbosity level for output.
        """
        def mad(x):
            """Compute median absolute deviation."""
            return jnp.median(jnp.abs(x - jnp.median(x)))
        
        if isinstance(zs, list):
            # Handle list format (from under-matching)
            mad_values = np.array([mad(np.array(z)) for z in zs])
        else:
            # Handle array format (standard case)
            # Reshape: (K, n_particles * n_obs)
            new_zs = zs.swapaxes(0, 1).reshape(zs.shape[1], -1)
            mad_values = vmap(mad, in_axes=0)(new_zs)
              
        # Compute inverse weights (higher variability → lower weight)
        weights = 1 / (mad_values + 1e-8)  # Add small epsilon for numerical stability
        self.weights_distance = weights / jnp.sum(weights)
        
        if verbose > 1: 
            print(f"Updated distance weights: min = {self.weights_distance.min():.3f}, "
                  f"max = {self.weights_distance.max():.3f}")
        
    def reset_weights_distance(self):
        """
        Reset distance weights to uniform distribution.
        
        This method resets all component weights to be equal, effectively
        giving equal importance to all components in distance computation.
        """
        self.weights_distance = jnp.ones(self.weights_distance.shape) / self.weights_distance.shape[0]


# Backward compatibility alias
model = ModelBase


# ==============================================================================
# MODEL IMPORTS AND MANAGEMENT
# ==============================================================================

# Import specific models with error handling
_available_models = {}

try:
    from .Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
    _available_models['GaussianWithNoSummaryStats'] = GaussianWithNoSummaryStats
except ImportError:
    pass

try:
    from .Gaussian_with_correlated_params import GaussianWithCorrelatedParams
    _available_models['GaussianWithCorrelatedParams'] = GaussianWithCorrelatedParams
except ImportError:
    pass

try:
    from .uniform_known import Uniform_known
    _available_models['Uniform_known'] = Uniform_known
except ImportError:
    pass

try:
    from .SIR import SIRWithKnownInit, SIRWithUnknownInit, simulate_sir_jax
    _available_models['SIRWithKnownInit'] = SIRWithKnownInit
    _available_models['SIRWithUnknownInit'] = SIRWithUnknownInit
except ImportError:
    pass

# Additional models that might be available
try:
    from .g_and_k import GAndKModel
    _available_models['GAndKModel'] = GAndKModel
except ImportError:
    pass

try:
    from .uniform import UniformModel
    _available_models['UniformModel'] = UniformModel
except ImportError:
    pass


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def list_available_models():
    """
    List all available models in this module.
    
    Returns
    -------
    dict
        Dictionary mapping model names to their classes and descriptions.
    """
    model_descriptions = {
        'ModelBase': "Abstract base class for all models",
        'GaussianWithNoSummaryStats': "Gaussian mixture with component-specific means",
        'GaussianWithCorrelatedParams': "Hierarchical Gaussian with correlated parameters",
        'Uniform_known': "Simple uniform model for demonstrating permutation effects",
        'SIRWithKnownInit': "SIR epidemic model with known initial conditions",
        'SIRWithUnknownInit': "SIR epidemic model with unknown initial conditions",
        'GAndKModel': "Generalized g-and-k distribution model",
        'UniformModel': "General uniform distribution model",
    }
    
    models = {'ModelBase': {'class': ModelBase, 'description': model_descriptions['ModelBase']}}
    
    for name, cls in _available_models.items():
        models[name] = {
            'class': cls,
            'description': model_descriptions.get(name, "No description available"),
            'module': cls.__module__ if hasattr(cls, '__module__') else 'unknown'
        }
    
    return models


def get_model_by_name(name):
    """
    Get a model class by name.
    
    Parameters
    ----------
    name : str
        Name of the model class.
        
    Returns
    -------
    class or None
        Model class if found, None otherwise.
        
    Examples
    --------
    >>> ModelClass = get_model_by_name('GaussianWithNoSummaryStats')
    >>> if ModelClass:
    ...     model = ModelClass(K=3)
    """
    if name == 'ModelBase':
        return ModelBase
    return _available_models.get(name)


def create_model(model_name, **kwargs):
    """
    Create a model instance by name with given parameters.
    
    Parameters
    ----------
    model_name : str
        Name of the model class.
    **kwargs
        Parameters to pass to the model constructor.
        
    Returns
    -------
    ModelBase
        Initialized model instance.
        
    Raises
    ------
    ValueError
        If model name is not found.
        
    Examples
    --------
    >>> model = create_model('GaussianWithNoSummaryStats', K=3, n_obs=10)
    >>> model = create_model('SIRWithKnownInit', K=5, n_obs=100, n_pop=10000)
    """
    ModelClass = get_model_by_name(model_name)
    if ModelClass is None:
        available = list(list_available_models().keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return ModelClass(**kwargs)


# Model categories for organization
GAUSSIAN_MODELS = ['GaussianWithNoSummaryStats', 'GaussianWithCorrelatedParams']
EPIDEMIC_MODELS = ['SIRWithKnownInit', 'SIRWithUnknownInit']
SIMPLE_MODELS = ['Uniform_known', 'UniformModel']

def get_models_by_category(category):
    """
    Get models belonging to a specific category.
    
    Parameters
    ----------
    category : str
        Category name ('gaussian', 'epidemic', 'simple', or 'all').
        
    Returns
    -------
    dict
        Dictionary of model names to classes in the category.
    """
    if category.lower() == 'gaussian':
        model_names = GAUSSIAN_MODELS
    elif category.lower() == 'epidemic':
        model_names = EPIDEMIC_MODELS
    elif category.lower() == 'simple':
        model_names = SIMPLE_MODELS
    elif category.lower() == 'all':
        model_names = GAUSSIAN_MODELS + EPIDEMIC_MODELS + SIMPLE_MODELS
    else:
        raise ValueError(f"Unknown category '{category}'. "
                        f"Available: 'gaussian', 'epidemic', 'simple', 'all'")
    
    models = {}
    for name in model_names:
        cls = get_model_by_name(name)
        if cls is not None:
            models[name] = cls
    
    return models


# ==============================================================================
# EXPORTS
# ==============================================================================

# Define what gets imported with "from permabc.models import *"
__all__ = [
    # Base classes
    'ModelBase',
    'model',  # Backward compatibility
    
    # Utility functions
    '_compute_distance_matrices_jit_static',
    'list_available_models',
    'get_model_by_name', 
    'create_model',
    'get_models_by_category',
]

# Add available models to exports
__all__.extend(_available_models.keys())

# Add model classes to global namespace for direct import
globals().update(_available_models)

# Backward compatibility aliases
try:
    if 'GaussianWithNoSummaryStats' in _available_models:
        GaussianNoSummary = GaussianWithNoSummaryStats
    if 'GaussianWithCorrelatedParams' in _available_models:
        GaussianCorrelated = GaussianWithCorrelatedParams
    if 'Uniform_known' in _available_models:
        UniformKnown = Uniform_known
    if 'SIRWithKnownInit' in _available_models:
        SIRKnown = SIRWithKnownInit
    if 'SIRWithUnknownInit' in _available_models:
        SIRUnknown = SIRWithUnknownInit
except:
    pass