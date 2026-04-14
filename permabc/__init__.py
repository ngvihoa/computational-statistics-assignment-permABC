"""
PermABC: Permutation-enhanced Approximate Bayesian Computation

A Python package for ABC methods with permutation-invariant inference.
"""

__version__ = "0.1.0"
__author__ = "Antoine Luciano"
__email__ = "luciano@ceremade.dauphine.fr"

# Import main algorithms.
# Some algorithms depend on optional compiled deps (e.g. numba), which may fail
# under certain numpy/jax environments. Keep those imports optional.
abc_smc = None
perm_abc_smc = None
abc_pmc = None
perm_abc_smc_os = None
perm_abc_smc_um = None
abc_vanilla = None
perm_abc_vanilla = None

try:
    from .algorithms.smc import perm_abc_smc, abc_smc, resolve_assignment_bools
except Exception:  # pragma: no cover
    pass

try:
    from .algorithms.pmc import abc_pmc
except Exception:  # pragma: no cover
    abc_pmc = None

try:
    from .algorithms.over_sampling import perm_abc_smc_os
except Exception:  # pragma: no cover
    perm_abc_smc_os = None

try:
    from .algorithms.under_matching import perm_abc_smc_um
except Exception:  # pragma: no cover
    perm_abc_smc_um = None

try:
    from .algorithms.vanilla import abc_vanilla, perm_abc_vanilla
except Exception:  # pragma: no cover
    pass

# Import core components
try:
    from .assignment import optimal_index_distance
    from .sampling import KernelTruncatedRW, KernelRW, move_smc
except ImportError:
    # Handle case where core modules are not yet available
    pass

# Import utilities
try:
    from .utils.functions import Theta, ess, resampling
except ImportError:
    # Handle case where utils are not yet available
    pass

# Define what gets imported with "from permabc import *"
__all__ = [
    # Algorithms
    'perm_abc_smc',
    'abc_smc',
    'abc_pmc',
    'perm_abc_smc_os',
    'perm_abc_smc_um',
    'abc_vanilla',
    'perm_abc_vanilla',
    
    # Core functions
    'optimal_index_distance',
    'KernelTruncatedRW',
    'KernelRW',
    'move_smc',
    
    # Utilities
    'Theta',
    'ess',
    'resampling',
]

def get_version():
    """Return the version string."""
    return __version__

def get_info():
    """Return package information."""
    return {
        'name': 'permabc',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': 'Permutation-enhanced Approximate Bayesian Computation'
    }