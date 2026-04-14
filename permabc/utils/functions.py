from dataclasses import dataclass
import jax.numpy as jnp
from jax import random
from jax import tree_util
try:
    import particles  # optional; may fail due to numba/numpy constraints
except Exception:  # pragma: no cover
    particles = None
import numpy as np

import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass
import copy

@tree_util.register_pytree_node_class
@dataclass
class Theta:
    loc: jnp.ndarray  # shape (n_particles, K, dim_loc)
    glob: jnp.ndarray  # shape (n_particles, dim_glob)


    def __init__(self, loc=None, glob=None):
        """
        Initialize Theta with local and global parameters.
        
        Accepts both NumPy and JAX arrays. Stores whatever type is passed
        to avoid unnecessary conversions in performance-critical loops.
        """
        if loc is None:
            loc = np.empty((0, 0, 0))
        if glob is None:
            glob = np.empty((0, 0))
        self.loc = np.asarray(loc) if isinstance(loc, np.ndarray) else jnp.asarray(loc)
        self.glob = np.asarray(glob) if isinstance(glob, np.ndarray) else jnp.asarray(glob)
    
    def __post_init__(self):
        pass

    def tree_flatten(self):
        children = (self.loc, self.glob)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def apply_permutation(self, perm):
        loc_np = np.asarray(self.loc)
        idx = np.arange(loc_np.shape[0])[:, None]
        new_loc = loc_np[idx, np.asarray(perm)]
        return Theta(loc=new_loc, glob=np.asarray(self.glob))

    def __getitem__(self, index):
        return Theta(loc=np.asarray(self.loc)[index], glob=np.asarray(self.glob)[index])
    
            
    def append(self, value):
        """
        Append a new Theta object to the current one and return a new Theta.
        
        Parameters
        ----------
        value : Theta
            The Theta object to append.
            
        Returns
        -------
        Theta
            A new Theta object with combined loc and glob parameters.
        """
        
        # Check if self is empty (newly initialized)
        self_is_empty = (self.loc.size == 0 or self.glob.size == 0)
        value_is_empty = (value.loc.size == 0 or value.glob.size == 0)
        
        if self_is_empty and value_is_empty:
            return Theta(loc=np.empty((0, self.loc.shape[1], self.loc.shape[2])),
                        glob=np.empty((0, self.glob.shape[1])))
        
        elif self_is_empty:
            return value.copy()
        
        elif value_is_empty:
            return self.copy()
        
        else:
            new_loc = np.concatenate([np.asarray(self.loc), np.asarray(value.loc)], axis=0)
            new_glob = np.concatenate([np.asarray(self.glob), np.asarray(value.glob)], axis=0)
            return Theta(loc=new_loc, glob=new_glob)
        
    def __setitem__(self, key, value):
        loc = self._ensure_numpy(self.loc)
        glob = self._ensure_numpy(self.glob)
        val_loc = np.asarray(value.loc)
        val_glob = np.asarray(value.glob)
        
        if isinstance(key, (jnp.ndarray, np.ndarray)) and key.dtype == bool:
            loc[key] = val_loc
            glob[key] = val_glob
        else:
            loc[key] = val_loc
            glob[key] = val_glob
        self.loc = loc
        self.glob = glob
    
    @staticmethod
    def _ensure_numpy(arr):
        """Convert to a writable numpy array if needed."""
        out = np.asarray(arr)
        if not out.flags.writeable:
            out = out.copy()
        return out
            
    def reshape_2d(self):
        n_particles = self.loc.shape[0]
        flat_loc = np.asarray(self.loc).reshape(n_particles, -1)
        return np.concatenate([flat_loc, np.asarray(self.glob)], axis=1)

    def shape(self):
        return self.loc.shape, self.glob.shape

    def __len__(self):
        return self.loc.shape[0]

    def truncating(self, new_M, old_M):
        loc_np = np.asarray(self.loc)
        loc_trunc = np.concatenate([loc_np[:, :new_M], loc_np[:, old_M:]], axis=1)
        return Theta(loc=loc_trunc, glob=np.asarray(self.glob))

    def duplicate(self, n_duplicate, perm_duplicated):
        new_loc = np.repeat(np.asarray(self.loc), repeats=n_duplicate, axis=0)
        new_glob = np.repeat(np.asarray(self.glob), repeats=n_duplicate, axis=0)
        duplicated = Theta(loc=new_loc, glob=new_glob)
        return duplicated.apply_permutation(perm_duplicated)

    def __eq__(self, other):
        return np.all(np.asarray(self.loc) == np.asarray(other.loc)) and np.all(np.asarray(self.glob) == np.asarray(other.glob))

    def __ne__(self, other):
        return not self.__eq__(other)
    
    # ============================================================================
    # MÉTHODES AJOUTÉES POUR COMPATIBILITÉ
    # ============================================================================
    
    def copy(self):
        """
        Crée une copie profonde de l'objet Theta.
        
        Returns:
            Theta: Nouvelle instance avec copies des arrays
        """
        return Theta(
            loc=np.array(self.loc, copy=True),
            glob=np.array(self.glob, copy=True)
        )
    
    def __copy__(self):
        """Support pour copy.copy()"""
        return self.copy()
    
    def __deepcopy__(self, memo):
        """Support pour copy.deepcopy()"""
        return Theta(
            loc=np.array(self.loc, copy=True),
            glob=np.array(self.glob, copy=True)
        )
    
    # Méthodes utilitaires supplémentaires qui peuvent être utiles
    
    def clone(self):
        """Alias pour copy() - plus explicite"""
        return self.copy()
    
    def detach(self):
        """
        Détache les gradients (utile pour l'optimisation).
        En JAX, les arrays n'ont pas de gradients attachés comme PyTorch,
        donc on retourne juste une copie.
        """
        return self.copy()
    
    def to_dict(self):
        """Convertit en dictionnaire (utile pour sauvegarder/charger)"""
        return {
            'loc': self.loc,
            'glob': self.glob
        }
    
    @classmethod
    def from_dict(cls, data_dict):
        """Crée un Theta depuis un dictionnaire"""
        return cls(
            loc=jnp.asarray(data_dict['loc']),
            glob=jnp.asarray(data_dict['glob'])
        )
    
    def numpy(self):
        """Convertit les arrays JAX en numpy (utile pour sauvegarder)"""
        import numpy as np
        return Theta(
            loc=np.array(self.loc),
            glob=np.array(self.glob)
        )
    
    def device_put(self, device=None):
        """Place les arrays sur un device spécifique (GPU/CPU)"""
        from jax import device_put
        return Theta(
            loc=device_put(self.loc, device),
            glob=device_put(self.glob, device)
        )
    
    def summary(self):
        """Affiche un résumé des dimensions et statistiques"""
        print(f"Theta object:")
        print(f"  loc shape: {self.loc.shape}")
        print(f"  glob shape: {self.glob.shape}")
        print(f"  loc range: [{jnp.min(self.loc):.3f}, {jnp.max(self.loc):.3f}]")
        print(f"  glob range: [{jnp.min(self.glob):.3f}, {jnp.max(self.glob):.3f}]")
        print(f"  total parameters: {self.loc.size + self.glob.size}")


# ============================================================================
# FONCTIONS UTILITAIRES POUR THETA
# ============================================================================

def concatenate_thetas(theta_list):
    """
    Concatène une liste d'objets Theta le long de la dimension des particules.
    
    Args:
        theta_list: Liste d'objets Theta
        
    Returns:
        Theta: Objet Theta concaténé
    """
    if not theta_list:
        raise ValueError("Liste vide fournie")
    
    if len(theta_list) == 1:
        return theta_list[0].copy()
    
    locs = [theta.loc for theta in theta_list]
    globs = [theta.glob for theta in theta_list]
    
    return Theta(
        loc=jnp.concatenate(locs, axis=0),
        glob=jnp.concatenate(globs, axis=0)
    )

def stack_thetas(theta_list, axis=0):
    """
    Empile une liste d'objets Theta le long d'un axe spécifié.
    
    Args:
        theta_list: Liste d'objets Theta
        axis: Axe le long duquel empiler
        
    Returns:
        Theta: Objet Theta empilé
    """
    if not theta_list:
        raise ValueError("Liste vide fournie")
    
    locs = [theta.loc for theta in theta_list]
    globs = [theta.glob for theta in theta_list]
    
    return Theta(
        loc=jnp.stack(locs, axis=axis),
        glob=jnp.stack(globs, axis=axis)
    )

def theta_zeros_like(theta_template):
    """
    Crée un objet Theta rempli de zéros avec la même forme qu'un template.
    
    Args:
        theta_template: Objet Theta servant de template
        
    Returns:
        Theta: Objet Theta rempli de zéros
    """
    return Theta(
        loc=jnp.zeros_like(theta_template.loc),
        glob=jnp.zeros_like(theta_template.glob)
    )

def theta_ones_like(theta_template):
    """
    Crée un objet Theta rempli de uns avec la même forme qu'un template.
    
    Args:
        theta_template: Objet Theta servant de template
        
    Returns:
        Theta: Objet Theta rempli de uns
    """
    return Theta(
        loc=jnp.ones_like(theta_template.loc),
        glob=jnp.ones_like(theta_template.glob)
    )


# Fonction de resampling (⚠️ particles.resampling est probablement non JAX-friendly)
# Si tu veux une version 100% JAX, je peux t'en écrire une
def resampling(key, weight, L_to_resample, n_particles=0):
    """
    Systematic resampling.

    If optional dependency `particles` is available, use it.
    Otherwise, fall back to a NumPy/JAX implementation (no numba).
    """
    if n_particles == 0:
        n_particles = len(weight)

    if particles is not None:
        index = particles.resampling.systematic(weight, n_particles)
        return [x[index] for x in L_to_resample]

    # Fallback: systematic resampling using cumulative sums.
    w = np.asarray(weight, dtype=float).reshape(-1)
    w = np.clip(w, 0.0, np.inf)
    sw = np.sum(w)
    if sw <= 0:
        # Degenerate case: uniform
        w = np.ones_like(w) / len(w)
    else:
        w = w / sw

    cdf = np.cumsum(w)

    # Sample u0 ~ Uniform(0, 1/n)
    # (Using jax for RNG since `key` comes from jax.random.split)
    try:
        u0 = float(random.uniform(key, shape=())) / float(n_particles)
    except Exception:
        u0 = float(np.random.rand()) / float(n_particles)

    positions = u0 + (np.arange(n_particles, dtype=float) / float(n_particles))
    index = np.searchsorted(cdf, positions, side='left')
    return [x[index] for x in L_to_resample]


def ess(weight):
    w = np.asarray(weight)
    return float(np.round(1.0 / np.sum(w ** 2)))
