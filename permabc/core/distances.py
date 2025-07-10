import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
from typing import Optional, Tuple
from .lsa import solve_lsa

def remove_under_matching(
    ys_index: np.ndarray,
    zs_index: np.ndarray,
    M: int,
    L: int,
    K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove assignments where indices are out of bounds for matching.

    Args:
        ys_index: Indices for y assignments, shape (N, K).
        zs_index: Indices for z assignments, shape (N, K).
        M: Maximum index for zs.
        L: Number of assignments to keep.
        K: Maximum index for ys.

    Returns:
        Tuple of filtered ys_index and zs_index, both shape (N, L).
    """
    new_zs_index = np.zeros((zs_index.shape[0], L), dtype=np.int64)
    new_ys_index = np.zeros((ys_index.shape[0], L), dtype=np.int64)
    for i in range(ys_index.shape[0]):
        l = 0
        for j in range(ys_index.shape[1]):
            if zs_index[i, j] < M and ys_index[i, j] < K:
                new_zs_index[i, l] = zs_index[i, j]
                new_ys_index[i, l] = ys_index[i, j]
                l += 1
    return np.array(new_ys_index, dtype=np.int32), np.array(new_zs_index, dtype=np.int32)

def compute_total_distance(
    zs_index: np.ndarray,
    ys_index: np.ndarray,
    local_dist_matrices: np.ndarray,
    global_distances: np.ndarray
) -> np.ndarray:
    """
    Compute the total distance for each particle given assignment indices.

    Args:
        zs_index: Indices for z assignments, shape (N, K).
        ys_index: Indices for y assignments, shape (N, K).
        local_dist_matrices: Local distance matrices, shape (N, K, K).
        global_distances: Global distances, shape (N,).

    Returns:
        Array of total distances, shape (N,).
    """
    return np.sqrt(vmap(jit(lambda matrix, zs_idx, ys_idx, glob: matrix[ys_idx, zs_idx].sum() + glob), in_axes=(0, 0, 0, 0))(
        local_dist_matrices, zs_index, ys_index, global_distances))



def optimal_index_distance(
    model,
    zs: np.ndarray,
    y_obs: np.ndarray,
    epsilon: float = 0,
    ys_index: Optional[np.ndarray] = None,
    zs_index: Optional[np.ndarray] = None,
    verbose: int = 0,
    M: int = 0,
    L: int = 0,
    parallel: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute optimal assignment distances with smart acceptance.

    Args:
        model: Model object with required methods and attributes.
        zs: Array of z particles, shape (N, K, dim).
        y_obs: Observed y values, shape (1, K, dim).
        epsilon: Acceptance threshold.
        ys_index: Optional initial y assignment indices.
        zs_index: Optional initial z assignment indices.
        verbose: Verbosity level.
        M: Max index for zs.
        L: Number of assignments to keep.
        parallel: Whether to use parallel computation.

    Returns:
        Tuple of (optimal_distances, ys_index, zs_index, num_lsa).
    """
    num_particles = zs.shape[0]
    K = model.K

    if M == 0: M = K
    if L == 0: L = K
    if zs_index is None or ys_index is None:
        if verbose > 1:
            print("Performing full optimal assignment (no prior indices).")
        
        local_dist_matrices = model.distance_matrices_loc(zs, y_obs, M, L)
        global_distances = model.distance_global(zs, y_obs)
        local_dist_matrices = np.where(np.isinf(local_dist_matrices), 1e12, local_dist_matrices)
        
        new_ys_index, new_zs_index = solve_lsa(local_dist_matrices, parallel=parallel)
        if L < K:
            new_ys_index, new_zs_index = remove_under_matching(new_ys_index, new_zs_index, M, L, K)

        optimal_distances = compute_total_distance(new_zs_index, new_ys_index, local_dist_matrices, global_distances)
        return optimal_distances, new_ys_index, new_zs_index, num_particles
    
    # Case 2: Selective computation with smart acceptance
    current_distances = compute_distances_with_current_assignment(
        model, zs, y_obs, ys_index, zs_index, M, L
    )
    particles_to_reassign = np.where(current_distances >= epsilon)[0]
    num_lsa = len(particles_to_reassign)
    
    if num_lsa == 0:
        if verbose > 1:
            print("All particles accepted via smart acceptance (no matrix computation needed).")
        return current_distances, ys_index, zs_index, 0
    
    if verbose > 1:
        print(f"Computing optimal assignment for {num_lsa}/{num_particles} particles ({num_lsa/num_particles:.1%})")
    
    zs_subset = zs[particles_to_reassign]
    local_dist_matrices_subset = model.distance_matrices_loc(zs_subset, y_obs, M, L)
    global_distances_subset = model.distance_global(zs_subset, y_obs)
    local_dist_matrices_subset = np.where(np.isinf(local_dist_matrices_subset), 1e12, local_dist_matrices_subset)
    
    new_ys_index_subset, new_zs_index_subset = solve_lsa(local_dist_matrices_subset, parallel=parallel)
    if L < K:
        new_ys_index_subset, new_zs_index_subset = remove_under_matching(
            new_ys_index_subset, new_zs_index_subset, M, L, K
        )
    
    optimal_distances_subset = compute_total_distance(
        new_zs_index_subset, new_ys_index_subset,
        local_dist_matrices_subset, global_distances_subset
    )
    
    optimal_distances = current_distances.copy()
    optimal_distances[particles_to_reassign] = optimal_distances_subset
    
    ys_index = ys_index.copy()
    zs_index = zs_index.copy()
 
    if L < K:
        new_ys_index_subset, new_zs_index_subset = remove_under_matching(
            new_ys_index_subset, new_zs_index_subset, M, L, K
        )
    ys_index[particles_to_reassign] = new_ys_index_subset
    zs_index[particles_to_reassign] = new_zs_index_subset
    
    if verbose > 1:
        smart_accepted = num_particles - num_lsa
        newly_accepted = np.sum(optimal_distances_subset < epsilon)
    if L < K:
        ys_index, zs_index = remove_under_matching(ys_index, zs_index, M, L, K)

    return optimal_distances, ys_index, zs_index, num_lsa

def compute_distances_with_current_assignment(
    model,
    zs: np.ndarray,
    y_obs: np.ndarray,
    ys_index: np.ndarray,
    zs_index: np.ndarray,
    M: int,
    L: int
) -> np.ndarray:
    """
    Compute total distances using current assignments without full matrix computation.

    Args:
        model: Model object with required methods and attributes.
        zs: Array of z particles, shape (N, K, dim).
        y_obs: Observed y values, shape (1, K, dim).
        ys_index: Current y assignment indices, shape (N, K).
        zs_index: Current z assignment indices, shape (N, K).
        M: Max index for zs.
        L: Number of assignments to keep.

    Returns:
        Array of total distances, shape (N,).
    """
    num_particles = zs.shape[0]
    global_distances = model.distance_global(zs, y_obs)
    local_distances = compute_local_distances_for_assignment(
        model, zs, y_obs, ys_index, zs_index, M, L
    )
    return np.sqrt(local_distances + global_distances)

def compute_local_distances_for_assignment(
    model,
    zs: np.ndarray,
    y_obs: np.ndarray,
    ys_index: np.ndarray,
    zs_index: np.ndarray,
    M: int,
    L: int
) -> np.ndarray:
    """
    Compute local distances for each particle using current assignments.

    Args:
        model: Model object with required attributes.
        zs: Array of z particles, shape (N, K, dim).
        y_obs: Observed y values, shape (1, K, dim).
        ys_index: Current y assignment indices, shape (N, K).
        zs_index: Current z assignment indices, shape (N, K).
        M: Max index for zs.
        L: Number of assignments to keep.

    Returns:
        Array of local distances, shape (N,).
    """
    K = model.K
    num_particles = ys_index.shape[0]
    dim = y_obs.shape[2]
    y_obs_slice = y_obs[0, :K]  # (K, dim)
    weights_slice = model.weights_distance[:K]  # (K,)
    particle_indices = np.arange(num_particles)[:, None]  # (num_particles, 1)
    ys_index = ys_index.astype(np.int32)
    y_assigned = y_obs_slice[ys_index]  # (num_particles, K, dim)
    zs_index = zs_index.astype(np.int32)
    z_assigned = zs[particle_indices, zs_index]  # (num_particles, K, dim)
    weights_assigned = weights_slice[ys_index]  # (num_particles, K)
    diff = y_assigned - z_assigned  # (num_particles, K, dim)
    squared_diffs = np.sum(diff ** 2, axis=2)  # (num_particles, K)
    weighted_distances = weights_assigned**2 * squared_diffs  # (num_particles, K)
    local_distances = np.sum(weighted_distances, axis=1)  # (num_particles,)
    return local_distances