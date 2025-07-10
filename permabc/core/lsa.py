"""
Linear Sum Assignment (LSA) solver with parallel processing.

This module provides an optimized wrapper around scipy's linear_sum_assignment
with automatic parallelization for handling multiple distance matrices efficiently.
"""

from scipy.optimize import linear_sum_assignment
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp
import concurrent.futures


def solve_lsa(dist_matrices, indices=None, parallel=True, n_jobs=-1):
    """
    Solve multiple Linear Sum Assignment problems with optional parallelization.
    
    This function is the core component for optimal permutation finding in permABC.
    It efficiently solves multiple LSA problems in parallel when dealing with large
    distance matrices or many particles.
    
    Parameters
    ----------
    dist_matrices : numpy.ndarray
        Array of distance matrices with shape (n_matrices, n_rows, n_cols).
        Each matrix [i] represents costs for assigning rows to columns.
    indices : numpy.ndarray, optional
        Subset of matrix indices to process. If None, processes all matrices.
        Useful for selective reassignment in ABC-SMC.
    parallel : bool, default=True
        Whether to use parallel processing. Automatically disabled for small problems.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available CPU cores.
    
    Returns
    -------
    ys_idx : numpy.ndarray
        Row indices of optimal assignments, shape (n_matrices, n_assignments).
    zs_idx : numpy.ndarray  
        Column indices of optimal assignments, shape (n_matrices, n_assignments).
        
    Notes
    -----
    - Automatically chooses sequential vs parallel processing based on problem size
    - Uses ThreadPoolExecutor for better performance with scipy functions
    - Chunk size is optimized based on number of matrices and CPU cores
    - For small problems (≤40x40), always uses sequential processing
    
    Examples
    --------
    >>> dist_matrices = np.random.rand(100, 10, 10)  # 100 matrices of size 10x10
    >>> row_idx, col_idx = solve_lsa(dist_matrices, parallel=True)
    >>> print(f"Solved {len(row_idx)} assignment problems")
    """
    def solve_chunk(chunk):
        """Solve a chunk of LSA problems sequentially."""
        return [linear_sum_assignment(matrix) for matrix in chunk]
    
    # Handle indices subset
    if indices is None:
        indices = np.arange(dist_matrices.shape[0])
    n_matrices = len(indices)
    
    # Decide on parallelization strategy
    if parallel and np.max(dist_matrices.shape[1:]) > 40:
        n_cpu = mp.cpu_count() if n_jobs == -1 else n_jobs
        
        # Adaptive chunk sizing based on problem size
        if n_matrices <= 1000:
            chunk_size = max(20, n_matrices // (n_cpu * 2))
        else:  # Large problems (10000+)
            chunk_size = max(50, n_matrices // (n_cpu * 3))

        # Create chunks for parallel processing
        chunks = [dist_matrices[i:i+chunk_size] for i in range(0, len(dist_matrices), chunk_size)]
        # print(f"Using {len(chunks)} chunks of size {chunk_size} for parallel processing on {n_cpu} CPUs.")
        
        # Single chunk doesn't need parallelization
        if len(chunks) == 1:
            return solve_chunk(chunks[0])
        
        # Parallel processing with ThreadPoolExecutor
        max_workers = min(n_cpu, len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(solve_chunk, chunks))
        
        # Flatten results from all chunks
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
            
    else:
        # Sequential processing for small problems
        # print("Using sequential processing.")
        results = solve_chunk(dist_matrices)

    # Unpack and convert results
    ys_idx, zs_idx = zip(*results)
    return np.array(ys_idx, dtype=np.int32), np.array(zs_idx, dtype=np.int32)