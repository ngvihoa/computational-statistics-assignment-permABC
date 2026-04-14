#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve a rectangular linear sum assignment problem (nr <= nc required).
 *
 * Algorithm: Jonker–Volgenant shortest-augmenting-path with optional warm start.
 *
 * Warm start: if use_init=1 and init_col4row is provided, pairs (i, init_col4row[i])
 * whose reduced cost is ≤ EPS are seeded directly into the matching before the
 * augmenting-path phase, reducing the number of paths that need to be found.
 *
 * Parameters
 * ----------
 * nr, nc         : matrix dimensions (nr rows, nc cols).  Must have nr <= nc.
 * cost           : row-major cost matrix of size nr*nc (doubles).
 * row4col[nc]    : output — for each column j, row assigned to j (-1 = unmatched).
 * col4row[nr]    : output — for each row i, column assigned to i.
 * use_init       : 1 to attempt warm start, 0 for cold start.
 * init_col4row   : warm-start hint: init_col4row[i] = column hint for row i.
 *                  May be NULL even when use_init=1 (treated as cold start).
 * warm_supported : output flag — 1 if warm start was attempted (nr<=nc, not transpose).
 * warm_used      : output — number of hint pairs actually seeded into the matching.
 *
 * Return codes
 * ------------
 *  0  : success
 * -1  : invalid dimensions (nr<=0 or nc<=0)
 * -2  : transpose case not supported (nr > nc) — swap the matrix before calling
 * -3  : non-finite value in cost matrix
 * -4  : augmenting path failed (should not happen for finite costs)
 */
int solve_rectangular_linear_sum_assignment_ws(
    int nr,
    int nc,
    const double* cost,
    int* row4col,
    int* col4row,
    int use_init,
    const int* init_col4row,
    int* warm_supported,
    int* warm_used);

#ifdef __cplusplus
}
#endif
