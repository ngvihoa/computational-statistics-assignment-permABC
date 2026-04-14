#include "rectangular_lsap_custom.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-9;   // reduced-cost tolerance for warm-start seeding

inline double cost_at(const double* c, int nc, int i, int j) {
    return c[static_cast<std::size_t>(i) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(j)];
}

// ── Dual initialisation ────────────────────────────────────────────────────
// Classic row-min / remaining-col-min heuristic.
// Produces feasible duals (reduced costs ≥ 0) without any assignment.
static void init_duals(int nr, int nc, const double* cost,
                       std::vector<double>& u, std::vector<double>& v) {
    u.assign(nr, 0.0);
    v.assign(nc, 0.0);

    // u[i] = min_j cost[i,j]
    for (int i = 0; i < nr; ++i) {
        double mn = INF;
        for (int j = 0; j < nc; ++j)
            mn = std::min(mn, cost_at(cost, nc, i, j));
        u[i] = mn;
    }
    // v[j] = min_i (cost[i,j] - u[i])
    for (int j = 0; j < nc; ++j) {
        double mn = INF;
        for (int i = 0; i < nr; ++i)
            mn = std::min(mn, cost_at(cost, nc, i, j) - u[i]);
        v[j] = mn;
    }
}

// ── Warm-start seeding ────────────────────────────────────────────────────
// Pre-assign pairs (i, hint[i]) that are already tight (reduced cost ≤ EPS)
// and do not conflict with each other.  Returns number of seeded pairs.
static int seed_warm_start(int nr, int nc, const double* cost,
                            const int* init_col4row,
                            const std::vector<double>& u,
                            const std::vector<double>& v,
                            int* row4col, int* col4row) {
    int seeded = 0;
    for (int i = 0; i < nr; ++i) {
        int j = init_col4row[i];
        if (j < 0 || j >= nc)    continue;  // out-of-range hint
        if (row4col[j] != -1)    continue;  // column already claimed
        if (col4row[i] != -1)    continue;  // row already assigned
        double red = cost_at(cost, nc, i, j) - u[i] - v[j];
        if (red <= EPS) {
            col4row[i] = j;
            row4col[j] = i;
            ++seeded;
        }
    }
    return seeded;
}

// ── Core Jonker–Volgenant (shortest augmenting path) ─────────────────────
static int jv_augment(int nr, int nc, const double* cost,
                      int* row4col, int* col4row,
                      std::vector<double>& u, std::vector<double>& v) {
    // Scratch buffers (reused across rows)
    std::vector<double> minv(nc);
    std::vector<int>    links_row(nc);
    std::vector<char>   used_col(nc);
    std::vector<char>   used_row(nr);

    for (int start_i = 0; start_i < nr; ++start_i) {
        if (col4row[start_i] != -1) continue;  // already matched (warm-start)

        std::fill(minv.begin(),      minv.end(),      INF);
        std::fill(links_row.begin(), links_row.end(), -1);
        std::fill(used_col.begin(),  used_col.end(),  0);
        std::fill(used_row.begin(),  used_row.end(),  0);

        int i    = start_i;
        int j0   = -1;

        while (true) {
            used_row[i] = 1;
            double delta = INF;
            int    j1    = -1;

            for (int j = 0; j < nc; ++j) {
                if (used_col[j]) continue;
                double cur = cost_at(cost, nc, i, j) - u[i] - v[j];
                if (cur < minv[j]) {
                    minv[j]      = cur;
                    links_row[j] = i;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1    = j;
                }
            }

            if (!std::isfinite(delta)) return -4;

            // Update duals
            for (int ii = 0; ii < nr; ++ii)
                if (used_row[ii]) u[ii] += delta;
            for (int j = 0; j < nc; ++j) {
                if (used_col[j]) v[j] -= delta;
                else             minv[j] -= delta;
            }

            used_col[j1] = 1;
            j0           = j1;
            int i0       = row4col[j0];

            if (i0 == -1) {
                // Augment along the path back to start_i
                while (true) {
                    int ii     = links_row[j0];
                    int next_j = col4row[ii];
                    row4col[j0] = ii;
                    col4row[ii] = j0;
                    j0          = next_j;
                    if (ii == start_i) break;
                }
                break;
            } else {
                i = i0;
            }
        }
    }
    return 0;
}

}  // namespace

// ── Public C interface ────────────────────────────────────────────────────
extern "C"
int solve_rectangular_linear_sum_assignment_ws(
    int nr, int nc,
    const double* cost,
    int* row4col, int* col4row,
    int use_init, const int* init_col4row,
    int* warm_supported, int* warm_used)
{
    if (nr <= 0 || nc <= 0)  return -1;
    if (nr > nc)             return -2;   // caller must transpose first

    // Validate all finite
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
            if (!std::isfinite(cost_at(cost, nc, i, j))) return -3;

    // Init output arrays
    for (int j = 0; j < nc; ++j) row4col[j] = -1;
    for (int i = 0; i < nr; ++i) col4row[i] = -1;

    const bool will_warm = (use_init && init_col4row);
    if (warm_supported) *warm_supported = will_warm ? 1 : 0;
    if (warm_used)      *warm_used      = 0;

    // Dual initialisation
    std::vector<double> u, v;
    init_duals(nr, nc, cost, u, v);

    // Optional warm-start seeding
    if (will_warm) {
        int seeded = seed_warm_start(nr, nc, cost, init_col4row, u, v,
                                     row4col, col4row);
        if (warm_used) *warm_used = seeded;
    }

    // Augmenting-path phase (skips already-matched rows)
    return jv_augment(nr, nc, cost, row4col, col4row, u, v);
}
