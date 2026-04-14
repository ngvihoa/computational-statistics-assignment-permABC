/**
 * cgal_hilbert.cpp — efficient pybind11 wrapper for CGAL::hilbert_sort
 *
 * Key design:
 *   - sort INDICES via Spatial_sort_traits_adapter_2/3/d
 *   - no O(n²) recover_permutation(), no floating-point matching
 *   - int64 output (compatible with numpy indexing directly)
 *
 * Exposes:
 *   hilbert_sort_2d(points, policy="median") -> np.int64 sorted indices
 *   hilbert_sort_3d(points, policy="median") -> np.int64 sorted indices
 *   hilbert_sort_nd(points, policy="median") -> np.int64 sorted indices  (d>=2)
 *
 * Note: sorted_idx[rank] = original_index
 *       i.e. points[sorted_idx] is the Hilbert-sorted order.
 *
 * Build:
 *   bash build_cgal.sh permabc
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/hilbert_sort.h>
#include <CGAL/Hilbert_policy_tags.h>
#include <CGAL/Spatial_sort_traits_adapter_2.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/Spatial_sort_traits_adapter_d.h>

#include <boost/property_map/property_map.hpp>

#include <vector>
#include <numeric>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <cstdint>

namespace py = pybind11;

// ── Kernels ──────────────────────────────────────────────────────────────────
using K2     = CGAL::Simple_cartesian<double>;
using Point_2 = K2::Point_2;
using Point_3 = K2::Point_3;

using Kd     = CGAL::Cartesian_d<double>;
using Point_d = Kd::Point_d;

// ── Property-map: index → point stored in a vector ───────────────────────────
template <class Point>
using IndexMap = boost::iterator_property_map<
    typename std::vector<Point>::const_iterator,
    boost::typed_identity_property_map<std::size_t>
>;

// ── Helper: run the sort and write result to a pre-allocated int64 array ─────
template <class TraitsT>
static void do_hilbert_sort(std::vector<std::size_t>& idx,
                             TraitsT& traits,
                             const std::string& policy)
{
    if (policy == "middle") {
        CGAL::hilbert_sort(idx.begin(), idx.end(), traits,
                           CGAL::Hilbert_sort_middle_policy());
    } else if (policy == "median") {
        CGAL::hilbert_sort(idx.begin(), idx.end(), traits,
                           CGAL::Hilbert_sort_median_policy());
    } else {
        throw std::runtime_error("policy must be 'median' or 'middle'");
    }
}

static py::array_t<std::int64_t> to_int64(const std::vector<std::size_t>& idx)
{
    py::array_t<std::int64_t> out(static_cast<py::ssize_t>(idx.size()));
    auto* ptr = static_cast<std::int64_t*>(out.request().ptr);
    for (std::size_t i = 0; i < idx.size(); ++i)
        ptr[i] = static_cast<std::int64_t>(idx[i]);
    return out;
}

// ── 2D ───────────────────────────────────────────────────────────────────────
py::array_t<std::int64_t>
hilbert_sort_2d(py::array_t<double, py::array::c_style | py::array::forcecast> pts,
                const std::string& policy = "median")
{
    auto buf = pts.request();
    if (buf.ndim != 2 || buf.shape[1] != 2)
        throw std::runtime_error("Expected K×2 array");

    const std::size_t n = static_cast<std::size_t>(buf.shape[0]);
    const double* data  = static_cast<const double*>(buf.ptr);

    std::vector<Point_2> points;
    points.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        points.emplace_back(data[2*i], data[2*i+1]);

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});

    using Traits = CGAL::Spatial_sort_traits_adapter_2<K2, IndexMap<Point_2>>;
    IndexMap<Point_2> pmap(points.begin());
    Traits traits(pmap);

    do_hilbert_sort(idx, traits, policy);
    return to_int64(idx);
}

// ── 3D ───────────────────────────────────────────────────────────────────────
py::array_t<std::int64_t>
hilbert_sort_3d(py::array_t<double, py::array::c_style | py::array::forcecast> pts,
                const std::string& policy = "median")
{
    auto buf = pts.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw std::runtime_error("Expected K×3 array");

    const std::size_t n = static_cast<std::size_t>(buf.shape[0]);
    const double* data  = static_cast<const double*>(buf.ptr);

    std::vector<Point_3> points;
    points.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        points.emplace_back(data[3*i], data[3*i+1], data[3*i+2]);

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});

    using Traits = CGAL::Spatial_sort_traits_adapter_3<K2, IndexMap<Point_3>>;
    IndexMap<Point_3> pmap(points.begin());
    Traits traits(pmap);

    do_hilbert_sort(idx, traits, policy);
    return to_int64(idx);
}

// ── d-D (dimension dynamique, d>=2) ─────────────────────────────────────────
py::array_t<std::int64_t>
hilbert_sort_nd(py::array_t<double, py::array::c_style | py::array::forcecast> pts,
                const std::string& policy = "median")
{
    auto buf = pts.request();
    if (buf.ndim != 2 || buf.shape[1] < 2)
        throw std::runtime_error("Expected K×d array with d>=2");

    const std::size_t n = static_cast<std::size_t>(buf.shape[0]);
    const int         d = static_cast<int>(buf.shape[1]);
    const double* data  = static_cast<const double*>(buf.ptr);

    std::vector<Point_d> points;
    points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double* row = data + i * static_cast<std::size_t>(d);
        points.emplace_back(d, row, row + d);
    }

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});

    using Traits = CGAL::Spatial_sort_traits_adapter_d<Kd, IndexMap<Point_d>>;
    IndexMap<Point_d> pmap(points.begin());
    Traits traits(pmap);

    do_hilbert_sort(idx, traits, policy);
    return to_int64(idx);
}

// ── Module ───────────────────────────────────────────────────────────────────
PYBIND11_MODULE(cgal_hilbert, m) {
    m.doc() = R"doc(
Efficient CGAL::hilbert_sort Python wrapper.

Design:
- Sort indices via Spatial_sort_traits_adapter_2/3/d — no O(n²) matching.
- Returns int64 sorted_idx such that  points[sorted_idx]  is the Hilbert order.

Functions:
  hilbert_sort_2d(points, policy="median") -> np.int64[K]
  hilbert_sort_3d(points, policy="median") -> np.int64[K]
  hilbert_sort_nd(points, policy="median") -> np.int64[K]  (d>=2)
)doc";

    m.def("hilbert_sort_2d", &hilbert_sort_2d,
          py::arg("points"), py::arg("policy") = "median",
          "Hilbert-sort K×2 array. Returns int64 sorted indices.");

    m.def("hilbert_sort_3d", &hilbert_sort_3d,
          py::arg("points"), py::arg("policy") = "median",
          "Hilbert-sort K×3 array. Returns int64 sorted indices.");

    m.def("hilbert_sort_nd", &hilbert_sort_nd,
          py::arg("points"), py::arg("policy") = "median",
          "Hilbert-sort K×d (d>=2) array. Returns int64 sorted indices.");
}
