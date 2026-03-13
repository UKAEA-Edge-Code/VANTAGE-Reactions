#ifndef REACTIONS_INTERP_UTILS_H
#define REACTIONS_INTERP_UTILS_H
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions::interp_utils {
/**
 * Helper macro that extracts the value of the binary representation of i at
 * position j (in the binary representation of i).
 */
template <typename T> inline T binary_extract(const T &i, const size_t &j) {
  return ((i >> j) & 1);
}

/**
 * @brief Helper function to calculate the index on a contiguous row-major grid
 * array where the indices run from fastest index to slowest index.
 *
 * @param indices Pointer to a vector that contains the indices to access grid
 * data (an index for each dimension of the non-flattened array).
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param ndim The number of dimensions
 *
 * @return size_t that specifies the index on a contiguous grid array
 */
inline size_t coeff_index_on_device(size_t const *indices,
                                    size_t const *dims_vec,
                                    const size_t &ndim) {
  size_t index = indices[ndim - 1];

  // slight re-factor to avoid underflow errors with size_t.
  size_t dimx = ndim - 1;
  while (dimx > 0) {
    --dimx;
    index *= dims_vec[dimx];
    index += indices[dimx];
  }

  return index;
}

/**
 * @brief Similar to coeff_index_on_device in that it returns an index on a
 * contiguous row-major array containing the ranges of each dimension of
 * relevance for the interpolation.
 *
 * @param sub_index The index for the specific dimension of interest
 * @param dim_index The index of the dimension itself, as in for the 2nd
 * dimension of a 4D grid, the dim_index=2
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 *
 * @return size_t that specifies the index on a contiguous ranges array.
 */
inline size_t range_index_on_device(const size_t &sub_index,
                                    const size_t &dim_index,
                                    size_t const *dims_vec) {
  size_t index = sub_index;

  for (size_t i = 0; i < dim_index; i++) {
    index += dims_vec[i];
  }

  return index;
}

/**
 * @brief Helper function to calculate the index on a given dimension that is
 * the closest to a given interpolation point. The preference is to provide the
 * lowest(or left-most) index.
 *
 * @param x_interp Value of the interpolation point for a given dimension
 * @param dim_range Pointer to a vector containing the range of values for a
 * given dimension.
 * @param last_index The last index in the range of the given dimension.
 *
 * @return size_t The index on a given dimension that is the closest to
 * x_interp.
 */
inline size_t calc_floor_point_index(const REAL &x_interp,
                                     REAL const *dim_range,
                                     const size_t &last_index) {
  size_t L = 0;
  size_t R = last_index;
  size_t m;

  while ((R - L) > 1) {
    m = L + ((R - L) / 2);
    if (dim_range[m] < x_interp) {
      L = m;
    } else if (dim_range[m] > x_interp) {
      R = m;
    } else {
      // for exact matches
      return m;
    }
  }

  return L;
}

/**
 * @brief Function to perform a 1D linear interpolation.
 *
 * @param x_interp The interpolation point in a given dimension
 * @param x0 The highest-value point on the dimension that is less than x_interp
 * @param x1 The lowest-value point on the dimension that is greater than
 * x_interp
 * @param f0 The function value at x0.
 * @param f1 The function value at x1.
 *
 * @return REAL value of the linearly interpolated function value at x_interp.
 */
inline REAL linear_interp(const REAL x_interp, const REAL x0, const REAL x1,
                          const REAL f0, const REAL f1) {
  // The excessive splitting of operations is due to a failed unit tests on
  // cudallvm compilationflow when using variables whose definitions combine
  // multiple operations.
  REAL df = f1 - f0;
  REAL dx = x1 - x0;
  REAL dfdx = dx != 0.0f ? (df / dx) : 0.0;
  REAL c = f0 - (dfdx * x0);

  return (dfdx * x_interp) + c;
}

/**
 * @brief Function to construct a series of points that constitute the vertices
 * of N-Dimensional hypercube. The points are integers but the binary
 * representations denote the normalised vertices. For example in 2D: 0, 1, 3, 2
 * where the binary representations would be: 00, 01, 11, 10 which would
 * correspond to the vertices (0, 0), (0, 1), (1, 1), (1, 0).
 *
 * @param ndim The number of dimensions.
 *
 * @return std::vector<INT> That contains the points denoting the vertices of
 * the hypercube.
 */
inline std::vector<size_t> construct_initial_hypercube(const size_t &ndim) {
  size_t total_num = 1 << ndim;
  std::vector<size_t> points(total_num);

  for (size_t i = 0; i < total_num; i++) {
    points[i] = (i ^ (i >> 1));
  }

  return points;
}

/**
 * Function to calculate the initial function values on the vertices of the
 * hypercube.
 *
 * @param vertex_func_evals Pointer to a vector to fill
 * with function evaluations.
 * @param vertex_coord Pointer to a vector to fill with
 * locations of the vertices of the hypercube.
 * @param func_grid Pointer to a contiguous array containing the ND grid data.
 * @param hypercube_vertices Pointer to a vector containing the vertices of the
 * hypercube (integers whose binary representations give the normalised
 * positions of the vertices).
 * @param origin_indices Pointer to a vector containing the
 * indices that will form the (0,0) point of the hypercube (that is to say, the
 * largest indices in each dimension that are still smaller than the desired
 * interpolation point).
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param ndim The number of dimensions.
 * @param num_points The number of vertices needed for the hypercube
 * representation
 */
inline void initial_func_eval_on_device(
    REAL *vertex_func_evals, size_t *vertex_coord, REAL const *func_grid,
    size_t const *hypercube_vertices, size_t const *origin_indices,
    size_t const *dims_vec, const size_t &ndim, const size_t &num_points) {
  for (size_t point_index = 0; point_index < num_points; point_index++) {
    for (size_t vertex_index = 0; vertex_index < ndim; vertex_index++) {
      vertex_coord[vertex_index] =
          origin_indices[vertex_index] +
          binary_extract(hypercube_vertices[point_index], vertex_index);
    }
    vertex_func_evals[point_index] =
        func_grid[coeff_index_on_device(vertex_coord, dims_vec, ndim)];
  }
}

/**
 * @brief Function to contract a hypercube down by 1 dimension via linear
 * interpolation.
 *
 * @param interp_points Pointer to a vector that contains
 * the interpolation points in each dimension.
 * @param dim_index Since this function is called multiple times, this counter
 * keeps track of the progress, it can be thought of as: ndim-1 where ndim is
 * the current dimensionality of the hypercube.
 * @param hypercube_vertices Pointer to a vector that contains
 * the vertices of the hypercube pre-contraction.
 * @param origin_indices Pointer to a vector containing the
 * indices that will form the (0,0) point of the hypercube (that is to say, the
 * largest indices in each dimension that are still smaller than the desired
 * interpolation point).
 * @param vertex_func_evals Pointer to a vector that
 * contains the function evaluations at initial vertices.
 * @param ranges_vec Pointer to a vector containing a
 * contiguous array of the ranges of each dimension of relevance for the
 * interpolation.
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param output_evals Pointer to a vector that contains
 * the function evaluations at contracted vertices.
 * @param varying_dim Pointer to a vector used for storing
 * the vertices whose coordinates vary in the dimension to be contracted.
 * @param vertex_coord Pointer to a vector used for
 * storing the vertices of the hypercube after they've been mapped to the actual
 * region in the dimensions of the grid that are of interest.
 */

inline void contract_hypercube_on_device(
    const REAL *interp_points, const size_t &dim_index,
    size_t const *hypercube_vertices, const size_t *origin_indices,
    const REAL *vertex_func_evals, REAL const *ranges_vec,
    size_t const *dims_vec, REAL *output_evals, size_t *varying_dim,
    size_t *vertex_coord) {
  size_t ndim = dim_index + 1;
  size_t num_points = (1 << ndim);
  size_t num_out_points = (1 << dim_index);

  for (size_t point_index = 0; point_index < num_points; point_index++) {
    for (size_t eval_index = 0; eval_index < ndim; eval_index++) {
      vertex_coord[eval_index] =
          origin_indices[eval_index] +
          binary_extract(hypercube_vertices[point_index], eval_index);
    }
    varying_dim[point_index] = vertex_coord[dim_index];
  }

  size_t vertex_0, vertex_1;
  REAL range_val_0, range_val_1, eval_point_0, eval_point_1;

  for (size_t i = 0; i < num_out_points; i++) {
    vertex_0 = varying_dim[i];
    vertex_1 = varying_dim[num_points - (i + 1)];

    range_val_0 = // x0
        ranges_vec[range_index_on_device(vertex_0, dim_index, dims_vec)];
    range_val_1 = // x1
        ranges_vec[range_index_on_device(vertex_1, dim_index, dims_vec)];

    eval_point_0 = vertex_func_evals[i];                    // f0
    eval_point_1 = vertex_func_evals[num_points - (i + 1)]; // f1

    output_evals[i] = linear_interp(interp_points[dim_index], range_val_0,
                                    range_val_1, eval_point_0, eval_point_1);
  }
}

} // namespace VANTAGE::Reactions::interp_utils
#endif
