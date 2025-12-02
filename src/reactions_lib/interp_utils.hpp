#ifndef REACTIONS_INTERP_UTILS_H
#define REACTIONS_INTERP_UTILS_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include <neso_particles.hpp>
#include <neso_particles/containers/local_memory_interlaced.hpp>
#include <vector>

#define binary_extract(i, j) ((i >> j) & 1)

using namespace NESO::Particles;
namespace VANTAGE::Reactions::interp_utils {

/**
 * @brief Helper function to calculate the index on a contiguous grid array from
 * multiple indices (in each dimension). For example in 2D, grid[5][4] for a
 * 8x10 grid: coeff_index_on_devices(indices = {5, 4}, dims_vec = {8, 10}, ndim
 * = 2)
 * -> (5 + (4 * 8)) = 37
 *
 * @param indices Accessor for LocalMemoryInterlaced that contains the indices
 * to access grid data
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param ndim The number of dimensions
 *
 * @return std::size_t that specifies the index on a contiguous grid array
 */
inline std::size_t
coeff_index_on_device(Access::LocalMemoryInterlaced::Write<std::size_t> indices,
                      std::size_t *dims_vec, const int &ndim) {
  std::size_t index = indices.at(0);

  std::size_t dim_mult;
  for (int i = 1; i < ndim; i++) {
    dim_mult = 1;
    for (int j = int(i - 1); j >= 0; j--) {
      dim_mult *= dims_vec[j];
    }
    index += dim_mult * indices.at(i);
  }

  return index;
}

/**
 * @brief Helper function to calculate the index on a contiguous grid array from
 * multiple indices (in each dimension). For example in 2D, grid[5][4] for a
 * 8x10 grid: coeff_index_on_devices(indices = {5, 4}, dims_vec = {8, 10}, ndim
 * = 2)
 * -> (5 + (4 * 8)) = 37
 *
 * @param indices Pointer to a vector that contains the indices to access grid
 * data
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param ndim The number of dimensions
 *
 * @return std::size_t that specifies the index on a contiguous grid array
 */
inline INT coeff_index_on_device(INT *indices, size_t *dims_vec,
                                 const int &ndim) {
  INT index = indices[0];

  std::size_t dim_mult;
  for (int i = 1; i < ndim; i++) {
    dim_mult = 1;
    for (int j = int(i - 1); j >= 0; j--) {
      dim_mult *= dims_vec[j];
    }
    index += dim_mult * indices[i];
  }

  return index;
}

/**
 * @brief Similar to coeff_index_on_device in that it returns an index on a
 * contiguous array containing the ranges of each dimension of relevance for the
 * interpolation. For example in 2D, for an 8x10 grid there's a 18 element
 * arrays containing the values of ranges for each dimension (such as the
 * x-values(8) and y-values9(10)).
 *
 * @param sub_index The index for the specific dimension of interest
 * @param dim_index The index of the dimension itself, as in for the 2nd
 * dimension of a 4D grid, the dim_index=3
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 *
 * @return std::size_t that specifies the index on a contiguous ranges array.
 */
inline std::size_t range_index_on_device(const std::size_t &sub_index,
                                         const std::size_t &dim_index,
                                         std::size_t *dims_vec) {
  std::size_t index = sub_index;

  for (std::size_t i = 0; i < dim_index; i++) {
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
 * @param dim_size The number of points in each dimension.
 *
 * @return std::size_t The index on a given dimension that is the closest to
 * x_interp.
 */
inline std::size_t calc_closest_point_index(const REAL &x_interp,
                                            REAL *dim_range,
                                            const std::size_t &dim_size) {
  std::size_t L = 0;
  std::size_t R = dim_size - 1;
  std::size_t m;

  while ((R - L) > 1) {
    m = ((L + R) / 2);
    if (dim_range[m] < x_interp) {
      L = m;
    } else if (dim_range[m] > x_interp) {
      R = m;
    } else {
      // for exact matches
      return m;
    }
  }

  return R == (dim_size - 1) ? R : L;
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
inline REAL linear_interp(const REAL &x_interp, const REAL &x0, const REAL &x1,
                          const REAL &f0, const REAL &f1) {
  REAL dfdx = (f1 - f0) / (x1 - x0);
  REAL c = f0 - (dfdx * x0);

  REAL f_interp = (dfdx * x_interp) + c;
  return f_interp;
}

/**
 * @brief Function to construct a series of points that constitute the vertices
 * of N-Dimensional hypercube. The points are integers but the binary
 * representations denote the normalised vertices. For example in 2D: 0, 1, 3, 2
 * where the binary representations would be: 00, 01, 11, 10 which would
 * correspond to (0, 0), (0, 1), (1, 1), (1, 0).
 *
 * @param ndim The number of dimensions.
 *
 * @return std::vector<INT> That contains the points denoting the vertices of
 * the hypercube.
 */
inline std::vector<INT> construct_initial_hypercube(const INT &ndim) {
  int total_num = 1 << ndim;
  std::vector<INT> points(total_num);

  for (int i = 0; i < total_num; i++) {
    points[i] = (i ^ (i >> 1));
  }

  return points;
}

/**
 * Function to calculate the initial function values on the vertices of the
 * hypercube.
 *
 * @param vertex_func_evals Accessor for LocalMemoryInterlaced object to fill
 * with function evaluations.
 * @param vertex_coord Accessor for LocalMemoryInterlaced object to fill with
 * locations of the vertices of the hypercube.
 * @param func_grid Pointer to a contiguous array containing the ND grid data.
 * @param hypercube_vertices Pointer to a vector containing the vertices of the
 * hypercube (integers whose binary representations give the normalised
 * positions of the vertices).
 * @param origin_indices Accessor for LocalMemoryInterlaced containing the
 * indices that will form the (0,0) point of the hypercube (that is to say, the
 * largest indices in each dimension that are still smaller than the desired
 * interpolation point).
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param ndim The number of dimensions.
 */
inline void initial_func_eval_on_device(
    Access::LocalMemoryInterlaced::Write<REAL> vertex_func_evals,
    Access::LocalMemoryInterlaced::Write<std::size_t> vertex_coord,
    REAL *func_grid, int *hypercube_vertices,
    Access::LocalMemoryInterlaced::Write<std::size_t> origin_indices,
    std::size_t *dims_vec, const int &ndim) {
  int num_points = (1 << ndim);
  int point_index = 0;
  int vertex_index = 0;

  for (int i = 0; i < (num_points * ndim); i++) {
    point_index = i / ndim;
    vertex_index = (ndim - 1) - (i % ndim);

    vertex_coord.at(vertex_index) =
        origin_indices.at(vertex_index) +
        binary_extract(hypercube_vertices[point_index], vertex_index);

    if ((ndim > 1) ? (i % ndim) : 1) {
      vertex_func_evals.at(point_index) =
          func_grid[coeff_index_on_device(vertex_coord, dims_vec, ndim)];
    }
  }
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
 */
inline void initial_func_eval_on_device(REAL *vertex_func_evals,
                                        INT *vertex_coord, REAL *func_grid,
                                        INT *hypercube_vertices,
                                        INT *origin_indices, size_t *dims_vec,
                                        const int &ndim,
                                        const int &num_points) {
  int point_index = 0;
  int vertex_index = 0;

  for (int i = 0; i < (num_points * ndim); i++) {
    point_index = i / ndim;
    vertex_index = (ndim - 1) - (i % ndim);

    vertex_coord[vertex_index] =
        origin_indices[vertex_index] +
        binary_extract(hypercube_vertices[point_index], vertex_index);

    if ((ndim > 1) ? (i % ndim) : 1) {
      vertex_func_evals[point_index] =
          func_grid[coeff_index_on_device(vertex_coord, dims_vec, ndim)];
    }
  }
}

/**
 * @brief Function to contract a hypercube down by 1 dimension via linear
 * interpolation.
 *
 * @param interp_points Accessor for LocalMemoryInterlaced object that contains
 * the interpolation points in each dimension.
 * @param dim_index Since this function is called multiple times, this counter
 * keeps track of the progress, it can be thought of as: ndim-1 where ndim is
 * the current dimensionality of the hypercube.
 * @param input_vertices Accessor for LocalMemoryInterlaced object that contains
 * the vertices of the hypercube pre-contraction.
 * @param origin_indices Accessor for LocalMemoryInterlaced containing the
 * indices that will form the (0,0) point of the hypercube (that is to say, the
 * largest indices in each dimension that are still smaller than the desired
 * interpolation point).
 * @param vertex_func_evals Accessor for LocalMemoryInterlaced object that
 * contains the function evaluations at input_vertices.
 * @param ranges_vec Pointer to a vector containing a
 * contiguous array containing the ranges of each dimension of relevance for the
 * interpolation.
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param output_vertices Accessor for LocalMemoryInterlaced object that
 * contains the vertices of the hypercube post-contraction.
 * @param output_evals Accessor for LocalMemoryInterlaced object that contains
 * the function evaluations at output_vertices.
 * @param varying_dim Accessor for LocalMemoryInterlaced object used for storing
 * the vertices whose coordinates vary in the dimension to be contracted.
 * @param vertex_coord Accessor for LocalMemoryInterlaced object used for
 * storing the vertices of the hypercube after they've been mapped to the actual
 * region in the dimensions of the grid that are of interest.
 */
inline void contract_hypercube_on_device(
    Access::LocalMemoryInterlaced::Write<REAL> interp_points,
    const int &dim_index,
    Access::LocalMemoryInterlaced::Write<int> input_vertices,
    Access::LocalMemoryInterlaced::Write<std::size_t> origin_indices,
    Access::LocalMemoryInterlaced::Write<REAL> vertex_func_evals,
    REAL *ranges_vec, std::size_t *dims_vec,
    Access::LocalMemoryInterlaced::Write<int> output_vertices,
    Access::LocalMemoryInterlaced::Write<REAL> output_evals,
    Access::LocalMemoryInterlaced::Write<std::size_t> varying_dim,
    Access::LocalMemoryInterlaced::Write<std::size_t> vertex_coord) {
  int ndim = dim_index + 1;
  int num_points = (1 << ndim);
  int num_out_points = (1 << dim_index);

  int point_index = 0;
  int eval_index = 0;

  for (int i = 0; i < (num_points * ndim); i++) {
    point_index = i / ndim;
    eval_index = (ndim - 1) - (i % ndim);

    vertex_coord.at(eval_index) =
        origin_indices.at(eval_index) +
        binary_extract(input_vertices.at(point_index), eval_index);

    if ((ndim > 1) ? (i % ndim) : 1) {
      varying_dim.at(point_index) = vertex_coord.at(dim_index);
    }
  }

  std::size_t vertex_0, vertex_1;
  REAL range_val_0, range_val_1, eval_point_0, eval_point_1;

  for (int i = 0; i < num_out_points; i++) {
    vertex_0 = varying_dim.at(i);
    vertex_1 = varying_dim.at(num_points - (i + 1));

    range_val_0 =
        ranges_vec[range_index_on_device(vertex_0, dim_index, dims_vec)];
    range_val_1 =
        ranges_vec[range_index_on_device(vertex_1, dim_index, dims_vec)];

    eval_point_0 = vertex_func_evals.at(i);
    eval_point_1 = vertex_func_evals.at(num_points - (i + 1));

    output_evals.at(i) = linear_interp(interp_points.at(dim_index), range_val_0,
                                       range_val_1, eval_point_0, eval_point_1);

    output_vertices.at(i) = input_vertices.at(i);
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
 * @param input_vertices Pointer to a vector that contains
 * the vertices of the hypercube pre-contraction.
 * @param origin_indices Pointer to a vector containing the
 * indices that will form the (0,0) point of the hypercube (that is to say, the
 * largest indices in each dimension that are still smaller than the desired
 * interpolation point).
 * @param vertex_func_evals Pointer to a vector that
 * contains the function evaluations at input_vertices.
 * @param ranges_vec Pointer to a vector containing a
 * contiguous array containing the ranges of each dimension of relevance for the
 * interpolation.
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param output_vertices Pointer to a vector that
 * contains the vertices of the hypercube post-contraction.
 * @param output_evals Pointer to a vector that contains
 * the function evaluations at output_vertices.
 * @param varying_dim Pointer to a vector used for storing
 * the vertices whose coordinates vary in the dimension to be contracted.
 * @param vertex_coord Pointer to a vector used for
 * storing the vertices of the hypercube after they've been mapped to the actual
 * region in the dimensions of the grid that are of interest.
 */
inline void contract_hypercube_on_device(
    const REAL *interp_points, const int &dim_index, INT *input_vertices,
    INT *origin_indices, REAL *vertex_func_evals, REAL *ranges_vec,
    std::size_t *dims_vec, INT *output_vertices, REAL *output_evals,
    INT *varying_dim, INT *vertex_coord) {
  int ndim = dim_index + 1;
  int num_points = (1 << ndim);
  int num_out_points = (1 << dim_index);

  int point_index = 0;
  int eval_index = 0;

  for (int i = 0; i < (num_points * ndim); i++) {
    point_index = i / ndim;
    eval_index = (ndim - 1) - (i % ndim);

    vertex_coord[eval_index] =
        origin_indices[eval_index] +
        binary_extract(input_vertices[point_index], eval_index);

    if ((ndim > 1) ? (i % ndim) : 1) {
      varying_dim[point_index] = vertex_coord[dim_index];
    }
  }

  INT vertex_0, vertex_1;
  REAL range_val_0, range_val_1, eval_point_0, eval_point_1;

  for (int i = 0; i < num_out_points; i++) {
    vertex_0 = varying_dim[i];
    vertex_1 = varying_dim[num_points - (i + 1)];

    range_val_0 =
        ranges_vec[range_index_on_device(vertex_0, dim_index, dims_vec)];
    range_val_1 =
        ranges_vec[range_index_on_device(vertex_1, dim_index, dims_vec)];

    eval_point_0 = vertex_func_evals[i];
    eval_point_1 = vertex_func_evals[num_points - (i + 1)];

    output_evals[i] = linear_interp(interp_points[dim_index], range_val_0,
                                    range_val_1, eval_point_0, eval_point_1);

    output_vertices[i] = input_vertices[i];
  }
}

} // namespace VANTAGE::Reactions::interp_utils
#endif
