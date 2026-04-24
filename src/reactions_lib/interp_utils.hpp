#ifndef REACTIONS_INTERP_UTILS_H
#define REACTIONS_INTERP_UTILS_H
#include <neso_particles.hpp>
#include <neso_particles/error_propagate.hpp>
#include <neso_particles/loop/particle_loop_index.hpp>
#include <neso_particles/typedefs.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions::interp_utils {
/**
 * Helper function that extracts the value of the binary representation of i at
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
 * @return std::size_t that specifies the index on a contiguous grid array
 */
inline INT coeff_index_on_device(INT const *indices, size_t const *dims_vec,
                                 const int &ndim) {
  INT index = indices[ndim - 1];

  for (int dimx = ndim - 2; dimx >= 0; dimx--) {
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
 * @return std::size_t that specifies the index on a contiguous ranges array.
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
 * @return std::size_t The index on a given dimension that is the closest to
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
inline REAL linear_interp(const REAL &x_interp, const REAL &x0, const REAL &x1,
                          const REAL &f0, const REAL &f1) {
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
inline std::vector<INT> construct_initial_hypercube(const INT &ndim) {
  int total_num = 1 << ndim;
  std::vector<INT> points(total_num);

  for (int i = 0; i < total_num; i++) {
    points[i] = (i ^ (i >> 1));
  }

  return points;
}

/**
 * @brief Function to bin REAL-valued (between 0.0 and 1.0) elements of an
 * input array, u, into INT-valued indices that lie between 0 and an upper limit
 * defined by the elements of dims. For example with u = {0.1, 0.7, 0.3} and
 * dims = {4, 6, 9} the output coords would be: {0, 4, 2}
 *
 * @tparam index_ndim The size of the u array, the dims array and the output
 * from the function.
 * @param u REAL-valued array of size index_ndim that contains the values
 * (between 0.0 and 1.0) that are to be converted to indices.
 * @param dims INT-valued array of size index_ndim that contains values that
 * define the upper limits for the results.
 * @return An INT-valued array of size index_ndim that contains required
 * indices.
 */
template <size_t index_ndim>
inline std::array<INT, index_ndim>
bin_uniform_indices(const std::array<REAL, index_ndim> &u,
                    const std::array<INT, index_ndim> &dims) {
  std::array<INT, index_ndim> coords;

  INT x = 0;
  for (size_t i = 0; i < index_ndim; i++) {
    x = static_cast<INT>(sycl::floor(u[i] * dims[i]));
    coords[i] = (x < dims[i]) ? x : (dims[i] - 1);
  }

  return coords;
}

/**
 * Function to calculate the initial function values on the vertices of the
 * hypercube.
 *
 * @tparam DATATYPE ReactionDataBaseOnDevice derived type corresponding to the
 * grid-function evaluation reaction data.
 * @tparam output_ndim Number of dimensions of the output of the grid-function
 * evaluation.
 * @tparam interp_ndim Number of dimensions being interpolated.
 * @tparam non_interp_ndim Number of non-interpolated dimensions (ie. dimensions
 * passed through without modification to calc_data(...)).
 * @tparam total_ndim The size of the input array to pass to calc_data(...) (ie.
 * interp_ndim + non_interp_ndim).
 * @param vertex_func_evals Pointer to a vector to fill
 * with function evaluations.
 * @param vertex_coord Pointer to a vector to fill with
 * locations of the vertices of the hypercube.
 * @param grid_func_data DATATYPE object that defines the grid-function
 * evaluation.
 * @param origin_indices Pointer to a vector containing the
 * indices that will form the (0,0) point of the hypercube (that is to say, the
 * largest indices in each dimension that are still smaller than the desired
 * interpolation point).
 * @param hypercube_vertices Pointer to a vector containing the vertices of the
 * hypercube (integers whose binary representations give the normalised
 * positions of the vertices).
 * @param ranges_vec Pointer to the flattened ranges array used to recover the
 * coordinate value for each interpolated dimension.
 * @param non_interpolation_points Values passed through without modification to
 * calc_data(...)
 * @param interpolation_indices Array of indices that correspond to the
 * dimensions that will be interpolated.
 * @param non_interpolation_indices Array of indices that correspond to the
 * dimensions that will not be interpolated.
 * @param dims_vec Pointer to a vector that contains the size of each dimension.
 * @param index Read-only accessor to a loop index for a ParticleLoop
 * inside which calc_data is called. Access using either
 * index.get_loop_linear_index(), index.get_local_linear_index(),
 * index.get_sub_linear_index() as required.
 * @param req_int_props Vector of symbols for integer-valued properties that
 * need to be used for the reaction data calculation.
 * @param req_real_props Vector of symbols for real-valued properties that
 * need to be used for the reaction data calculation.
 * @param rng_kernel The random number generator kernel potentially used in
 * the calculation
 */

template <typename DATATYPE, int output_ndim, int interp_ndim,
          int non_interp_ndim, int total_ndim>
inline void initial_func_eval_on_device(
    REAL *vertex_func_evals, INT *vertex_coord, const DATATYPE &grid_func_data,
    INT const *origin_indices, INT const *hypercube_vertices,
    REAL const *ranges_vec,
    const std::array<REAL, non_interp_ndim> &non_interpolation_points,
    const std::array<size_t, interp_ndim> &interpolation_indices,
    const std::array<size_t, non_interp_ndim> &non_interpolation_indices,
    size_t const *dims_vec, const Access::LoopIndex::Read &index,
    const Access::SymVector::Write<INT> &req_int_props,
    const Access::SymVector::Read<REAL> &req_real_props,
    typename TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>>::
        KernelType &rng_kernel) {

  std::array<REAL, output_ndim> grid_func_output;
  for (size_t idim = 0; idim < output_ndim; idim++)
    grid_func_output[idim] = 0.0;

  std::array<REAL, interp_ndim> vertex_val;
  for (size_t i = 0; i < interp_ndim; i++) {
    vertex_val[i] = 0.0;
  }

  std::array<REAL, total_ndim> grid_func_input;
  for (size_t i = 0; i < total_ndim; i++)
    grid_func_input[i] = 0.0;

  size_t num_points = 1 << interp_ndim;
  for (size_t point_index = 0; point_index < num_points; point_index++) {
    for (size_t i = 0; i < total_ndim; i++)
      grid_func_input[i] = 0.0;
    for (size_t vertex_index = 0; vertex_index < interp_ndim; vertex_index++) {
      vertex_coord[vertex_index] =
          origin_indices[vertex_index] +
          binary_extract(hypercube_vertices[point_index], vertex_index);
      vertex_val[vertex_index] = ranges_vec[range_index_on_device(
          vertex_coord[vertex_index], vertex_index, dims_vec)];

      grid_func_input[interpolation_indices[vertex_index]] =
          vertex_val[vertex_index];
    }

    for (size_t i = 0; i < non_interp_ndim; i++) {
      grid_func_input[non_interpolation_indices[i]] =
          non_interpolation_points[i];
    }

    grid_func_output =
        grid_func_data.calc_data(grid_func_input, index, req_int_props,
                                 req_real_props, rng_kernel.template get<0>());

    for (size_t idim = 0; idim < output_ndim; idim++) {
      vertex_func_evals[(point_index * output_ndim) + idim] =
          grid_func_output[idim];
    }
  }
}

/**
 * @brief Function to contract a hypercube down by 1 dimension via linear
 * interpolation.
 *
 * @tparam output_ndim Number of output dimensions of output_evals.
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
template <int output_ndim>
inline void contract_hypercube_on_device(
    const REAL *interp_points, const size_t &dim_index,
    INT const *hypercube_vertices, const INT *origin_indices,
    const REAL *vertex_func_evals, REAL const *ranges_vec,
    size_t const *dims_vec, REAL *output_evals, INT *varying_dim,
    INT *vertex_coord) {
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

  INT vertex_0, vertex_1;
  REAL range_val_0, range_val_1, eval_point_0, eval_point_1;

  for (size_t i = 0; i < num_out_points; i++) {
    INT index_0 = i;
    INT index_1 = num_points - (i + 1);

    vertex_0 = varying_dim[index_0];
    vertex_1 = varying_dim[index_1];

    range_val_0 = // x0
        ranges_vec[range_index_on_device(vertex_0, dim_index, dims_vec)];
    range_val_1 = // x1
        ranges_vec[range_index_on_device(vertex_1, dim_index, dims_vec)];

    for (size_t idim = 0; idim < output_ndim; idim++) {
      eval_point_0 = vertex_func_evals[(index_0 * output_ndim) + idim]; // f0
      eval_point_1 = vertex_func_evals[(index_1 * output_ndim) + idim]; // f1

      output_evals[(index_0 * output_ndim) + idim] =
          linear_interp(interp_points[dim_index], range_val_0, range_val_1,
                        eval_point_0, eval_point_1);
    }
  }
}

} // namespace VANTAGE::Reactions::interp_utils
#endif
