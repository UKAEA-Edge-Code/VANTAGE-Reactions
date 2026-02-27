/**
 * InterpolateData and InterpolateDataOnDevice, correspond to class definitions
 * corresponding to implementations of an interpolation algorithm. A description
 * of the algorithm can be found here:
 * https://www.nas.nasa.gov/assets/nas/pdf/staff/Murman_S_apnum_jun13.pdf
 * Specifically Figure 2 provides a good illustration of how the algorithm
 * works.
 *
 * The interface is primarily meant to be through the construction of the
 * InterpolateData object and the usage of calc_data from the
 * InterpolateDataOnDevice object where appropriate. The construction of the
 * InterpolateData object provides most of the information that is needed.
 *
 * 3 primary components are needed: dims_vec, ranges_vec and grid.
 * - dims_vec: 1D size_t vector of length ndim that specifies the number of
 * points of each of the dimensions of grid.
 * - ranges_vec: 1D REAL flattened vector that contains the range of values for
 * each dimension of grid.
 * - grid: 1D REAL flattened vector that contains all of
 * the pre-computed function evaluations typically loaded in from an external
 * source like ADAS (pre-processing from ND to 1D prior to passing to
 * InterpolateData is currently left to the user). Optionally it is possible for
 * the user to define how they wish to deal with extrapolations (by passing one
 * of the options defined in ExtrapolationType).
 *
 * The on-host object construction mostly just sets up the necessary data for
 * the on-device object. There is an additionaly processing step to calculate
 * the initial hypercube. This is just a series of vertices of an N-Dimensional
 * hypercube (where N is the number of dimensions of the grid that's suppplied).
 * Note that the hypercube vertices produced don't actually have any direct
 * mapping onto the values of the dimensions of the grid (ie. the x-values or
 * y-values), rather it can be thought of as a stencil to be used with the
 * origin_indices array that's calculated in InterpolateDataOnDevice.calc_data.
 * More details on the construct_initial_hypercube can be found in the docstring
 * in interp_utils.hpp.
 *
 * In calc_data in the InterpolateDataOnDevice class, firstly, there's a good
 * deal of setup that creates the temporary arrays used for storing intermediate
 * values.
 *
 * Next, the origin_indices array is filled. This array contains the indices
 * that form the "origin" of the hypercube that will be used for the
 * interpolation. For example, if the interpolation points are x=3.7 and y=4.9
 * and the grid dimensions are defined as running from 0-10 with a spacing of 1
 * in both x and y then the origin indices will be 3, 4 and will be used in
 * conjunction with the d_hypercube_vertices to construct a 2D hypercube(square)
 * that has vertex coordinates of (3, 4), (3, 5), (4, 5), (4, 4) in that order.
 * Note that the actual ranges for the dimensions used in this calculation are
 * the "extended" versions which are padded with -INF_DOUBLE below their lower
 * bounds and +INF_DOUBLE above their upper bounds. This is for the sake of
 * aiding in extrapolation handling and is reset after extrapolation handling.
 *
 * Processing related to extrapolation scenarios is performed next, the exact
 * behaviour will depend on the user choice at the construction of the
 * InterpolateData object.
 *
 * Now that the origin indices for the hypercube in question have been
 * calculated, the next step is to calculate the function values at each of
 * those vertices with initial_func_eval_on_device. The results are stored in
 * vertex_func_evals. The coordinates of the vertices (origin_indices combined
 * with the hypercube vertices stencil) are also stored in vertex_coords.
 *
 * Finally a loop is performed that counts down from the number of dimensions
 * down to 0D and contracts the hypercube that's defined by vertex_coords and
 * vertex_func_evals.
 */

#ifndef REACTIONS_INTERPOLATE_DATA_H
#define REACTIONS_INTERPOLATE_DATA_H
#include "../reaction_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
constexpr auto INF_INTERP_DOUBLE = std::numeric_limits<double>::infinity();

enum class ExtrapolationType { continue_linear, clamp_to_zero, clamp_to_edge };

template <size_t input_ndim>
struct InterpolateDataOnDevice
    : public ReactionDataBaseOnDevice<1, DEFAULT_RNG_KERNEL, input_ndim> {
  InterpolateDataOnDevice(ExtrapolationType extrapolation_type =
                              ExtrapolationType::continue_linear) {
    switch (extrapolation_type) {
    case VANTAGE::Reactions::ExtrapolationType::continue_linear:
      this->continue_linear = true;
      break;
    case VANTAGE::Reactions::ExtrapolationType::clamp_to_zero:
      this->clamp_to_zero = true;
      break;
    case VANTAGE::Reactions::ExtrapolationType::clamp_to_edge:
      this->clamp_to_edge = true;
      break;
    }
  };

  std::array<REAL, 1> calc_data(
      const std::array<REAL, input_ndim> &interpolation_points,
      const Access::LoopIndex::Read &index,
      [[maybe_unused]] const Access::SymVector::Write<INT> &req_int_props,
      [[maybe_unused]] const Access::SymVector::Read<REAL> &req_real_props,
      typename ReactionDataBaseOnDevice<1, DEFAULT_RNG_KERNEL,
                                        input_ndim>::RNG_KERNEL_TYPE::KernelType
          &kernel) const {
    // Temporary arrays for intermediate calculation (in principle will hold
    // unique values for each particle)
    std::array<REAL, input_ndim> mut_interpolation_points =
        interpolation_points;
    std::array<INT, input_ndim> origin_indices;

    std::array<REAL, initial_num_points> vertex_func_evals;
    std::array<INT, initial_num_points> vertex_coord;

    std::array<REAL, initial_num_points> output_evals;
    std::array<INT, initial_num_points> varying_dim;

    for (size_t i = 0; i < input_ndim; i++) {
      origin_indices[i] = 0;
    }

    for (size_t i = 0; i < initial_num_points; i++) {
      vertex_func_evals[i] = 0.0;
      vertex_coord[i] = 0;

      output_evals[i] = 0.0;
      varying_dim[i] = 0;
    }

    // Counter
    INT num_points = this->initial_num_points;

    // Array of length 1 to maintain compatibility with pipelining interface for
    // ReactionData objects.
    std::array<REAL, 1> calculated_interpolated_vals;
    calculated_interpolated_vals[0] = 0.0;

    // Calculation of the indices that will form the "origin" of the
    // hypercube. These are the smallest indices in each dimension that
    // still have coordinate values that are less than the interpolation
    // values in that dimension.
    for (size_t i = 0; i < input_ndim; i++) {
      origin_indices[i] = interp_utils::calc_floor_point_index(
          mut_interpolation_points[i],
          this->d_extended_ranges_vec + this->d_extended_ranges_strides[i],
          this->d_extended_dims_vec[i] - 1);
    }

    // Out-of-range clamping handling
    bool out_of_range = false;
    bool above_range = false;
    bool below_range = false;

    REAL above_clamp_to_edge = 0.0;
    REAL below_clamp_to_edge = 0.0;

    bool out_of_range_clamp_to_zero = false;

    for (size_t i = 0; i < input_ndim; i++) {
      above_range = (origin_indices[i] == (this->d_extended_dims_vec[i] - 2));
      below_range = (origin_indices[i] == 0);

      out_of_range = (above_range || below_range);

      above_clamp_to_edge = this->d_ranges_vec[this->d_dims_vec[i] - 1 +
                                               this->d_ranges_strides[i]];
      below_clamp_to_edge = this->d_ranges_vec[this->d_ranges_strides[i]];

      mut_interpolation_points[i] = (above_range && this->clamp_to_edge)
                                        ? above_clamp_to_edge
                                        : ((below_range && this->clamp_to_edge)
                                               ? below_clamp_to_edge
                                               : mut_interpolation_points[i]);

      out_of_range_clamp_to_zero = (out_of_range && this->clamp_to_zero);
    }

    // Limit origin_indices to be between the standard dimensional ranges.
    // Note that the upper limit is set by this->d_dims_vec[i] - 2 since that
    // represents the penultimate element in the standard dimensional range
    // which is the last left-most index that can be selected such that the
    // linear gradient can be calculated.
    for (size_t i = 0; i < input_ndim; i++) {
      origin_indices[i]--;
      origin_indices[i] = Kernel::min(Kernel::max(origin_indices[i], 0),
                                      this->d_dims_vec[i] - 2);
    }

    // Necessary for using the interp_utils functions.
    auto mut_interpolation_points_ptr = mut_interpolation_points.data();
    auto origin_indices_ptr = origin_indices.data();

    auto vertex_func_evals_ptr = vertex_func_evals.data();
    auto vertex_coord_ptr = vertex_coord.data();

    auto output_evals_ptr = output_evals.data();
    auto varying_dim_ptr = varying_dim.data();

    // Initial function evaluation (ie values of the coeffs_vec) based on
    // the vertices of the hypercube.
    interp_utils::initial_func_eval_on_device(
        vertex_func_evals_ptr, vertex_coord_ptr, this->d_grid,
        this->d_hypercube_vertices, origin_indices_ptr, this->d_dims_vec,
        input_ndim, num_points);

    // Loop until the last dimension (down to 0D)
    // Note that despite the dim_index = input_ndim assignment, the loop
    // actually starts at dim_index = (input_ndim - 1) , as desired, due to the
    // decrement and store during the first check against 0.
    for (size_t dim_index = input_ndim; dim_index-- > 0;) {
      // Contract the hypercube vertices and evaluations by performing linear
      // interpolation on the current dimension (denoted by dim_index). The
      // contraction is expressed by the reduction in the length of the
      // output_evals (ie. for 3D to 2D, the vertex_func_evals would be of
      // length 8 and output_evals would be of length 4).
      interp_utils::contract_hypercube_on_device(
          mut_interpolation_points_ptr, dim_index, this->d_hypercube_vertices,
          origin_indices_ptr, vertex_func_evals_ptr, this->d_ranges_vec,
          this->d_dims_vec, output_evals_ptr, varying_dim_ptr,
          vertex_coord_ptr);

      // This now accounts for the smaller size of output_evals, and makes sure
      // that any loops in future contract_hypercube(...) invocations remain
      // consistent.
      num_points = num_points >> 1;

      // Reset vertex_func_evals for the next contraction
      for (int i = 0; i < num_points; i++) {
        vertex_func_evals[i] = output_evals[i];
      }
    }

    calculated_interpolated_vals[0] = out_of_range_clamp_to_zero
                                          ? calculated_interpolated_vals[0]
                                          : vertex_func_evals[0];

    return calculated_interpolated_vals;
  }

public:
  INT *d_hypercube_vertices;
  size_t *d_dims_vec;
  REAL *d_ranges_vec;
  REAL *d_grid;
  REAL *d_extended_ranges_vec;
  size_t *d_extended_dims_vec;
  size_t *d_ranges_strides;
  size_t *d_extended_ranges_strides;

  static constexpr INT initial_num_points = 1 << input_ndim;

  bool continue_linear = false;
  bool clamp_to_zero = false;
  bool clamp_to_edge = false;
};

template <size_t input_ndim>
struct InterpolateData
    : public ReactionDataBase<InterpolateDataOnDevice<input_ndim>, 1,
                              DEFAULT_RNG_KERNEL, input_ndim> {
  InterpolateData(
      const std::vector<size_t> &dims_vec, const std::vector<REAL> &ranges_vec,
      const std::vector<REAL> &grid, SYCLTargetSharedPtr sycl_target,
      ExtrapolationType extrapolation_type = ExtrapolationType::continue_linear)
      : ReactionDataBase<InterpolateDataOnDevice<input_ndim>, 1,
                         DEFAULT_RNG_KERNEL, input_ndim>() {
    auto initial_hypercube =
        interp_utils::construct_initial_hypercube(input_ndim);

    this->on_device_obj =
        InterpolateDataOnDevice<input_ndim>(extrapolation_type);

    // BufferDevice<REAL> mock setup
    this->h_dims_vec =
        std::make_shared<BufferDevice<size_t>>(sycl_target, dims_vec);
    this->on_device_obj->d_dims_vec = this->h_dims_vec->ptr;

    std::vector<size_t> ranges_strides(input_ndim);
    std::vector<size_t> extended_dims_vec(input_ndim);
    std::vector<size_t> extended_ranges_strides(input_ndim);
    for (size_t idim = 0; idim < input_ndim; idim++) {
      extended_dims_vec[idim] = dims_vec[idim] + 2;
      for (size_t j = 0; j < idim; j++) {
        ranges_strides[idim] += dims_vec[j];
        extended_ranges_strides[idim] += extended_dims_vec[j];
      }
    }

    std::vector<REAL> extended_ranges_vec;
    for (size_t idim = 0; idim < input_ndim; idim++) {
      extended_ranges_vec.push_back(-INF_INTERP_DOUBLE);
      for (size_t irange = 0; irange < dims_vec[idim]; irange++) {
        extended_ranges_vec.push_back(
            ranges_vec[irange + ranges_strides[idim]]);
      }
      extended_ranges_vec.push_back(INF_INTERP_DOUBLE);
    }

    this->h_extended_ranges_vec =
        std::make_shared<BufferDevice<REAL>>(sycl_target, extended_ranges_vec);
    this->on_device_obj->d_extended_ranges_vec =
        this->h_extended_ranges_vec->ptr;

    this->h_extended_dims_vec =
        std::make_shared<BufferDevice<size_t>>(sycl_target, extended_dims_vec);
    this->on_device_obj->d_extended_dims_vec = this->h_extended_dims_vec->ptr;

    this->h_ranges_vec =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges_vec);
    this->on_device_obj->d_ranges_vec = this->h_ranges_vec->ptr;

    this->h_ranges_strides =
        std::make_shared<BufferDevice<size_t>>(sycl_target, ranges_strides);
    this->on_device_obj->d_ranges_strides = this->h_ranges_strides->ptr;

    this->h_extended_ranges_strides = std::make_shared<BufferDevice<size_t>>(
        sycl_target, extended_ranges_strides);
    this->on_device_obj->d_extended_ranges_strides =
        this->h_extended_ranges_strides->ptr;

    this->h_grid = std::make_shared<BufferDevice<REAL>>(sycl_target, grid);
    this->on_device_obj->d_grid = this->h_grid->ptr;

    this->h_hypercube_vertices =
        std::make_shared<BufferDevice<INT>>(sycl_target, initial_hypercube);
    this->on_device_obj->d_hypercube_vertices = this->h_hypercube_vertices->ptr;
  };

  std::shared_ptr<BufferDevice<size_t>> h_dims_vec;
  std::shared_ptr<BufferDevice<REAL>> h_ranges_vec;
  std::shared_ptr<BufferDevice<REAL>> h_extended_ranges_vec;
  std::shared_ptr<BufferDevice<size_t>> h_extended_dims_vec;
  std::shared_ptr<BufferDevice<size_t>> h_ranges_strides;
  std::shared_ptr<BufferDevice<size_t>> h_extended_ranges_strides;
  std::shared_ptr<BufferDevice<REAL>> h_grid;
  std::shared_ptr<BufferDevice<INT>> h_hypercube_vertices;
};
}; // namespace VANTAGE::Reactions
#endif
