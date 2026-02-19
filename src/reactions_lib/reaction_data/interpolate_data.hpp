#ifndef REACTIONS_INTERPOLATE_DATA_H
#define REACTIONS_INTERPOLATE_DATA_H
#include "../reaction_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/typedefs.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
constexpr auto INF_DOUBLE = std::numeric_limits<double>::infinity();
/**
 * @tparam EXTRAPOLATION_TYPE Possible values are [0, 1, 2]. For 0, the
 * extrapolation continues the linear interpolation from the edges of the range.
 * For 1, the extrapolation clamps the result to 0. For 2, the extrapolation
 * clamps the value to the last value in the range (effectively flattens the
 * grid out beyond the edges of the dimension).
 */
template <size_t input_ndim, INT EXTRAPOLATION_TYPE = 0, size_t output_ndim = 1>
struct InterpolateDataOnDevice
    : public ReactionDataBaseOnDevice<output_ndim, DEFAULT_RNG_KERNEL,
                                      input_ndim> {
  InterpolateDataOnDevice() = default;

  std::array<REAL, output_ndim> calc_data(
      const std::array<REAL, input_ndim> &interpolation_points,
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Read<REAL> &req_real_props,
      typename ReactionDataBaseOnDevice<output_ndim, DEFAULT_RNG_KERNEL,
                                        input_ndim>::RNG_KERNEL_TYPE::KernelType
          &kernel) const {
    // Temporary arrays for intermediate calculation (in principle will hold
    // unique values for each particle)
    std::array<REAL, input_ndim> mut_interpolation_points =
        interpolation_points;
    std::array<INT, input_ndim> origin_indices;

    std::array<REAL, initial_num_points> vertex_func_evals;
    std::array<INT, initial_num_points> vertex_coord;

    std::array<INT, initial_num_points> input_vertices;
    std::array<INT, initial_num_points> output_vertices;
    std::array<REAL, initial_num_points> output_evals;

    std::array<INT, initial_num_points> varying_dim;

    for (size_t i = 0; i < input_ndim; i++) {
      origin_indices[i] = 0;
    }

    for (size_t i = 0; i < initial_num_points; i++) {
      vertex_func_evals[i] = 0.0;
      vertex_coord[i] = 0;

      input_vertices[i] = 0;
      output_vertices[i] = 0;
      output_evals[i] = 0.0;

      varying_dim[i] = 0;
    }

    // Counters
    int dim_index = input_ndim - 1;
    int num_points = this->initial_num_points;
    auto current_count = index.get_loop_linear_index();

    // This seems to be necessary to correctly retrieve the values in the
    // BufferDevice ptrs
    auto hypercube_vertices_buf = this->hypercube_vertices_ptr;
    auto ranges_vec_buf = this->ranges_vec_ptr;
    auto dims_vec_buf = this->dims_vec_ptr;
    auto grid_buf = this->grid_ptr;
    auto extended_ranges_vec_buf = this->extended_ranges_vec_ptr;
    auto extended_dims_vec_buf = this->extended_dims_vec_ptr;
    auto ranges_strides_buf = this->ranges_strides_ptr;
    auto extended_ranges_strides_buf = this->extended_ranges_strides_ptr;

    std::array<REAL, output_ndim> calculated_interpolated_vals;

    for (size_t i = 0; i < output_ndim; i++) {
      calculated_interpolated_vals[i] = 0.0;
    }

    // Calculation of the indices that will form the "origin" of the
    // hypercube. These are the smallest indices in each dimension that
    // still have coordinate values that are less than the interpolation
    // values in that dimension.
    for (int i = 0; i < input_ndim; i++) {
      origin_indices[i] = interp_utils::calc_closest_point_index(
          mut_interpolation_points[i],
          extended_ranges_vec_buf + extended_ranges_strides_buf[i],
          (dims_vec_buf[i] + 1));
    }

    bool constexpr continue_last = (EXTRAPOLATION_TYPE == 0) ? true : false;
    bool constexpr clamp_to_zero = (EXTRAPOLATION_TYPE == 1) ? true : false;
    bool constexpr clamp_to_last = (EXTRAPOLATION_TYPE == 2) ? true : false;

    // Out-of-range clamping handling
    bool out_of_range = false;
    for (int i = 0; i < input_ndim; i++) {
      // check for out-of-range beyond the upper limit of the given dimension.
      if (origin_indices[i] == (extended_dims_vec_buf[i] - 2)) {
        out_of_range = true;
        if constexpr (clamp_to_last) {
          mut_interpolation_points[i] =
              (ranges_vec_buf + ranges_strides_buf[i])[dims_vec_buf[i] - 1];
        }
      }
      // check for out-of-range below the lower limit of the given dimension.
      else if (origin_indices[i] == 0) {
        out_of_range = true;
        if constexpr (clamp_to_last) {
          mut_interpolation_points[i] =
              (ranges_vec_buf + ranges_strides_buf[i])[0];
        }
      }
      if (out_of_range) {
        // Handles special case of clamp_to_zero - simply returns
        // zero-filled interpolated values if any interpolation point is out
        // of the range of allowed points for its dimension.
        if constexpr (clamp_to_zero) {
          return calculated_interpolated_vals;
        }
      }
    }

    // Limit origin_indices to be between the standard dimensional ranges.
    for (int i = 0; i < input_ndim; i++) {
      origin_indices[i]--;
      origin_indices[i] =
          Kernel::min(Kernel::max(origin_indices[i], 0), dims_vec_buf[i] - 2);
    }

    // Necessary for using the interp_utils functions.
    auto mut_interpolation_points_ptr = mut_interpolation_points.data();
    auto origin_indices_ptr = origin_indices.data();

    auto vertex_func_evals_ptr = vertex_func_evals.data();
    auto vertex_coord_ptr = vertex_coord.data();

    auto input_vertices_ptr = input_vertices.data();
    auto output_vertices_ptr = output_vertices.data();
    auto output_evals_ptr = output_evals.data();

    auto varying_dim_ptr = varying_dim.data();

    // Initial function evaluation (ie values of the coeffs_vec) based on
    // the vertices of the hypercube.
    interp_utils::initial_func_eval_on_device(
        vertex_func_evals_ptr, vertex_coord_ptr, grid_buf,
        hypercube_vertices_buf, origin_indices_ptr, dims_vec_buf, input_ndim,
        num_points);

    // Fill input_vertices vector
    for (int i = 0; i < num_points; i++) {
      input_vertices[i] = hypercube_vertices_buf[i];
    }

    // Loop until the last dimension (down to 0D)
    while (dim_index >= static_cast<int>(output_ndim - 1)) {
      // Contract the hypercube vertices and evaluations, eg. if
      // input_vertices.size() == 2^3 then output_vertices.size() == 2^2, as
      // in going from a 3D hypercube(cube) to a 2D hypercube(square).
      interp_utils::contract_hypercube_on_device(
          mut_interpolation_points_ptr, dim_index, input_vertices_ptr,
          origin_indices_ptr, vertex_func_evals_ptr, ranges_vec_buf,
          dims_vec_buf, output_vertices_ptr, output_evals_ptr, varying_dim_ptr,
          vertex_coord_ptr);

      // This now accounts for the smaller size of output_vertices and
      // output_evals, and makes sure that any loops in future
      // contract_hypercube(...) invocations remain consistent.
      num_points = num_points >> 1;
      dim_index--;

      // This resets the input_vertices and vertex_func_evals
      for (int i = 0; i < num_points; i++) {
        input_vertices[i] = output_vertices[i];
        vertex_func_evals[i] = output_evals[i];
      }
    }

    for (int i = 0; i < output_ndim; i++) {
      calculated_interpolated_vals[i] = output_evals[i];
    }

    return calculated_interpolated_vals;
  }

public:
  INT *hypercube_vertices_ptr;
  size_t *dims_vec_ptr;
  REAL *ranges_vec_ptr;
  REAL *grid_ptr;
  REAL *extended_ranges_vec_ptr;
  size_t *extended_dims_vec_ptr;
  size_t *ranges_strides_ptr;
  size_t *extended_ranges_strides_ptr;

  static constexpr INT initial_num_points = 1 << input_ndim;
};

template <size_t input_ndim, INT EXTRAPOLATION_TYPE = 0, size_t output_ndim = 1>
struct InterpolateData
    : public ReactionDataBase<
          InterpolateDataOnDevice<input_ndim, EXTRAPOLATION_TYPE, output_ndim>,
          output_ndim, DEFAULT_RNG_KERNEL, input_ndim> {
  InterpolateData(const std::vector<size_t> &dims_vec,
                  const std::vector<REAL> &ranges_vec,
                  const std::vector<REAL> &grid,
                  SYCLTargetSharedPtr sycl_target)
      : ReactionDataBase<InterpolateDataOnDevice<input_ndim, EXTRAPOLATION_TYPE,
                                                 output_ndim>,
                         output_ndim, DEFAULT_RNG_KERNEL, input_ndim>() {
    if constexpr ((EXTRAPOLATION_TYPE < 0) || (EXTRAPOLATION_TYPE > 2)) {
      NESOASSERT(false,
                 "Please pass a valid EXTRAPOLATION_TYPE (either 0, 1 or 2) as "
                 "template to the InterpolateData constructor.");
    }

    auto initial_hypercube =
        interp_utils::construct_initial_hypercube(INT(input_ndim));

    this->on_device_obj =
        InterpolateDataOnDevice<input_ndim, EXTRAPOLATION_TYPE, output_ndim>();

    // BufferDevice<REAL> mock setup
    this->dims_vec_buf =
        std::make_shared<BufferDevice<size_t>>(sycl_target, dims_vec);
    this->on_device_obj->dims_vec_ptr = this->dims_vec_buf->ptr;

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
      extended_ranges_vec.push_back(-INF_DOUBLE);
      for (size_t irange = 0; irange < dims_vec[idim]; irange++) {
        extended_ranges_vec.push_back(
            ranges_vec[irange + ranges_strides[idim]]);
      }
      extended_ranges_vec.push_back(INF_DOUBLE);
    }

    this->extended_ranges_vec_buf =
        std::make_shared<BufferDevice<REAL>>(sycl_target, extended_ranges_vec);
    this->on_device_obj->extended_ranges_vec_ptr =
        this->extended_ranges_vec_buf->ptr;

    this->extended_dims_vec_buf =
        std::make_shared<BufferDevice<size_t>>(sycl_target, extended_dims_vec);
    this->on_device_obj->extended_dims_vec_ptr =
        this->extended_dims_vec_buf->ptr;

    this->ranges_vec_buf =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges_vec);
    this->on_device_obj->ranges_vec_ptr = this->ranges_vec_buf->ptr;

    this->ranges_strides_buf =
        std::make_shared<BufferDevice<size_t>>(sycl_target, ranges_strides);
    this->on_device_obj->ranges_strides_ptr = this->ranges_strides_buf->ptr;

    this->extended_ranges_strides_buf = std::make_shared<BufferDevice<size_t>>(
        sycl_target, extended_ranges_strides);
    this->on_device_obj->extended_ranges_strides_ptr =
        this->extended_ranges_strides_buf->ptr;

    this->grid_buf = std::make_shared<BufferDevice<REAL>>(sycl_target, grid);
    this->on_device_obj->grid_ptr = this->grid_buf->ptr;

    this->hypercube_vertices_buf =
        std::make_shared<BufferDevice<INT>>(sycl_target, initial_hypercube);
    this->on_device_obj->hypercube_vertices_ptr =
        this->hypercube_vertices_buf->ptr;
  };

  std::shared_ptr<BufferDevice<size_t>> dims_vec_buf;
  std::shared_ptr<BufferDevice<REAL>> ranges_vec_buf;
  std::shared_ptr<BufferDevice<REAL>> extended_ranges_vec_buf;
  std::shared_ptr<BufferDevice<size_t>> extended_dims_vec_buf;
  std::shared_ptr<BufferDevice<size_t>> ranges_strides_buf;
  std::shared_ptr<BufferDevice<size_t>> extended_ranges_strides_buf;
  std::shared_ptr<BufferDevice<REAL>> grid_buf;
  std::shared_ptr<BufferDevice<INT>> hypercube_vertices_buf;
};
}; // namespace VANTAGE::Reactions
#endif
