#ifndef REACTIONS_INTERPOLATE_DATA_H
#define REACTIONS_INTERPOLATE_DATA_H
#include "../reaction_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

#define INF std::numeric_limits<double>::infinity()

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
/**
 * @tparam EXTRAPOLATION_TYPE Possible values are [0, 1, 2]. For 0, the
 * extrapolation continues the linear interpolation from the edges of the range.
 * For 1, the extrapolation clamps the result to 0. For 2, the extrapolation
 * clamps the value to the last value in the range (effectively flattens the
 * grid out beyond the edges of the dimension).
 */
template <size_t ndim, INT EXTRAPOLATION_TYPE = 0>
struct InterpolateDataOnDevice : public ReactionDataBaseOnDevice<ndim> {
  InterpolateDataOnDevice() = default;

  std::array<REAL, ndim>
  calc_data(const std::array<REAL, ndim> &interpolation_points,
            const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice<ndim>::RNG_KERNEL_TYPE::KernelType
                &kernel) const {
    // Temporary arrays for intermediate calculation (in principle will hold
    // unique values for each particle)
    std::array<INT, ndim> origin_indices;
    origin_indices.fill(0);

    std::array<REAL, initial_num_points> vertex_func_evals;
    std::array<INT, initial_num_points> vertex_coord;
    vertex_func_evals.fill(0.0);
    vertex_coord.fill(0);

    std::array<INT, initial_num_points> input_vertices;
    std::array<INT, initial_num_points> output_vertices;
    std::array<REAL, initial_num_points> output_evals;
    input_vertices.fill(0);
    output_vertices.fill(0);
    output_evals.fill(0.0);

    std::array<INT, initial_num_points> varying_dim;
    varying_dim.fill(0);

    // Counters
    int dim_index = ndim - 1;
    int num_points = this->initial_num_points;
    auto current_count = index.get_loop_linear_index();

    // This seems to be necessary to correctly retrieve the values in the
    // BufferDevice ptrs
    auto hypercube_vertices_buf = this->hypercube_vertices_ptr;
    auto ranges_vec_buf = this->ranges_vec_ptr;
    auto dims_vec_buf = this->dims_vec_ptr;
    auto grid_buf = this->grid_ptr;
    auto extended_ranges_vec_buf = this->extended_ranges_vec_ptr;

    // Calculation of the indices that will form the "origin" of the
    // hypercube. These are the smallest indices in each dimension that
    // still have coordinate values that are less than the interpolation
    // values in that dimension.
    origin_indices[0] = interp_utils::calc_closest_point_index(
        interpolation_points[0], extended_ranges_vec_buf,
        (dims_vec_buf[0] + 1));
    for (int i = 1; i < ndim; i++) {
      origin_indices[i] = interp_utils::calc_closest_point_index(
          interpolation_points[i],
          extended_ranges_vec_buf + (dims_vec_buf[i - 1] + 2),
          (dims_vec_buf[i] + 1));
    }

    bool constexpr continue_last = (EXTRAPOLATION_TYPE == 0) ? true : false;
    bool constexpr clamp_to_zero = (EXTRAPOLATION_TYPE == 1) ? true : false;

    // TODO: Implement clamp_to_last (currently will return the same value as clamp_to_zero)
    bool constexpr clamp_to_last = (EXTRAPOLATION_TYPE == 2) ? true : false;

    if constexpr (continue_last) {
      for (int i = 0; i < ndim; i++) {
        origin_indices[i] =
            Kernel::min(Kernel::max(origin_indices[i], 1), dims_vec_buf[i] - 1);
      }
    }

    if constexpr (!clamp_to_zero && !clamp_to_last) {
      // Necessary for using the interp_utils functions.
      auto interpolation_points_ptr = interpolation_points.data();
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
          hypercube_vertices_buf, origin_indices_ptr, dims_vec_buf, ndim,
          num_points);

      // Fill input_vertices vector
      for (int i = 0; i < num_points; i++) {
        input_vertices[i] = hypercube_vertices_buf[i];
      }

      // Loop until the last dimension (down to 0D)
      while (dim_index >= 0) {
        // Contract the hypercube vertices and evaluations, eg. if
        // input_vertices.size() == 2^3 then output_vertices.size() == 2^2, as
        // in going from a 3D hypercube(cube) to a 2D hypercube(square).
        // Additionally input_evals and output_evals are scaled down in the
        // same way via linear interpolation.
        interp_utils::contract_hypercube_on_device(
            interpolation_points_ptr, dim_index, input_vertices_ptr,
            origin_indices_ptr, vertex_func_evals_ptr, ranges_vec_buf,
            dims_vec_buf, output_vertices_ptr, output_evals_ptr,
            varying_dim_ptr, vertex_coord_ptr);

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
    }

    std::array<REAL, ndim> calculated_interpolated_vals;
    calculated_interpolated_vals.fill(0.0);

    // Assign to first element of returned array. (See
    // test/unit/test_interpolation.cpp for reasons for the workaround.)
    calculated_interpolated_vals[0] = output_evals[0];

    return calculated_interpolated_vals;
  }

public:
  INT *hypercube_vertices_ptr;
  size_t *dims_vec_ptr;
  REAL *ranges_vec_ptr;
  REAL *grid_ptr;
  REAL *extended_ranges_vec_ptr;

  static constexpr INT initial_num_points = 1 << ndim;
};

template <size_t ndim, INT EXTRAPOLATION_TYPE = 0>
struct InterpolateData
    : public ReactionDataBase<InterpolateDataOnDevice<ndim>, ndim> {
  InterpolateData(const std::vector<size_t> &dims_vec,
                  const std::vector<REAL> &ranges_vec,
                  const std::vector<REAL> &grid,
                  SYCLTargetSharedPtr sycl_target)
      : ReactionDataBase<InterpolateDataOnDevice<ndim, EXTRAPOLATION_TYPE>,
                         ndim>() {
    if constexpr ((EXTRAPOLATION_TYPE < 0) || (EXTRAPOLATION_TYPE > 2)) {
      NESOASSERT(false,
                 "Please pass a valid EXTRAPOLATION_TYPE (either 0, 1 or 2) as "
                 "template to the InterpolateData constructor.");
    }

    auto initial_hypercube =
        interp_utils::construct_initial_hypercube(INT(ndim));

    this->on_device_obj = InterpolateDataOnDevice<ndim>();

    // BufferDevice<REAL> mock setup
    this->dims_vec_buf =
        std::make_shared<BufferDevice<size_t>>(sycl_target, dims_vec);
    this->on_device_obj->dims_vec_ptr = this->dims_vec_buf->ptr;

    std::vector<size_t> extended_dims_vec(ndim);
    for (size_t idim = 0; idim < ndim; idim++) {
      extended_dims_vec[idim] = dims_vec[idim] + 2;
    }

    std::vector<REAL> extended_ranges_vec;
    for (size_t idim = 0; idim < ndim; idim++) {
      int dim_counter = 0;
      extended_ranges_vec.push_back(-INF);
      for (size_t irange = 0; irange < dims_vec[idim]; irange++) {
        extended_ranges_vec.push_back(ranges_vec[irange]);
      }
      extended_ranges_vec.push_back(INF);
      dim_counter += extended_dims_vec[idim];
    }

    this->extended_ranges_vec_buf =
        std::make_shared<BufferDevice<REAL>>(sycl_target, extended_ranges_vec);
    this->on_device_obj->extended_ranges_vec_ptr =
        this->extended_ranges_vec_buf->ptr;

    this->ranges_vec_buf =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges_vec);
    this->on_device_obj->ranges_vec_ptr = this->ranges_vec_buf->ptr;

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
  std::shared_ptr<BufferDevice<REAL>> grid_buf;
  std::shared_ptr<BufferDevice<INT>> hypercube_vertices_buf;
};
}; // namespace VANTAGE::Reactions
#endif
