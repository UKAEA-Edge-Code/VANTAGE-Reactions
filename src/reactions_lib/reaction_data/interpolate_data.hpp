#ifndef REACTIONS_INTERPOLATE_DATA_H
#define REACTIONS_INTERPOLATE_DATA_H
#include "../reaction_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/compute_target.hpp>
#include <neso_particles/containers/local_memory_interlaced.hpp>
#include <neso_particles/containers/sym_vector.hpp>
#include <neso_particles/device_buffers.hpp>
#include <neso_particles/device_functions.hpp>
#include <neso_particles/loop/particle_loop_index.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
template <size_t ndim>
struct InterpolateDataOnDevice : public ReactionDataBaseOnDevice<ndim> {
  InterpolateDataOnDevice() = default;

  std::array<REAL, ndim>
  calc_data(const std::array<REAL, ndim> &interpolation_points,
            const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice<ndim>::RNG_KERNEL_TYPE::KernelType
                &kernel) const {
    // Temporary vectors for intermediate calculation (in principle will hold
    // unique values for each particle)
    std::vector<INT> origin_indices(ndim);

    std::vector<REAL> vertex_func_evals(this->initial_num_points);
    std::vector<INT> vertex_coord(this->initial_num_points);

    std::vector<INT> input_vertices(this->initial_num_points);
    std::vector<INT> output_vertices(this->initial_num_points);
    std::vector<REAL> output_evals(this->initial_num_points);

    std::vector<INT> varying_dim(this->initial_num_points);

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

    // Calculation of the indices that will form the "origin" of the
    // hypercube. These are the smallest indices in each dimension that
    // still have coordinate values that are less than the interpolation
    // values in that dimension.
    origin_indices[0] = interp_utils::calc_closest_point_index(
        interpolation_points[0], ranges_vec_buf, dims_vec_buf[0]);
    for (int i = 1; i < ndim; i++) {
      origin_indices[i] = interp_utils::calc_closest_point_index(
          interpolation_points[i], ranges_vec_buf + dims_vec_buf[i - 1],
          dims_vec_buf[i]);
    }

    // Necessary for using the interp_utils functions.
    // Specifically converting the vectors to pointers prevents std::vector
    // being in the function signature for any of the interp_utils:: functions
    // so they should remain trivially copyable.
    auto interpolation_points_ptr = interpolation_points.data();
    auto origin_indices_ptr = origin_indices.data();

    auto vertex_func_evals_ptr = &(*vertex_func_evals.begin());
    auto vertex_coord_ptr = &(*vertex_coord.begin());

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
      input_vertices.at(i) = hypercube_vertices_buf[i];
    }

    // Loop until the last dimension (down to 0D)
    while (dim_index >= 0) {
      // Contract the hypercube vertices and evaluations, eg. if
      // input_vertices.size() == 2^3 then output_vertices.size() == 2^2, as
      // in going from a 3D hypercube(cube) to a 2D hypercube(square).
      // Additionally input_evals and output_evals are scaled down in the
      // same way via linear interpolation.
      interp_utils::contract_hypercube_on_device(
          current_count, interpolation_points_ptr, dim_index,
          input_vertices_ptr, origin_indices_ptr, vertex_func_evals_ptr,
          ranges_vec_buf, dims_vec_buf, output_vertices_ptr, output_evals_ptr,
          varying_dim_ptr, vertex_coord_ptr);

      // This now accounts for the lower size of output_vertices and
      // output_evals, and makes sure that any loops in future
      // contract_hypercube(...) invocations remain consistent.
      num_points = 1 << dim_index;
      dim_index--;

      // This resets the input_vertices and vertex_func_evals
      for (int i = 0; i < num_points; i++) {
        input_vertices.at(i) = output_vertices.at(i);
        vertex_func_evals.at(i) = output_evals.at(i);
      }
    }

    std::array<REAL, ndim> calculated_interpolated_vals;
    calculated_interpolated_vals.fill(0.0);

    // Assign to first element of returned array. (See
    // test/unit/test_interpolation.cpp for reasons for the workaround.)
    calculated_interpolated_vals[0] = output_evals.at(0);

    return calculated_interpolated_vals;
  }

public:
  INT *hypercube_vertices_ptr;
  size_t *dims_vec_ptr;
  REAL *ranges_vec_ptr;
  REAL *grid_ptr;

  static constexpr INT initial_num_points = 1 << ndim;
};

template <size_t ndim>
struct InterpolateData
    : public ReactionDataBase<InterpolateDataOnDevice<ndim>, ndim> {
  InterpolateData(const std::vector<size_t> &dims_vec,
                  const std::vector<REAL> &ranges_vec,
                  const std::vector<REAL> &grid,
                  SYCLTargetSharedPtr sycl_target) {
    auto initial_hypercube =
        interp_utils::construct_initial_hypercube(INT(ndim));

    this->on_device_obj = InterpolateDataOnDevice<ndim>();

    // BufferDevice<REAL> mock setup
    this->dims_vec_buf =
        std::make_shared<BufferDevice<size_t>>(sycl_target, dims_vec);
    this->on_device_obj->dims_vec_ptr = this->dims_vec_buf->ptr;

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
  std::shared_ptr<BufferDevice<REAL>> grid_buf;
  std::shared_ptr<BufferDevice<INT>> hypercube_vertices_buf;
};
}; // namespace VANTAGE::Reactions
#endif
