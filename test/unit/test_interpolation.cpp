#include "include/mock_particle_group.hpp"
#include "reactions_lib/interp_utils.hpp"
#include <array>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <reactions/reactions.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

struct coefficient_values {
private:
  const int ndim = 2;
  static const size_t ndens_dim = 8;
  static const size_t ntemp_dim = 10;

  static constexpr std::array<REAL, ndens_dim> dens_range = {
      1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18, 5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  static constexpr std::array<REAL, ntemp_dim> temp_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};

  std::array<std::array<REAL, ndens_dim>, ntemp_dim> coeffs;

  std::vector<REAL> coeffs_vec;
  std::vector<std::vector<REAL>> ranges_vec;
  std::vector<REAL> ranges_flat_vec;

public:
  coefficient_values() {
    REAL temp_i = 0.0;
    for (int itemp = 0; itemp < ntemp_dim; itemp++) {
      temp_i = this->temp_range[itemp];
      for (int idens = 0; idens < ndens_dim; idens++) {
        this->coeffs[itemp][idens] = temp_i * dens_range[idens];
        this->coeffs_vec.push_back(this->coeffs[itemp][idens]);
      }
    }

    std::vector<REAL> dens_range_vec(this->dens_range.begin(),
                                     this->dens_range.end());
    std::vector<REAL> temp_range_vec(this->temp_range.begin(),
                                     this->temp_range.end());

    std::vector<REAL> ranges_flat_vec = dens_range_vec;
    ranges_flat_vec.insert(ranges_flat_vec.end(), temp_range_vec.begin(),
                           temp_range_vec.end());
    this->ranges_flat_vec = ranges_flat_vec;

    this->ranges_vec.push_back(temp_range_vec);
    this->ranges_vec.push_back(dens_range_vec);
  };

  const std::array<REAL, ndens_dim> &get_dens_range() {
    return this->dens_range;
  }

  const std::array<REAL, ntemp_dim> &get_temp_range() {
    return this->temp_range;
  }

  const std::array<std::array<REAL, ndens_dim>, ntemp_dim> &get_coeffs() {
    return this->coeffs;
  }

  const std::vector<std::vector<REAL>> &get_ranges_vec() {
    return this->ranges_vec;
  }

  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }

  const std::vector<REAL> &get_ranges_flat_vec() {
    return this->ranges_flat_vec;
  }

  std::vector<size_t> get_dims_vec() {
    // return std::vector<size_t>{this->ndens_dim, this->ntemp_dim};
    return std::vector<size_t>{this->ndens_dim, this->ntemp_dim};
  }

  const int &get_ndim() { return this->ndim; }
};

inline void
diagnostic_output(const int &particle_count_, const int &dim_index_,
                  const int &num_points_,
                  Access::LocalMemoryInterlaced::Write<size_t> origin_indices_,
                  Access::LocalMemoryInterlaced::Write<int> vertices_,
                  Access::LocalMemoryInterlaced::Write<REAL> func_evals_,
                  size_t *dims_vec_ptr_, REAL *ranges_vec_ptr_) {
  for (int i = 0; i < num_points_; i++) {
    if (particle_count_ == 0) {
      for (int j = dim_index_; j >= 0; j--) {
        size_t temp_binary_repr =
            origin_indices_.at(j) + binary_extract(vertices_.at(i), j);
        REAL ranges_val = ranges_vec_ptr_[interp_utils::range_index_on_device(
            temp_binary_repr, size_t(j), dims_vec_ptr_)];
        printf("%ld, %e\t", temp_binary_repr, ranges_val);
      }
      printf("%e\n", func_evals_.at(i));
    }
  }
  if (particle_count_ == 0) {
    printf("\n");
  }
};

TEST(InterpolationTest, INTERP_2D) {
  // Interpolation points
  REAL fluid_density_interp = 6.4e18;
  REAL fluid_temp_interp = 1.9e3;
  REAL expected_interp_value = fluid_density_interp * fluid_temp_interp;
  printf("Expected interpolated value: %e\n", expected_interp_value);

  // Initialize a particle group with the fluid density and fluid temperature
  // for all particles set to the interpolation values.
  auto particle_group = create_test_particle_group(1e1);

  auto npart = particle_group->get_npart_local();

  particle_loop(
      particle_group,
      [=](auto fdens, auto ftemp) {
        fdens.at(0) = fluid_density_interp;
        ftemp.at(0) = fluid_temp_interp;
      },
      Access::write(Sym<REAL>("FLUID_DENSITY")),
      Access::write(Sym<REAL>("FLUID_TEMPERATURE")))
      ->execute();

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values();
  auto ndim = ADAS_values.get_ndim();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto coeffs_vec = ADAS_values.get_coeffs_vec();

  // BufferDevice<REAL> mock setup
  auto dims_vec_buf = std::make_shared<BufferDevice<size_t>>(
      particle_group->sycl_target, dims_vec);
  auto dims_vec_ptr = dims_vec_buf->ptr;

  auto ranges_vec_buf = std::make_shared<BufferDevice<REAL>>(
      particle_group->sycl_target, ranges_vec);
  auto ranges_vec_ptr = ranges_vec_buf->ptr;

  auto coeffs_vec_buf = std::make_shared<BufferDevice<REAL>>(
      particle_group->sycl_target, coeffs_vec);
  auto coeffs_vec_ptr = coeffs_vec_buf->ptr;

  // Processed on-device-ready data setup
  LocalMemoryInterlaced<size_t> origin_indices_mem(ndim);

  auto initial_hypercube = interp_utils::construct_initial_hypercube(ndim);
  int initial_num_points = initial_hypercube.size();

  auto hypercube_vertices_buf = std::make_shared<BufferDevice<int>>(
      particle_group->sycl_target, initial_hypercube);
  auto hypercube_vertices_ptr = hypercube_vertices_buf->ptr;

  LocalMemoryInterlaced<REAL> vertex_func_evals_mem(initial_num_points);
  LocalMemoryInterlaced<size_t> vertex_coord_mem(initial_num_points);
  LocalMemoryInterlaced<REAL> interp_points_mem(ndim);
  LocalMemoryInterlaced<int> input_vertices_mem(initial_num_points);
  LocalMemoryInterlaced<int> output_vertices_mem(initial_num_points);
  LocalMemoryInterlaced<REAL> output_evals_mem(initial_num_points);
  LocalMemoryInterlaced<size_t> varying_dim_mem(initial_num_points);

  auto result_buf = std::make_shared<LocalArray<REAL>>(
      particle_group->sycl_target, npart, 0.0);

  // Main loop
  auto interp_loop = particle_loop(
      "interp_loop", particle_group,
      [=](auto particle_index, auto fluid_dens_interp, auto fluid_temp_interp,
          auto origin_indices, auto vertex_func_evals, auto vertex_coord,
          auto interp_points, auto input_vertices, auto output_vertices,
          auto output_evals, auto varying_dim, auto result) {
        auto particle_count = particle_index.get_loop_linear_index();

        // Initial dim_index and num_points values. The variable, dim_index,
        // tracks the progress through the hypercube contraction.
        int dim_index = ndim - 1;
        int num_points = initial_num_points;

        interp_points.at(0) = fluid_dens_interp.at(0);
        interp_points.at(1) = fluid_temp_interp.at(0);

        // Calculation of the indices that will form the "origin" of the
        // hypercube. These are the smallest indices in each dimension that
        // still have coordinate values that are less than the interpolation
        // values in that dimension.
        origin_indices.at(0) = interp_utils::calc_closest_point_index(
            interp_points.at(0), ranges_vec_ptr, dims_vec_ptr[0]);
        for (int i = 1; i < ndim; i++) {
          origin_indices.at(i) = interp_utils::calc_closest_point_index(
              interp_points.at(i), ranges_vec_ptr + dims_vec_ptr[i - 1],
              dims_vec_ptr[i]);
        }

        // Initial function evaluation (ie values of the coeffs_vec) based on
        // the vertices of the hypercube.
        interp_utils::initial_func_eval_on_device(
            vertex_func_evals, vertex_coord, coeffs_vec_ptr,
            hypercube_vertices_ptr, origin_indices, dims_vec_ptr, ndim);

        // Fill input_vertices vector
        for (int i = 0; i < initial_num_points; i++) {
          input_vertices.at(i) = hypercube_vertices_ptr[i];
        }

        diagnostic_output(particle_count, dim_index, initial_num_points,
                          origin_indices, input_vertices, vertex_func_evals,
                          dims_vec_ptr, ranges_vec_ptr);

        // Loop until the last dimension (down to 0D)
        while (dim_index >= 0) {
          // Contract the hypercube vertices and evaluations, eg. if
          // input_vertices.size() == 2^3 then output_vertices.size() == 2^2, as
          // in going from a 3D hypercube(cube) to a 2D hypercube(square).
          // Additionally input_evals and output_evals are scaled down in the
          // same way via linear interpolation.
          interp_utils::contract_hypercube_on_device(
              particle_count, interp_points, dim_index, input_vertices,
              origin_indices, vertex_func_evals, ranges_vec_ptr, dims_vec_ptr,
              output_vertices, output_evals, varying_dim, vertex_coord);

          // This now accounts for the lower size of output_vertices and
          // output_evals, and makes sure that any loops in future
          // contract_hypercube(...) invocations remain consistent.
          dim_index--;
          num_points = 1 << (dim_index + 1);

          // This resets the input_vertices and vertex_func_evals
          for (int i = 0; i < num_points; i++) {
            input_vertices.at(i) = output_vertices.at(i);
            vertex_func_evals.at(i) = output_evals.at(i);
          }

          diagnostic_output(particle_count, dim_index, num_points,
                            origin_indices, output_vertices, output_evals,
                            dims_vec_ptr, ranges_vec_ptr);
        }

        // Save final result to a buffer
        result.at(particle_count) = output_evals.at(0);
      },
      Access::read(ParticleLoopIndex{}),
      Access::read(Sym<REAL>("FLUID_DENSITY")),
      Access::read(Sym<REAL>("FLUID_TEMPERATURE")),
      Access::write(origin_indices_mem), Access::write(vertex_func_evals_mem),
      Access::write(vertex_coord_mem), Access::write(interp_points_mem),
      Access::write(input_vertices_mem), Access::write(output_vertices_mem),
      Access::write(output_evals_mem), Access::write(varying_dim_mem),
      Access::write(result_buf));

  interp_loop->execute();

  auto result_data = result_buf->get();

  for (int i = 0; i < npart; i++) {
    EXPECT_DOUBLE_EQ(result_data[i], expected_interp_value);
  }

  particle_group->domain->mesh->free();
}
