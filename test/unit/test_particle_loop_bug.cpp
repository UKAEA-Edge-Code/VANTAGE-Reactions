#include "include/mock_debug_particle_group.hpp"
#include "include/mock_interpolation_data.hpp"
#include "include/test_vantage_reactions_utils.hpp"
#include <gtest/gtest.h>
#include <neso_particles/error_propagate.hpp>
#include <neso_particles/typedefs.hpp>

namespace particle_loop_bug_test {
static inline bool fp_eq(REAL a, REAL b, REAL eps) {
  return std::fabs((a) - (b)) <= (eps);
}
} // namespace particle_loop_bug_test

using namespace particle_loop_bug_test;

TEST(ParticleLoopBug, BUG_DEMO) {
  static constexpr int ndim = 2;
  static constexpr int trim_ndim = 3;

  // If number of particles is reduced to 1, the behaviour is correct.
  auto particle_group = create_trim_test_particle_group(5e1);

  particle_group->add_particle_dat(Sym<REAL>("PROPS"), ndim);
  particle_group->add_particle_dat(Sym<INT>("SCALED_TRIM_INDICES"), trim_ndim);
  particle_group->add_particle_dat(Sym<REAL>("STORED_VALUE"), trim_ndim);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto mock_data = trim_coefficient_values(particle_group->sycl_target);
  auto grid_func = mock_data.get_grid_func();
  auto upper_bounds = mock_data.get_upper_bounds();
  auto lower_bounds = mock_data.get_lower_bounds();
  auto trim_dims_vec = mock_data.get_trim_dims_vec();

  // Random number generator kernel
  auto rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);

  auto rng_kernel0 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel1 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);

  std::array<INT, trim_ndim> trim_dims_arr;
  for (int i = 0; i < trim_ndim; i++)
    trim_dims_arr[i] = trim_dims_vec[i];

  // Hard-coded for reproducibility
  std::array<INT, trim_ndim> expected_indices = {0, 0, 1};

  // Not the ideal way to extract values from the loop but seems functional and
  // doesn't affect test behaviour/misbehaviour.
  auto mock_prop0 = std::make_shared<REAL>(0.0);
  auto mock_prop1 = std::make_shared<REAL>(0.0);

  printf("\nBEFORE GRID FUNCTION EVALUATION.....\n\n");

  particle_loop(
      particle_group,
      [=](auto index, auto props, auto scaled_trim_indices, auto kernel0,
          auto kernel1) {
        props.at(0) = kernel0.at(index, 0);
        props.at(1) = kernel1.at(index, 0);

        scaled_trim_indices.at(0) = expected_indices[0];
        scaled_trim_indices.at(1) = expected_indices[1];
        scaled_trim_indices.at(2) = expected_indices[2];

        auto current_count = index.get_loop_linear_index();
        if (current_count == 0) {
          (*mock_prop0) = props.at(0);
          (*mock_prop1) = props.at(1);

          printf("props0: %e\n", props.at(0));
          printf("props1: %e\n", props.at(1));
          printf("\n");
          printf("scaled_trim_indices[0]: %ld\n", scaled_trim_indices.at(0));
          printf("scaled_trim_indices[1]: %ld\n", scaled_trim_indices.at(1));
          printf("scaled_trim_indices[2]: %ld\n", scaled_trim_indices.at(2));
          printf("\n");
        }
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROPS")),
      Access::write(Sym<INT>("SCALED_TRIM_INDICES")), Access::read(rng_kernel0),
      Access::read(rng_kernel1))
      ->execute();

  auto expected_stored_value =
      grid_func((*mock_prop0), (*mock_prop1), expected_indices, trim_dims_arr);
  for (int i = 0; i < trim_ndim; i++) {
    printf("expected_stored_value[%d]: %e\n", i, expected_stored_value[i]);
  }

  particle_loop(
      particle_group,
      [=](auto index, auto props, auto scaled_trim_indices, auto stored_value) {
        std::array<INT, trim_ndim> scaled_indices;
        for (int i = 0; i < trim_ndim; i++) {
          scaled_indices[i] = scaled_trim_indices.at(i);
        }

        auto result =
            grid_func(props.at(0), props.at(1), scaled_indices, trim_dims_arr);

        stored_value.at(0) = result[0];
        stored_value.at(1) = result[1];
        stored_value.at(2) = result[2];

        auto current_count = index.get_loop_linear_index();
        if (current_count == 0) {
          // un-comment the following print statement to restore correct
          // behaviour. printf("\n");
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<REAL>("PROPS")),
      Access::read(Sym<INT>("SCALED_TRIM_INDICES")),
      Access::write(Sym<REAL>("STORED_VALUE")))
      ->execute();

  printf("\nAFTER GRID FUNCTION EVALUATION.....\n\n");
  particle_loop(
      particle_group,
      [=](auto index, auto props, auto scaled_trim_indices, auto stored_value) {
        auto current_count = index.get_loop_linear_index();
        if (current_count == 0) {
          printf("props0: %e\n", props.at(0));
          printf("props1: %e\n", props.at(1));
          printf("\n");
          printf("scaled_trim_indices[0]: %ld\n", scaled_trim_indices.at(0));
          printf("scaled_trim_indices[1]: %ld\n", scaled_trim_indices.at(1));
          printf("scaled_trim_indices[2]: %ld\n", scaled_trim_indices.at(2));
          printf("\n");
          printf("stored_value[0]: %e\n", stored_value.at(0));
          printf("stored_value[1]: %e\n", stored_value.at(1));
          printf("stored_value[2]: %e\n", stored_value.at(2));
          printf("\n");
          fp_eq(stored_value.at(2), expected_stored_value[2], 1e-6)
              ? printf("PASS!\n")
              : printf("FAIL!\n");
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<REAL>("PROPS")),
      Access::read(Sym<INT>("SCALED_TRIM_INDICES")),
      Access::read(Sym<REAL>("STORED_VALUE")))
      ->execute();

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}