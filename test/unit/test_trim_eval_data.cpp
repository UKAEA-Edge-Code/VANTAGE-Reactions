#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <neso_particles/typedefs.hpp>
#include <random>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(TrimEval, INVALID_BOUNDS_CHECK) {
  static constexpr int ndim = 4;
  static constexpr int trim_ndim = 5;
  static constexpr int input_ndim = ndim + trim_ndim;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_int_distribution<size_t> uniform_dist_0(3, 7);
  std::uniform_real_distribution<REAL> uniform_dist_1(0.0, 10.0);

  std::vector<size_t> invalid_dims_vec;

  // Deliberately over allocating
  for (int i = 0; i < ndim + 1; i++) {
    invalid_dims_vec.push_back(uniform_dist_0(rng));
  }

  // Deliberately under allocating
  std::vector<size_t> invalid_trim_dims_vec;
  for (int i = 0; i < trim_ndim - 1; i++) {
    invalid_trim_dims_vec.push_back(uniform_dist_0(rng));
  }

  std::vector<REAL> invalid_ranges_vec;
  for (int i = 0; i < ndim; i++) {
    // Deliberately over-allocating
    for (int j = 0; j < invalid_dims_vec[i] + 2; j++) {
      invalid_ranges_vec.push_back(uniform_dist_1(rng));
    }
  }

  auto invalid_grid_stride = 0;
  auto invalid_aggregate_dim = 1;
  for (int i = 0; i < trim_ndim; i++) {
    // Deliberately under-allocating
    invalid_aggregate_dim *= invalid_trim_dims_vec[i] - 2;
    invalid_grid_stride += invalid_aggregate_dim;
  }

  auto num_grid_elems = 0;
  for (int i = 0; i < ndim; i++) {
    // Deliberately under-allocating
    num_grid_elems += invalid_ranges_vec[i] - 2;
  }

  num_grid_elems *= invalid_grid_stride;

  std::vector<REAL> invalid_grid_vec;
  for (int i = 0; i < num_grid_elems; i++) {
    invalid_grid_vec.push_back(uniform_dist_1(rng));
  }

  // Test dims_size error
  // Extra brackets around TrimEval needed due to googletest-related
  // preprocessor issue.
  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW((TrimEval<input_ndim, trim_ndim>(
                     invalid_grid_vec, invalid_ranges_vec, invalid_dims_vec,
                     invalid_trim_dims_vec, particle_group->sycl_target)),
                 std::logic_error);

  // Test trim_dims_size error
  std::vector<size_t> dims_vec;
  for (int i = 0; i < ndim; i++) {
    dims_vec.push_back(uniform_dist_0(rng));
  }

  // Extra brackets around TrimEval needed due to googletest-related
  // preprocessor issue.
  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW((TrimEval<input_ndim, trim_ndim>(
                     invalid_grid_vec, invalid_ranges_vec, dims_vec,
                     invalid_trim_dims_vec, particle_group->sycl_target)),
                 std::logic_error);

  // Test ranges_size error
  std::vector<size_t> trim_dims_vec;
  for (int i = 0; i < trim_ndim; i++) {
    trim_dims_vec.push_back(uniform_dist_0(rng));
  }

  // Extra brackets around TrimEval needed due to googletest-related
  // preprocessor issue.
  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW((TrimEval<input_ndim, trim_ndim>(
                     invalid_grid_vec, invalid_ranges_vec, dims_vec,
                     trim_dims_vec, particle_group->sycl_target)),
                 std::logic_error);

  // Test grid_size error
  std::vector<REAL> ranges_vec;
  for (int i = 0; i < ndim; i++) {
    for (int j = 0; j < dims_vec[i]; j++) {
      ranges_vec.push_back(uniform_dist_1(rng));
    }
  }

  // Extra brackets around TrimEval needed due to googletest-related
  // preprocessor issue.
  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW((TrimEval<input_ndim, trim_ndim>(invalid_grid_vec, ranges_vec,
                                                  dims_vec, trim_dims_vec,
                                                  particle_group->sycl_target)),
                 std::logic_error);

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}