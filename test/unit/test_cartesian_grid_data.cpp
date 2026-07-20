#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <neso_particles/typedefs.hpp>
#include <random>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(CartesianGridData, INVALID_BOUNDS_CHECK) {
  static constexpr int ndim = 3;

  auto [mesh, sycl_target] = mesh_sycl_target_only();

  const int rank = sycl_target->comm_pair.rank_parent;
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_int_distribution<size_t> uniform_dist_0(1, 5);
  std::uniform_real_distribution<REAL> uniform_dist_1(0.0, 10.0);
  std::vector<size_t> invalid_dims_vec;

  // Deliberately over allocating
  for (int i = 0; i < ndim + 1; i++) {
    invalid_dims_vec.push_back(uniform_dist_0(rng));
  }

  std::vector<REAL> invalid_coords_vec;
  for (int i = 0; i < ndim; i++) {
    // Deliberately over-allocating
    for (int j = 0; j < invalid_dims_vec[i] + 2; j++) {
      invalid_coords_vec.push_back(uniform_dist_1(rng));
    }
  }

  auto num_grid_elems = 0;
  for (int i = 0; i < ndim; i++) {
    // Deliberately under-allocating
    num_grid_elems += invalid_coords_vec[i] - 2;
  }

  std::vector<REAL> invalid_grid_vec;
  for (int i = 0; i < num_grid_elems; i++) {
    invalid_grid_vec.push_back(uniform_dist_1(rng));
  }

  // Test dims_size error
  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW(CartesianGridData<ndim>(invalid_grid_vec, invalid_coords_vec,
                                         invalid_dims_vec, sycl_target),
                 std::logic_error);

  // Test ranges_size error
  std::vector<size_t> dims_vec;
  for (int i = 0; i < ndim; i++) {
    dims_vec.push_back(uniform_dist_0(rng));
  }

  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW(CartesianGridData<ndim>(invalid_grid_vec, invalid_coords_vec,
                                         dims_vec, sycl_target),
                 std::logic_error);

  // Test grid_size error
  std::vector<REAL> coords_vec;
  int ranges_size = 0;
  for (auto &idim : dims_vec) {
    ranges_size += idim;
  }

  for (int i = 0; i < ranges_size; i++) {
    coords_vec.push_back(uniform_dist_1(rng));
  }

  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW(CartesianGridData<ndim>(invalid_grid_vec, coords_vec, dims_vec,
                                         sycl_target),
                 std::logic_error);

  sycl_target->free();
  mesh->free();
}
