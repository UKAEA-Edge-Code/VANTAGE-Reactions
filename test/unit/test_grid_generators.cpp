#include "include/mock_interpolation_data.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <neso_particles/typedefs.hpp>
#include <random>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(GridGenerators, INVALID_TRIM_FUNCS) {
  static constexpr int ndim = 2;
  static constexpr int trim_ndim = 3;

  auto dims = std::vector<int>(ndim, 2);

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;

  auto mesh =
      std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                                       subdivision_order, stencil_width);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  const int rank = sycl_target->comm_pair.rank_parent;

  auto rng = std::mt19937(52234126 + rank);

  std::uniform_real_distribution<REAL> uniform_dist_m1(0.0, 10.0);
  std::array<REAL, trim_ndim> random_grid_nums;
  for (int i = 0; i < trim_ndim; i++) {
    random_grid_nums[i] = uniform_dist_m1(rng);
  }

  auto coeffs_data = trim_coefficient_values(random_grid_nums, sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_flat_vec = coeffs_data.get_ranges_flat_vec();
  auto trim_dims_vec = coeffs_data.get_trim_dims_vec();

  auto trim_grid_func0 = coeffs_data.get_trim_grid_func_0();
  auto trim_grid_func1 = coeffs_data.get_trim_grid_func_1();

  // Deliberately incorrect trim_grid_func2
  auto invalid_trim_grid_func2 =
      [&](const REAL &dim0_val, const REAL &dim1_val,
          const std::array<REAL, trim_ndim> &rand_nums) {
        std::array<REAL, 1> result{0};
        return result;
      };

  std::array<std::vector<REAL>, ndim> ranges;

  for (int idim = 0; idim < dims_vec[0]; idim++) {
    ranges[0].push_back(ranges_flat_vec[idim]);
  }
  for (int i = 1; i < ndim; i++) {
    for (int j = 0; j < dims_vec[i]; j++) {
      ranges[i].push_back(ranges_flat_vec[dims_vec[i - 1] + j]);
    }
  }

  std::array<size_t, trim_ndim> trim_dims{};
  for (int i = 0; i < trim_ndim; i++) {
    trim_dims[i] = trim_dims_vec[i];
  }

  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW((TrimGridGenerator<ndim, trim_ndim>(
                     ranges, trim_dims, trim_grid_func0, trim_grid_func1,
                     invalid_trim_grid_func2, random_grid_nums)),
                 std::logic_error);

  auto trim_grid_func2 = coeffs_data.get_trim_grid_func_2();

  // Assumes trim_dims_vec[i] = 5 for all i.
  std::uniform_int_distribution<INT> uniform_dist_0(-2, 3);
  std::array<REAL, trim_ndim> random_trim_nums;
  for (int i = 0; i < trim_ndim; i++) {
    random_trim_nums[i] = uniform_dist_0(rng);
  }

  std::array<size_t, trim_ndim> invalid_trim_dims{};
  for (int i = 0; i < trim_ndim; i++) {
    invalid_trim_dims[i] = trim_dims_vec[i] + random_trim_nums[i];
  }

  if (std::getenv("TEST_NESOASSERT") != nullptr)
    EXPECT_THROW((TrimGridGenerator<ndim, trim_ndim>(
                     ranges, invalid_trim_dims, trim_grid_func0,
                     trim_grid_func1, trim_grid_func2, random_grid_nums)),
                 std::logic_error);
}
