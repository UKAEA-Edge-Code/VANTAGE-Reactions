
#include "../include/mock_interpolation_data.hpp"
#include "../include/mock_particle_group.hpp"
#include "../include/test_vantage_reactions_utils.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <gtest/gtest.h>

using namespace VANTAGE::Reactions;

TEST(ExtrapolationTest, TRIM_DATA_OVER_UNDER_OVER) {
  // This is mainly to test out of bound random numbers given to TrimEval
  // through InterpolateData

  static constexpr int ndim = 2;
  static constexpr int trim_ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto rng = std::mt19937(52234126 + rank);

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROPS"), ndim);
  particle_group->add_particle_dat(NP::Sym<NP::REAL>("TRIM_INDICES"),
                                   trim_ndim);

  particle_group->add_particle_dat(
      NP::Sym<NP::REAL>("EXPECTED_INTERPOLATION_VALUE"), trim_ndim);

  // PANIC flag for TrimEval
  particle_group->remove_particle_dat(NP::Sym<NP::INT>("REACTIONS_PANIC_FLAG"));
  particle_group->add_particle_dat(NP::Sym<NP::INT>("REACTIONS_PANIC_FLAG"),
                                   trim_ndim);

  // Setup the mock data.
  std::uniform_real_distribution<NP::REAL> uniform_dist_m1(0.0, 10.0);
  std::array<NP::REAL, trim_ndim> random_grid_nums;
  for (int i = 0; i < trim_ndim; i++) {
    random_grid_nums[i] = uniform_dist_m1(rng);
  }

  auto coeffs_data = trim_coefficient_values_asym(random_grid_nums,
                                                  particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func_data = coeffs_data.get_grid_func_data();
  auto grid_func = coeffs_data.get_grid_func();
  auto trim_dims_vec = coeffs_data.get_trim_dims_vec();

  std::array<NP::INT, trim_ndim> trim_dims_arr;
  for (int i = 0; i < trim_ndim; i++) {
    trim_dims_arr[i] = trim_dims_vec[i];
  }

  // Random number generator kernel
  std::uniform_real_distribution<NP::REAL> uniform_dist_0(lower_bounds[0],
                                                          upper_bounds[0]);
  std::uniform_real_distribution<NP::REAL> uniform_dist_1(lower_bounds[1],
                                                          upper_bounds[1]);

  auto rng_kernel0 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel1 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto props, auto prop0_kernel, auto prop1_kernel,
          auto trim_indices, auto panic_flags) {
        props.at(0) = prop0_kernel.at(index, 0);
        props.at(1) = prop1_kernel.at(index, 0);

        // Intentionally out-of-bound trim numbers.
        std::array<NP::REAL, trim_ndim> real_trim_indices = {1.0, -0.15, 2.7};

        trim_indices.at(0) = real_trim_indices[0];
        trim_indices.at(1) = real_trim_indices[1];
        trim_indices.at(2) = real_trim_indices[2];

        panic_flags.at(0) = 0;
        panic_flags.at(1) = 0;
        panic_flags.at(2) = 0;
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<NP::REAL>("PROPS")),
      NP::Access::read(rng_kernel0), NP::Access::read(rng_kernel1),
      NP::Access::write(NP::Sym<NP::REAL>("TRIM_INDICES")),
      NP::Access::write(NP::Sym<NP::INT>("REACTIONS_PANIC_FLAG")))
      ->execute();

  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto props_extract = extract<ndim>("PROPS");

  auto trim_extract = extract<trim_ndim>("TRIM_INDICES");

  auto concatenator = ConcatenatorData(props_extract, trim_extract);

  std::array<size_t, ndim> interp_indices = {0, 1};

  auto interpolator_data =
      InterpolateData<trim_ndim, ndim, decltype(grid_func_data), trim_ndim>(
          dims_vec, coords_vec, interp_indices, particle_group->sycl_target,
          grid_func_data, ExtrapolationType::continue_linear);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto concat_data_calc = DataCalculator(pipeline);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    NP::INT expected_panic_value;
    std::vector<NP::INT> cells;
    std::vector<NP::INT> layers;
    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      cells.push_back(i);
      layers.push_back(ipart);
    }

    auto particles = particle_sub_group->get_particles(cells, layers);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      layers[0] = ipart;
      for (int icomp = 0; icomp < trim_ndim; icomp++) {
        expected_panic_value =
            particles->at(NP::Sym<NP::INT>("REACTIONS_PANIC_FLAG"), 0, icomp);

        EXPECT_GT(expected_panic_value, 0);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
