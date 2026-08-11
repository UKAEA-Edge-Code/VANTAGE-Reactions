
#include "../include/mock_interpolation_data.hpp"
#include "../include/mock_particle_group.hpp"
#include "../include/test_vantage_reactions_utils.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/typedefs.hpp>
#include <random>

#define INTERPOLATION_TOLERANCE 1e-14

using namespace VANTAGE::Reactions;

TEST(InterpolationTest, REACTION_DATA_5D_PIPELINE) {
  static constexpr int ndim = 5;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP0"), 1);
  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP1"), 1);
  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP2"), 1);
  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP3"), 1);
  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP4"), 1);
  particle_group->add_particle_dat(
      NP::Sym<NP::REAL>("EXPECTED_INTERPOLATION_VALUE"), 1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_5D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<NP::REAL> uniform_dist_0(lower_bounds[0],
                                                          upper_bounds[0]);
  std::uniform_real_distribution<NP::REAL> uniform_dist_1(lower_bounds[1],
                                                          upper_bounds[1]);
  std::uniform_real_distribution<NP::REAL> uniform_dist_2(lower_bounds[2],
                                                          upper_bounds[2]);
  std::uniform_real_distribution<NP::REAL> uniform_dist_3(lower_bounds[3],
                                                          upper_bounds[3]);
  std::uniform_real_distribution<NP::REAL> uniform_dist_4(lower_bounds[4],
                                                          upper_bounds[4]);

  auto rng_kernel_0 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel_1 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);
  auto rng_kernel_2 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), 1);
  auto rng_kernel_3 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_3, rng), 1);
  auto rng_kernel_4 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_4, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto prop1, auto prop2, auto prop3,
          auto prop4, auto expected_value, auto kernel0, auto kernel1,
          auto kernel2, auto kernel3, auto kernel4) {
        prop0.at(0) = kernel0.at(index, 0);
        prop1.at(0) = kernel1.at(index, 0);
        prop2.at(0) = kernel2.at(index, 0);
        prop3.at(0) = kernel3.at(index, 0);
        prop4.at(0) = kernel4.at(index, 0);
        auto coords = std::array<NP::REAL, ndim>{
            prop0.at(0), prop1.at(0), prop2.at(0), prop3.at(0), prop4.at(0)};
        expected_value.at(0) = grid_func(coords);
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<NP::REAL>("PROP0")),
      NP::Access::write(NP::Sym<NP::REAL>("PROP1")),
      NP::Access::write(NP::Sym<NP::REAL>("PROP2")),
      NP::Access::write(NP::Sym<NP::REAL>("PROP3")),
      NP::Access::write(NP::Sym<NP::REAL>("PROP4")),
      NP::Access::write(NP::Sym<NP::REAL>("EXPECTED_INTERPOLATION_VALUE")),
      NP::Access::read(rng_kernel_0), NP::Access::read(rng_kernel_1),
      NP::Access::read(rng_kernel_2), NP::Access::read(rng_kernel_3),
      NP::Access::read(rng_kernel_4))
      ->execute();

  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto prop3_extract = extract<1>("PROP3");
  auto prop4_extract = extract<1>("PROP4");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract,
                       prop3_extract, prop4_extract);

  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto expect_results_dat = expect_pre_req_data->get();

    EXPECT_EQ(n_part_cell, calc_pre_req_data->index.shape[0]);
    EXPECT_EQ(n_part_cell, expect_pre_req_data->index.shape[0]);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {

      auto calculated_interpolation_value = calc_results_dat[ipart];
      auto expected_interpolation_value = expect_results_dat[ipart];

      auto rel_error = relative_error(expected_interpolation_value,
                                      calculated_interpolation_value);
      EXPECT_NEAR(rel_error, 0.0, INTERPOLATION_TOLERANCE);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
