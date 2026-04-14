#include "include/mock_debug_particle_group.hpp"
#include "include/mock_interpolation_data.hpp"
#include "include/mock_particle_group.hpp"
#include "include/test_vantage_reactions_utils.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/device_buffers.hpp>
#include <neso_particles/error_propagate.hpp>
#include <random>

#define INTERPOLATION_TOLERANCE 1e-14

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(InterpolationTest, REACTION_DATA_1D_PIPELINE) {
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist(lower_bounds[0],
                                                    upper_bounds[0]);

  auto rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto expected_value, auto kernel) {
        prop0.at(0) = kernel.at(index, 0);
        expected_value.at(0) = grid_func(prop0.at(0));
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");

  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, ranges_vec, particle_group->sycl_target, grid_func_data);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0.0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    expect_pre_req_data->fill(0.0);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < calc_pre_req_data->index.shape[0]; ipart++) {
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

TEST(InterpolationTest, REACTION_DATA_2D_PIPELINE) {
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);

  auto rng_kernel_0 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel_1 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto prop1, auto expected_value, auto kernel0,
          auto kernel1) {
        prop0.at(0) = kernel0.at(index, 0);
        prop1.at(0) = kernel1.at(index, 0);
        expected_value.at(0) = grid_func(prop0.at(0), prop1.at(0));
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, ranges_vec, particle_group->sycl_target, grid_func_data);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();

    shape = expect_data_calc.get_data_size();
    n_part_cell = particle_sub_group->get_npart_cell(i);
    buffer_size = n_part_cell;
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto expect_results_dat = expect_pre_req_data->get();

    for (int ipart = 0; ipart < calc_pre_req_data->index.shape[0]; ipart++) {

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

TEST(InterpolationTest, REACTION_DATA_3D_PIPELINE) {
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_3D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(lower_bounds[2],
                                                      upper_bounds[2]);

  auto rng_kernel_0 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel_1 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);
  auto rng_kernel_2 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto prop1, auto prop2, auto expected_value,
          auto kernel0, auto kernel1, auto kernel2) {
        prop0.at(0) = kernel0.at(index, 0);
        prop1.at(0) = kernel1.at(index, 0);
        prop2.at(0) = kernel2.at(index, 0);
        expected_value.at(0) = grid_func(prop0.at(0), prop1.at(0), prop2.at(0));
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("PROP1")), Access::write(Sym<REAL>("PROP2")),
      Access::write(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1),
      Access::read(rng_kernel_2))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, ranges_vec, particle_group->sycl_target, grid_func_data);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();

    shape = expect_data_calc.get_data_size();
    n_part_cell = particle_sub_group->get_npart_cell(i);
    buffer_size = n_part_cell;
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto expect_results_dat = expect_pre_req_data->get();

    for (int ipart = 0; ipart < calc_pre_req_data->index.shape[0]; ipart++) {

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

TEST(InterpolationTest, REACTION_DATA_4D_PIPELINE) {
  static constexpr int ndim = 4;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP3"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_4D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(lower_bounds[2],
                                                      upper_bounds[2]);
  std::uniform_real_distribution<REAL> uniform_dist_3(lower_bounds[3],
                                                      upper_bounds[3]);

  auto rng_kernel_0 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel_1 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);
  auto rng_kernel_2 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), 1);
  auto rng_kernel_3 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_3, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto prop1, auto prop2, auto prop3,
          auto expected_value, auto kernel0, auto kernel1, auto kernel2,
          auto kernel3) {
        prop0.at(0) = kernel0.at(index, 0);
        prop1.at(0) = kernel1.at(index, 0);
        prop2.at(0) = kernel2.at(index, 0);
        prop3.at(0) = kernel3.at(index, 0);
        expected_value.at(0) =
            grid_func(prop0.at(0), prop1.at(0), prop2.at(0), prop3.at(0));
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("PROP1")), Access::write(Sym<REAL>("PROP2")),
      Access::write(Sym<REAL>("PROP3")),
      Access::write(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1),
      Access::read(rng_kernel_2), Access::read(rng_kernel_3))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto prop3_extract = extract<1>("PROP3");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract,
                                       prop2_extract, prop3_extract);

  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, ranges_vec, particle_group->sycl_target, grid_func_data);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
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

TEST(InterpolationTest, REACTION_DATA_5D_PIPELINE) {
  static constexpr int ndim = 5;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP3"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP4"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_5D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(lower_bounds[2],
                                                      upper_bounds[2]);
  std::uniform_real_distribution<REAL> uniform_dist_3(lower_bounds[3],
                                                      upper_bounds[3]);
  std::uniform_real_distribution<REAL> uniform_dist_4(lower_bounds[4],
                                                      upper_bounds[4]);

  auto rng_kernel_0 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel_1 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);
  auto rng_kernel_2 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), 1);
  auto rng_kernel_3 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_3, rng), 1);
  auto rng_kernel_4 = host_per_particle_block_rng<REAL>(
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
        expected_value.at(0) = grid_func(prop0.at(0), prop1.at(0), prop2.at(0),
                                         prop3.at(0), prop4.at(0));
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("PROP1")), Access::write(Sym<REAL>("PROP2")),
      Access::write(Sym<REAL>("PROP3")), Access::write(Sym<REAL>("PROP4")),
      Access::write(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1),
      Access::read(rng_kernel_2), Access::read(rng_kernel_3),
      Access::read(rng_kernel_4))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto prop3_extract = extract<1>("PROP3");
  auto prop4_extract = extract<1>("PROP4");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract,
                       prop3_extract, prop4_extract);

  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, ranges_vec, particle_group->sycl_target, grid_func_data);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
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

TEST(InterpolationTest, TRIM_DATA_PIPELINE_EXACT) {
  static constexpr int ndim = 2;
  static constexpr int trim_ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROPS"), ndim);
  particle_group->add_particle_dat(Sym<REAL>("TRIM_INDICES"), trim_ndim);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"),
                                   trim_ndim);

  // Setup the mock data.
  auto coeffs_data = trim_coefficient_values(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func_data = coeffs_data.get_grid_func_data();
  auto grid_func = coeffs_data.get_grid_func();
  auto trim_dims_vec = coeffs_data.get_trim_dims_vec();

  std::array<size_t, ndim> dims_arr;
  for (int i = 0; i < ndim; i++) {
    dims_arr[i] = dims_vec[i];
  }

  auto h_ranges_arr = std::make_shared<BufferDevice<REAL>>(
      particle_group->sycl_target, ranges_vec);
  auto d_ranges_arr = h_ranges_arr->ptr;

  std::array<INT, trim_ndim> trim_dims_arr;
  for (int i = 0; i < trim_ndim; i++) {
    trim_dims_arr[i] = trim_dims_vec[i];
  }

  // Random number generator kernel
  auto rng = std::mt19937(52234126 + rank);
  std::uniform_int_distribution<INT> uniform_dist_0(0, dims_vec[0] - 1);
  std::uniform_int_distribution<INT> uniform_dist_1(0, dims_vec[1] - 1);
  std::uniform_real_distribution<REAL> uniform_dist_2(0.0, 1.0);

  auto rng_kernel0 = host_per_particle_block_rng<INT>(
      rng_lambda_wrapper_int(uniform_dist_0, rng), 1);
  auto rng_kernel1 = host_per_particle_block_rng<INT>(
      rng_lambda_wrapper_int(uniform_dist_1, rng), 1);
  auto trim_rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), trim_ndim);

  auto test_error_propagate =
      std::make_shared<ErrorPropagate>(particle_group->sycl_target);
  auto d_test_error_propagate_ptr = test_error_propagate->device_ptr();

  particle_loop(
      particle_group,
      [=](auto index, auto props, auto prop0_kernel, auto prop1_kernel,
          auto trim_indices, auto trim_kernel, auto expected_value) {
        auto index0 = prop0_kernel.at(index, 0);
        auto index1 = prop1_kernel.at(index, 0);

        auto indices = std::array<INT, ndim>{index0, index1};

        props.at(0) = d_ranges_arr[index0];
        props.at(1) = d_ranges_arr[dims_arr[0] + index1];

        auto current_count = index.get_loop_linear_index();

        std::array<REAL, trim_ndim> real_trim_indices = {
            trim_kernel.at(index, 0), trim_kernel.at(index, 1),
            trim_kernel.at(index, 2)};

        trim_indices.at(0) = real_trim_indices[0];
        trim_indices.at(1) = real_trim_indices[1];
        trim_indices.at(2) = real_trim_indices[2];

        std::array<INT, trim_ndim> normalized_trim_indices =
            interp_utils::bin_uniform_sub_indices(
                real_trim_indices, trim_dims_arr, d_test_error_propagate_ptr);

        auto result = grid_func(props.at(0), props.at(1),
                                normalized_trim_indices, trim_dims_arr);

        expected_value.at(0) = result[0];
        expected_value.at(1) = result[1];
        expected_value.at(2) = result[2];
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROPS")),
      Access::read(rng_kernel0), Access::read(rng_kernel1),
      Access::write(Sym<REAL>("TRIM_INDICES")), Access::read(trim_rng_kernel),
      Access::write(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")))
      ->execute();

  test_error_propagate->check_and_throw(
      "Error in setting up uniform sub indices for calculating expected test "
      "results!");

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto props_extract = extract<ndim>("PROPS");

  auto trim_extract = extract<trim_ndim>("TRIM_INDICES");

  auto concatenator = ConcatenatorData(props_extract, trim_extract);

  auto interpolator_data =
      InterpolateData<trim_ndim, ndim, decltype(grid_func_data), trim_ndim>(
          dims_vec, ranges_vec, particle_group->sycl_target, grid_func_data,
          ExtrapolationType::continue_linear, trim_dims_vec);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value =
      extract<trim_ndim>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    REAL calculated_interpolation_value;
    REAL expected_interpolation_value;

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      for (int icomp = 0; icomp < trim_ndim; icomp++) {
        calculated_interpolation_value =
            calc_results_dat[(ipart * trim_ndim) + icomp];
        expected_interpolation_value =
            expect_results_dat[(ipart * trim_ndim) + icomp];

        EXPECT_DOUBLE_EQ(calculated_interpolation_value,
                         expected_interpolation_value);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(InterpolationTest, TRIM_DATA_PIPELINE_INTERP) {
  static constexpr int ndim = 2;
  static constexpr int trim_ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROPS"), ndim);
  particle_group->add_particle_dat(Sym<REAL>("TRIM_INDICES"), trim_ndim);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"),
                                   trim_ndim);

  // Setup the mock data.
  auto coeffs_data = trim_coefficient_values(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func_data = coeffs_data.get_grid_func_data();
  auto grid_func = coeffs_data.get_grid_func();
  auto trim_dims_vec = coeffs_data.get_trim_dims_vec();

  std::array<INT, trim_ndim> trim_dims_arr;
  for (int i = 0; i < trim_ndim; i++) {
    trim_dims_arr[i] = trim_dims_vec[i];
  }

  // Random number generator kernel
  auto rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(0.0, 1.0);

  auto rng_kernel0 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel1 = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);
  auto trim_rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), trim_ndim);

  auto test_error_propagate =
      std::make_shared<ErrorPropagate>(particle_group->sycl_target);
  auto d_test_error_propagate_ptr = test_error_propagate->device_ptr();

  particle_loop(
      particle_group,
      [=](auto index, auto props, auto prop0_kernel, auto prop1_kernel,
          auto trim_indices, auto trim_kernel, auto expected_value) {
        props.at(0) = prop0_kernel.at(index, 0);
        props.at(1) = prop1_kernel.at(index, 0);

        std::array<REAL, trim_ndim> real_trim_indices = {
            trim_kernel.at(index, 0), trim_kernel.at(index, 1),
            trim_kernel.at(index, 2)};

        trim_indices.at(0) = real_trim_indices[0];
        trim_indices.at(1) = real_trim_indices[1];
        trim_indices.at(2) = real_trim_indices[2];

        std::array<INT, trim_ndim> normalized_trim_indices =
            interp_utils::bin_uniform_sub_indices(
                real_trim_indices, trim_dims_arr, d_test_error_propagate_ptr);

        auto result = grid_func(props.at(0), props.at(1),
                                normalized_trim_indices, trim_dims_arr);

        expected_value.at(0) = result[0];
        expected_value.at(1) = result[1];
        expected_value.at(2) = result[2];
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROPS")),
      Access::read(rng_kernel0), Access::read(rng_kernel1),
      Access::write(Sym<REAL>("TRIM_INDICES")), Access::read(trim_rng_kernel),
      Access::write(Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")))
      ->execute();

  test_error_propagate->check_and_throw(
      "Error in setting up uniform sub indices for calculating expected test "
      "results!");

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto props_extract = extract<ndim>("PROPS");

  auto trim_extract = extract<trim_ndim>("TRIM_INDICES");

  auto concatenator = ConcatenatorData(props_extract, trim_extract);

  auto interpolator_data =
      InterpolateData<trim_ndim, ndim, decltype(grid_func_data), trim_ndim>(
          dims_vec, ranges_vec, particle_group->sycl_target, grid_func_data,
          ExtrapolationType::continue_linear, trim_dims_vec);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value =
      extract<trim_ndim>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    REAL calculated_interpolation_value;
    REAL expected_interpolation_value;

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      for (int icomp = 0; icomp < trim_ndim; icomp++) {
        calculated_interpolation_value =
            calc_results_dat[(ipart * trim_ndim) + icomp];
        expected_interpolation_value =
            expect_results_dat[(ipart * trim_ndim) + icomp];

        EXPECT_DOUBLE_EQ(calculated_interpolation_value,
                         expected_interpolation_value);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
