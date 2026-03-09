#include "include/mock_interpolation_data.hpp"
#include "include/mock_particle_group.hpp"
#include "include/test_vantage_reactions_utils.hpp"
#include <gtest/gtest.h>
#include <memory>
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
  auto coeffs_data = coefficient_values_1D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist(lower_bounds[0],
                                                    upper_bounds[0]);
  auto rng_lambda = [&]() -> REAL {
    REAL rng_sample = 0.0;
    do {
      rng_sample = uniform_dist(rng);
    } while (rng_sample == 0.0);
    return rng_sample;
  };

  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, 1);

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

  auto interpolator_data = InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                                                 particle_group->sycl_target);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_interpolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_interpolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_interpolation_value = results_dat[ipart * 2];
      auto expected_interpolation_value = results_dat[(ipart * 2) + 1];

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
  auto coeffs_data = coefficient_values_2D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);

  auto rng_lambda_wrapper = [&](std::uniform_real_distribution<REAL> &dist) {
    auto rng_lambda = [&]() -> REAL {
      REAL rng_sample = 0.0;
      do {
        rng_sample = dist(rng);
      } while (rng_sample == 0.0);
      return rng_sample;
    };
    return rng_lambda;
  };

  auto rng_kernel_0 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_0), 1);
  auto rng_kernel_1 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_1), 1);

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

  auto interpolator_data = InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                                                 particle_group->sycl_target);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_interpolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_interpolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_interpolation_value = results_dat[ipart * 2];
      auto expected_interpolation_value = results_dat[(ipart * 2) + 1];

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
  auto coeffs_data = coefficient_values_3D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(lower_bounds[2],
                                                      upper_bounds[2]);

  auto rng_lambda_wrapper = [&](std::uniform_real_distribution<REAL> &dist) {
    auto rng_lambda = [&]() -> REAL {
      REAL rng_sample = 0.0;
      do {
        rng_sample = dist(rng);
      } while (rng_sample == 0.0);
      return rng_sample;
    };
    return rng_lambda;
  };

  auto rng_kernel_0 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_0), 1);
  auto rng_kernel_1 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_1), 1);
  auto rng_kernel_2 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_2), 1);

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

  auto interpolator_data = InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                                                 particle_group->sycl_target);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_interpolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_interpolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_interpolation_value = results_dat[ipart * 2];
      auto expected_interpolation_value = results_dat[(ipart * 2) + 1];

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
  auto coeffs_data = coefficient_values_4D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();

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

  auto rng_lambda_wrapper = [&](std::uniform_real_distribution<REAL> &dist) {
    auto rng_lambda = [&]() -> REAL {
      REAL rng_sample = 0.0;
      do {
        rng_sample = dist(rng);
      } while (rng_sample == 0.0);
      return rng_sample;
    };
    return rng_lambda;
  };

  auto rng_kernel_0 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_0), 1);
  auto rng_kernel_1 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_1), 1);
  auto rng_kernel_2 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_2), 1);
  auto rng_kernel_3 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_3), 1);

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

  auto interpolator_data = InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                                                 particle_group->sycl_target);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_interpolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_interpolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_interpolation_value = results_dat[ipart * 2];
      auto expected_interpolation_value = results_dat[(ipart * 2) + 1];

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
  auto coeffs_data = coefficient_values_5D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();

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

  auto rng_lambda_wrapper = [&](std::uniform_real_distribution<REAL> &dist) {
    auto rng_lambda = [&]() -> REAL {
      REAL rng_sample = 0.0;
      do {
        rng_sample = dist(rng);
      } while (rng_sample == 0.0);
      return rng_sample;
    };
    return rng_lambda;
  };

  auto rng_kernel_0 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_0), 1);
  auto rng_kernel_1 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_1), 1);
  auto rng_kernel_2 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_2), 1);
  auto rng_kernel_3 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_3), 1);
  auto rng_kernel_4 =
      host_per_particle_block_rng<REAL>(rng_lambda_wrapper(uniform_dist_4), 1);

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

  auto interpolator_data = InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                                                 particle_group->sycl_target);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_interpolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_interpolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_interpolation_value = results_dat[ipart * 2];
      auto expected_interpolation_value = results_dat[(ipart * 2) + 1];

      auto rel_error = relative_error(expected_interpolation_value,
                                      calculated_interpolation_value);
      EXPECT_NEAR(rel_error, 0.0, INTERPOLATION_TOLERANCE);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
