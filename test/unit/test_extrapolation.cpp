#include "include/mock_interpolation_data.hpp"
#include "include/mock_particle_group.hpp"
#include "include/test_vantage_reactions_utils.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>

#define EXTRAPOLATION_TOLERANCE 1e-14

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ExtrapolationTest, BINARY_SEARCH_EXTRAPOLATE_UNDER) {
  auto test_values = coefficient_values_1D();
  auto dims_vec = test_values.get_dims_vec();
  auto ranges_vec = test_values.get_ranges_flat_vec();
  auto grid = test_values.get_coeffs_vec();

  REAL interp_point = 1.0e17;

  ranges_vec.insert(ranges_vec.begin(), -INF_INTERP_DOUBLE);
  ranges_vec.push_back(INF_INTERP_DOUBLE);

  auto left_most_index = interp_utils::calc_floor_point_index(
      interp_point, ranges_vec.data(), dims_vec[0] + 1);

  EXPECT_EQ(left_most_index, 0);
}

TEST(ExtrapolationTest, BINARY_SEARCH_EXTRAPOLATE_OVER) {
  auto test_values = coefficient_values_1D();
  auto dims_vec = test_values.get_dims_vec();
  auto ranges_vec = test_values.get_ranges_flat_vec();
  auto grid = test_values.get_coeffs_vec();

  REAL interp_point = 1.0e19;
  ranges_vec.insert(ranges_vec.begin(), -INF_INTERP_DOUBLE);
  ranges_vec.push_back(INF_INTERP_DOUBLE);

  auto left_most_index = interp_utils::calc_floor_point_index(
      interp_point, ranges_vec.data(), dims_vec[0] + 1);

  EXPECT_EQ(left_most_index, dims_vec[0]);
}

TEST(ExtrapolationTest, REACTION_DATA_1D_OVER_TYPE_0) {
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
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

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      upper_bounds[0], 0.5 * std::numeric_limits<REAL>::max());

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
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_extrapolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_extrapolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      auto rel_error = relative_error(expected_extrapolation_value,
                                      calculated_extrapolation_value);
      EXPECT_NEAR(rel_error, 0.0, EXTRAPOLATION_TOLERANCE);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_1D_UNDER_TYPE_0) {
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
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

  // The special lower bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      -(0.5 * std::numeric_limits<REAL>::max()), lower_bounds[0]);

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
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_extrapolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_extrapolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      auto rel_error = relative_error(expected_extrapolation_value,
                                      calculated_extrapolation_value);
      EXPECT_NEAR(rel_error, 0.0, EXTRAPOLATION_TOLERANCE);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_1D_OVER_TYPE_1) {
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      upper_bounds[0], (0.5 * std::numeric_limits<REAL>::max()));

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
        expected_value.at(0) = 0.0; // ExtrapolationType::clamp_to_zero;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_extrapolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_extrapolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_1D_UNDER_TYPE_1) {
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      -(0.5 * std::numeric_limits<REAL>::max()), lower_bounds[0]);

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
        expected_value.at(0) = 0.0; // ExtrapolationType::clamp_to_zero;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_extrapolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_extrapolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_1D_OVER_TYPE_2) {
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto upper_bound_0 = upper_bounds[0];

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      upper_bound_0, (0.5 * std::numeric_limits<REAL>::max()));

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
        expected_value.at(0) =
            grid_func(upper_bound_0); // ExtrapolationType::clamp_to_edge;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_extrapolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_extrapolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_1D_UNDER_TYPE_2) {
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto lower_bound_0 = lower_bounds[0];

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      -(0.5 * std::numeric_limits<REAL>::max()), lower_bound_0);

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
        expected_value.at(0) =
            grid_func(lower_bound_0); // ExtrapolationType::clamp_to_edge;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_extrapolation_expect =
      ConcatenatorData(pipeline, extract_expected_value);

  auto concat_data_calc = DataCalculator(concat_extrapolation_expect);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_0) {
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
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
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_2D being f(x1, x2) = x1 * x2.
  std::uniform_real_distribution<REAL> uniform_dist_0(
      upper_bounds[0], std::sqrt(std::numeric_limits<REAL>::max()));
  std::uniform_real_distribution<REAL> uniform_dist_1(
      -(std::sqrt(std::numeric_limits<REAL>::max())), lower_bounds[1]);

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
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

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
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      auto rel_error = relative_error(expected_extrapolation_value,
                                      calculated_extrapolation_value);
      EXPECT_NEAR(rel_error, 0.0, EXTRAPOLATION_TOLERANCE);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_1) {
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_2D being f(x1, x2) = x1 * x2.
  std::uniform_real_distribution<REAL> uniform_dist_0(
      upper_bounds[0], std::sqrt(std::numeric_limits<REAL>::max()));
  std::uniform_real_distribution<REAL> uniform_dist_1(
      -(std::sqrt(std::numeric_limits<REAL>::max())), lower_bounds[1]);

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
        expected_value.at(0) = 0.0; // ExtrapolationType::clamp_to_zero
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

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
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_2) {
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto upper_bound_0 = upper_bounds[0];
  auto lower_bound_1 = lower_bounds[1];

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_2D being f(x1, x2) = x1 * x2.
  std::uniform_real_distribution<REAL> uniform_dist_0(
      upper_bounds[0], std::sqrt(std::numeric_limits<REAL>::max()));
  std::uniform_real_distribution<REAL> uniform_dist_1(
      -(std::sqrt(std::numeric_limits<REAL>::max())), lower_bounds[1]);

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
        expected_value.at(0) =
            grid_func(upper_bound_0,
                      lower_bound_1); // ExtrapolationType::clamp_to_edge
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

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
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_3D_OVER_UNDER_OVER_UNDER_TYPE_0) {
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
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
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_3D being f(x1, x2, x3) = x1 * x2 * x3.
  std::uniform_real_distribution<REAL> uniform_dist_0(
      upper_bounds[0], std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0));
  std::uniform_real_distribution<REAL> uniform_dist_1(
      -(std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0)),
      lower_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(
      upper_bounds[2], std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0));

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
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1),
      Access::read(rng_kernel_2))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

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
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      auto rel_error = relative_error(expected_extrapolation_value,
                                      calculated_extrapolation_value);
      EXPECT_NEAR(rel_error, 0.0, EXTRAPOLATION_TOLERANCE);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_3D_OVER_UNDER_OVER_UNDER_TYPE_1) {
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_3D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_3D being f(x1, x2, x3) = x1 * x2 * x3.
  std::uniform_real_distribution<REAL> uniform_dist_0(
      upper_bounds[0], std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0));
  std::uniform_real_distribution<REAL> uniform_dist_1(
      -(std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0)),
      lower_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(
      upper_bounds[2], std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0));

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
        expected_value.at(0) = 0.0; // ExtrapolationType::clamp_to_zero
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("PROP1")), Access::write(Sym<REAL>("PROP2")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1),
      Access::read(rng_kernel_2))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

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
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ExtrapolationTest, REACTION_DATA_3D_OVER_UNDER_OVER_UNDER_TYPE_2) {
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);
  particle_group->add_particle_dat(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE"),
                                   1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_3D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto upper_bound_0 = upper_bounds[0];
  auto lower_bound_1 = lower_bounds[1];
  auto upper_bound_2 = upper_bounds[2];

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_3D being f(x1, x2, x3) = x1 * x2 * x3.
  std::uniform_real_distribution<REAL> uniform_dist_0(
      upper_bounds[0], std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0));
  std::uniform_real_distribution<REAL> uniform_dist_1(
      -(std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0)),
      lower_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(
      upper_bounds[2], std::pow(std::numeric_limits<REAL>::max(), 1.0 / 3.0));

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
        expected_value.at(0) =
            grid_func(upper_bound_0, lower_bound_1,
                      upper_bound_2); // ExtrapolationType::clamp_to_edge
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("PROP1")), Access::write(Sym<REAL>("PROP2")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel_0), Access::read(rng_kernel_1),
      Access::read(rng_kernel_2))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid,
                            particle_group->sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

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
    pre_req_data->fill(0);

    concat_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      // due to interleaving
      auto calculated_extrapolation_value = results_dat[ipart * 2];
      auto expected_extrapolation_value = results_dat[(ipart * 2) + 1];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
