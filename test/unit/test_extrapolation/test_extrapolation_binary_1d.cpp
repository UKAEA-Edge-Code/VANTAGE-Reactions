#include "../include/mock_interpolation_data.hpp"
#include "../include/mock_particle_group.hpp"
#include "../include/test_vantage_reactions_utils.hpp"
#include <gtest/gtest.h>

#define EXTRAPOLATION_TOLERANCE 1e-14

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ExtrapolationTest, BINARY_SEARCH_EXTRAPOLATE_UNDER) {
  auto test_values = coefficient_values_1D();
  auto dims_vec = test_values.get_dims_vec();
  auto coords_vec = test_values.get_coords_flat_vec();
  auto grid = test_values.get_coeffs_vec();

  REAL interp_point = 1.0e17;

  coords_vec.insert(coords_vec.begin(), -INF_INTERP_DOUBLE);
  coords_vec.push_back(INF_INTERP_DOUBLE);

  auto left_most_index = interp_utils::calc_floor_point_index(
      interp_point, coords_vec.data(), dims_vec[0] + 1);

  EXPECT_EQ(left_most_index, 0);
}

TEST(ExtrapolationTest, BINARY_SEARCH_EXTRAPOLATE_OVER) {
  auto test_values = coefficient_values_1D();
  auto dims_vec = test_values.get_dims_vec();
  auto coords_vec = test_values.get_coords_flat_vec();
  auto grid = test_values.get_coeffs_vec();

  REAL interp_point = 1.0e19;
  coords_vec.insert(coords_vec.begin(), -INF_INTERP_DOUBLE);
  coords_vec.push_back(INF_INTERP_DOUBLE);

  auto left_most_index = interp_utils::calc_floor_point_index(
      interp_point, coords_vec.data(), dims_vec[0] + 1);

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
  auto coeffs_data = coefficient_values_1D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      upper_bounds[0], 0.5 * std::numeric_limits<REAL>::max());

  auto rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto expected_value, auto kernel) {
        prop0.at(0) = kernel.at(index, 0);
        auto coords = std::array<REAL, ndim>{prop0.at(0)};
        expected_value.at(0) = grid_func(coords);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      auto calculated_extrapolation_value = calc_results_dat[ipart];
      auto expected_extrapolation_value = expect_results_dat[ipart];

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
  auto coeffs_data = coefficient_values_1D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special lower bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      -(0.5 * std::numeric_limits<REAL>::max()), lower_bounds[0]);

  auto rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto expected_value, auto kernel) {
        prop0.at(0) = kernel.at(index, 0);
        auto coords = std::array<REAL, ndim>{prop0.at(0)};
        expected_value.at(0) = grid_func(coords);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      auto calculated_extrapolation_value = calc_results_dat[ipart];
      auto expected_extrapolation_value = expect_results_dat[ipart];

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
  auto coeffs_data = coefficient_values_1D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      upper_bounds[0], (0.5 * std::numeric_limits<REAL>::max()));

  auto rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist, rng), 1);

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
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      auto calculated_extrapolation_value = calc_results_dat[ipart];
      auto expected_extrapolation_value = expect_results_dat[ipart];

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
  auto coeffs_data = coefficient_values_1D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      -(0.5 * std::numeric_limits<REAL>::max()), lower_bounds[0]);

  auto rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist, rng), 1);

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
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      auto calculated_extrapolation_value = calc_results_dat[ipart];
      auto expected_extrapolation_value = expect_results_dat[ipart];

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
  auto coeffs_data = coefficient_values_1D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto bounds_arr = std::array<REAL, ndim>{upper_bounds[0]};
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      bounds_arr[0], (0.5 * std::numeric_limits<REAL>::max()));

  auto rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto expected_value, auto kernel) {
        prop0.at(0) = kernel.at(index, 0);
        expected_value.at(0) =
            grid_func(bounds_arr); // ExtrapolationType::clamp_to_edge;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      auto calculated_extrapolation_value = calc_results_dat[ipart];
      auto expected_extrapolation_value = expect_results_dat[ipart];

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
  auto coeffs_data = coefficient_values_1D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto bounds_arr = std::array<REAL, ndim>{lower_bounds[0]};
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);

  // The special upper bound is due to the grid_func from
  // coefficient_values_1D being f(x) = 2*x.
  std::uniform_real_distribution<REAL> uniform_dist(
      -(0.5 * std::numeric_limits<REAL>::max()), bounds_arr[0]);

  auto rng_kernel = host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto expected_value, auto kernel) {
        prop0.at(0) = kernel.at(index, 0);
        expected_value.at(0) =
            grid_func(bounds_arr); // ExtrapolationType::clamp_to_edge;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("PROP0")),
      Access::write(Sym<REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      Access::read(rng_kernel))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    concat_data_calc.fill_buffer(calc_pre_req_data, particle_sub_group, i,
                                 i + 1);

    expect_data_calc.fill_buffer(expect_pre_req_data, particle_sub_group, i,
                                 i + 1);

    auto calc_results_dat = calc_pre_req_data->get();
    auto expect_results_dat = expect_pre_req_data->get();

    EXPECT_EQ(calc_pre_req_data->index.shape[0], n_part_cell);
    EXPECT_EQ(expect_pre_req_data->index.shape[0], n_part_cell);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      auto calculated_extrapolation_value = calc_results_dat[ipart];
      auto expected_extrapolation_value = expect_results_dat[ipart];

      EXPECT_DOUBLE_EQ(calculated_extrapolation_value,
                       expected_extrapolation_value);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
