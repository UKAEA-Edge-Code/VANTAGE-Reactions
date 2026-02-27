#include "include/mock_interpolation_data.hpp"
#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>

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

  EXPECT_DOUBLE_EQ(left_most_index, 0);
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

  EXPECT_DOUBLE_EQ(left_most_index, dims_vec[0]);
}

TEST(ExtrapolationTest, REACTION_DATA_1D_LINEAR_OVER_TYPE_0) {
  // Interpolation points
  REAL prop_interp_0 = 10.1e18;
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("PROP0")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D_linear();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  auto expected_interp_value = coeffs_data.grid_func(prop_interp_0);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_1D_LINEAR_UNDER_TYPE_0) {
  // Interpolation points
  REAL prop_interp_0 = 0.3e18;
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("PROP0")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D_linear();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  auto expected_interp_value = coeffs_data.grid_func(prop_interp_0);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_1D_UNDER_TYPE_0) {
  // Interpolation points
  REAL prop_interp_0 = 0.8e18;
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("PROP0")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  auto expected_interp_value = coeffs_data.grid_func(prop_interp_0);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      // log_10(x) grid function leads to severly limited extrapolation but the
      // test is included for posterity and demonstration purposes.
      EXPECT_NEAR(
          relative_error(calculated_interp_value, expected_interp_value), 0.0,
          1e-1);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_1D_LINEAR_OVER_TYPE_1) {
  // Interpolation points
  REAL prop_interp_0 = 12.7e18;
  REAL expected_interp_value = 0.0;
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("PROP0")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D_linear();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_1D_LINEAR_UNDER_TYPE_1) {
  // Interpolation points
  REAL prop_interp_0 = 0.55e18;
  REAL expected_interp_value = 0.0;
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("PROP0")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D_linear();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_1D_LINEAR_OVER_TYPE_2) {
  // Interpolation points
  REAL prop_interp_0 = 15.3e18;
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("PROP0")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D_linear();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  REAL expected_interp_value =
      coeffs_data.grid_func(ranges_vec[dims_vec[0] - 1]);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_1D_LINEAR_UNDER_TYPE_2) {
  // Interpolation points
  REAL prop_interp_0 = 0.3e18;
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("PROP0")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_1D_linear();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  REAL expected_interp_value = coeffs_data.grid_func(ranges_vec[0]);

  auto prop0_extract = extract<1>("PROP0");
  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(prop0_extract, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_0) {
  // Interpolation points
  REAL prop_interp_0 = 10.2e18; // over the range
  REAL prop_interp_1 = 7.8;     // under the range
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);

  particle_loop(
      particle_group,
      [=](auto prop0, auto prop1) {
        prop0.at(0) = prop_interp_0;
        prop1.at(0) = prop_interp_1;
      },
      Access::write(Sym<REAL>("PROP0")), Access::write(Sym<REAL>("PROP1")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  REAL expected_interp_value =
      coeffs_data.grid_func(prop_interp_0, prop_interp_1);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_1) {
  // Interpolation points
  REAL prop_interp_0 = 13.7e18; // over the range
  REAL prop_interp_1 = 4.5;     // under the range
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);

  particle_loop(
      particle_group,
      [=](auto prop0, auto prop1) {
        prop0.at(0) = prop_interp_0;
        prop1.at(0) = prop_interp_1;
      },
      Access::write(Sym<REAL>("PROP0")), Access::write(Sym<REAL>("PROP1")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  REAL expected_interp_value = 0.0;

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_2) {
  // Interpolation points
  REAL prop_interp_0 = 8.4e18; // over the range
  REAL prop_interp_1 = 0.1;    // under the range
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);

  particle_loop(
      particle_group,
      [=](auto prop0, auto prop1) {
        prop0.at(0) = prop_interp_0;
        prop1.at(0) = prop_interp_1;
      },
      Access::write(Sym<REAL>("PROP0")), Access::write(Sym<REAL>("PROP1")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  REAL expected_interp_value = coeffs_data.grid_func(
      ranges_vec[dims_vec[0] - 1], ranges_vec[dims_vec[0]]);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_3D_OVER_UNDER_OVER_UNDER_TYPE_0) {
  // Interpolation points
  REAL prop_interp_0 = 8.3e18; // over the range
  REAL prop_interp_1 = 8.9;    // under the range
  REAL prop_interp_2 = 3.3e2;  // over the range
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);

  particle_loop(
      particle_group,
      [=](auto prop0, auto prop1, auto prop2) {
        prop0.at(0) = prop_interp_0;
        prop1.at(0) = prop_interp_1;
        prop2.at(0) = prop_interp_2;
      },
      Access::write(Sym<REAL>("PROP0")), Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("PROP2")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_3D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  auto expected_interp_value =
      coeffs_data.grid_func(prop_interp_0, prop_interp_1, prop_interp_2);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_3D_OVER_UNDER_OVER_UNDER_TYPE_1) {
  // Interpolation points
  REAL prop_interp_0 = 10.3e18; // over the range
  REAL prop_interp_1 = 5.9;     // under the range
  REAL prop_interp_2 = 4.3e2;   // over the range
  REAL expected_interp_value = 0.0;
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);

  particle_loop(
      particle_group,
      [=](auto prop0, auto prop1, auto prop2) {
        prop0.at(0) = prop_interp_0;
        prop1.at(0) = prop_interp_1;
        prop2.at(0) = prop_interp_2;
      },
      Access::write(Sym<REAL>("PROP0")), Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("PROP2")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_3D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculate_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculate_interp_value, expected_interp_value);
    }
  }
}

TEST(ExtrapolationTest, REACTION_DATA_3D_OVER_UNDER_OVER_UNDER_TYPE_2) {
  // Interpolation points
  REAL prop_interp_0 = 10.3e18; // over the range
  REAL prop_interp_1 = 5.9;     // under the range
  REAL prop_interp_2 = 4.3e2;   // over the range
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);

  particle_loop(
      particle_group,
      [=](auto prop0, auto prop1, auto prop2) {
        prop0.at(0) = prop_interp_0;
        prop1.at(0) = prop_interp_1;
        prop2.at(0) = prop_interp_2;
      },
      Access::write(Sym<REAL>("PROP0")), Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("PROP2")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_3D();
  auto dims_vec = coeffs_data.get_dims_vec();
  auto ranges_vec = coeffs_data.get_ranges_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();

  REAL expected_interp_value = coeffs_data.grid_func(
      ranges_vec[dims_vec[0] - 1], ranges_vec[dims_vec[0]],
      ranges_vec[dims_vec[0] + dims_vec[1] + dims_vec[2] - 1]);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data = InterpolateData<ndim>(
      dims_vec, ranges_vec, grid, sycl_target, extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto pipeline_data_calc = DataCalculator(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      auto calculated_interp_value = results_dat[ipart * dim_size];

      EXPECT_DOUBLE_EQ(calculated_interp_value, expected_interp_value);
    }
  }
}