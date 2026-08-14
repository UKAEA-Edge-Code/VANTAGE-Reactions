#include "../include/mock_interpolation_data.hpp"
#include "../include/mock_particle_group.hpp"
#include "../include/test_common.hpp"
#include "../include/test_vantage_reactions_utils.hpp"

#define EXTRAPOLATION_TOLERANCE 1e-14

using namespace VANTAGE::Reactions;

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_0) {
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP0"), 1);
  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP1"), 1);
  particle_group->add_particle_dat(
      NP::Sym<NP::REAL>("EXPECTED_EXTRAPOLATION_VALUE"), 1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_2D being f(x1, x2) = x1 * x2.
  std::uniform_real_distribution<NP::REAL> uniform_dist_0(
      upper_bounds[0], std::sqrt(std::numeric_limits<NP::REAL>::max()));
  std::uniform_real_distribution<NP::REAL> uniform_dist_1(
      -(std::sqrt(std::numeric_limits<NP::REAL>::max())), lower_bounds[1]);

  auto rng_kernel_0 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel_1 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto prop1, auto expected_value, auto kernel0,
          auto kernel1) {
        prop0.at(0) = kernel0.at(index, 0);
        prop1.at(0) = kernel1.at(index, 0);
        auto coords = std::array<NP::REAL, ndim>{prop0.at(0), prop1.at(0)};
        expected_value.at(0) = grid_func(coords);
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<NP::REAL>("PROP0")),
      NP::Access::write(NP::Sym<NP::REAL>("PROP1")),
      NP::Access::write(NP::Sym<NP::REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      NP::Access::read(rng_kernel_0), NP::Access::read(rng_kernel_1))
      ->execute();

  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::continue_linear;
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
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

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_1) {
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP0"), 1);
  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP1"), 1);
  particle_group->add_particle_dat(
      NP::Sym<NP::REAL>("EXPECTED_EXTRAPOLATION_VALUE"), 1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_2D being f(x1, x2) = x1 * x2.
  std::uniform_real_distribution<NP::REAL> uniform_dist_0(
      upper_bounds[0], std::sqrt(std::numeric_limits<NP::REAL>::max()));
  std::uniform_real_distribution<NP::REAL> uniform_dist_1(
      -(std::sqrt(std::numeric_limits<NP::REAL>::max())), lower_bounds[1]);

  auto rng_kernel_0 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel_1 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto prop1, auto expected_value, auto kernel0,
          auto kernel1) {
        prop0.at(0) = kernel0.at(index, 0);
        prop1.at(0) = kernel1.at(index, 0);
        expected_value.at(0) = 0.0; // ExtrapolationType::clamp_to_zero
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<NP::REAL>("PROP0")),
      NP::Access::write(NP::Sym<NP::REAL>("PROP1")),
      NP::Access::write(NP::Sym<NP::REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      NP::Access::read(rng_kernel_0), NP::Access::read(rng_kernel_1))
      ->execute();

  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_zero;
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
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

TEST(ExtrapolationTest, REACTION_DATA_2D_OVER_UNDER_TYPE_2) {
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP0"), 1);
  particle_group->add_particle_dat(NP::Sym<NP::REAL>("PROP1"), 1);
  particle_group->add_particle_dat(
      NP::Sym<NP::REAL>("EXPECTED_EXTRAPOLATION_VALUE"), 1);

  // Setup the mock data.
  auto coeffs_data = coefficient_values_2D(particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto grid = coeffs_data.get_coeffs_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func = coeffs_data.get_grid_func();
  auto bounds_arr =
      std::array<NP::REAL, ndim>{upper_bounds[0], lower_bounds[1]};
  auto grid_func_data = coeffs_data.get_grid_func_data();

  // Random number generator kernel
  std::mt19937 rng = std::mt19937(52234126 + rank);
  // The special limits on the upper and lower bounds are due to the grid_func
  // from coefficient_values_2D being f(x1, x2) = x1 * x2.
  std::uniform_real_distribution<NP::REAL> uniform_dist_0(
      upper_bounds[0], std::sqrt(std::numeric_limits<NP::REAL>::max()));
  std::uniform_real_distribution<NP::REAL> uniform_dist_1(
      -(std::sqrt(std::numeric_limits<NP::REAL>::max())), lower_bounds[1]);

  auto rng_kernel_0 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel_1 = NP::host_per_particle_block_rng<NP::REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);

  particle_loop(
      particle_group,
      [=](auto index, auto prop0, auto prop1, auto expected_value, auto kernel0,
          auto kernel1) {
        prop0.at(0) = kernel0.at(index, 0);
        prop1.at(0) = kernel1.at(index, 0);
        expected_value.at(0) =
            grid_func(bounds_arr); // ExtrapolationType::clamp_to_edge
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<NP::REAL>("PROP0")),
      NP::Access::write(NP::Sym<NP::REAL>("PROP1")),
      NP::Access::write(NP::Sym<NP::REAL>("EXPECTED_EXTRAPOLATION_VALUE")),
      NP::Access::read(rng_kernel_0), NP::Access::read(rng_kernel_1))
      ->execute();

  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto extrapolation_type = ExtrapolationType::clamp_to_edge;
  auto interpolator_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data,
      extrapolation_type);

  auto pipeline = pipe(concatenator, interpolator_data);
  auto extract_expected_value = extract<1>("EXPECTED_EXTRAPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);
    calc_pre_req_data->fill(0);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(
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
