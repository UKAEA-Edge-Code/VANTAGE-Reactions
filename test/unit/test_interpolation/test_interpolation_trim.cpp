#include "../include/mock_interpolation_data.hpp"
#include "../include/mock_particle_group.hpp"
#include "../include/test_common.hpp"
#include "../include/test_vantage_reactions_utils.hpp"
#include <memory>
#include <random>

#define INTERPOLATION_TOLERANCE 1e-14

using namespace VANTAGE::Reactions;

TEST(InterpolationTest, TRIM_DATA_PIPELINE_EXACT) {
  static constexpr size_t ndim = 2;
  static constexpr size_t trim_ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto rng = std::mt19937(52234126 + rank);

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(NP::Sym<REAL>("PROPS"), ndim);
  particle_group->add_particle_dat(NP::Sym<REAL>("TRIM_INDICES"), trim_ndim);
  particle_group->add_particle_dat(
      NP::Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"), trim_ndim);

  // PANIC flag for TrimEval
  particle_group->remove_particle_dat(NP::Sym<INT>("REACTIONS_PANIC_FLAG"));
  particle_group->add_particle_dat(NP::Sym<INT>("REACTIONS_PANIC_FLAG"),
                                   trim_ndim);

  // Setup the mock data.
  std::uniform_real_distribution<REAL> uniform_dist_m1(0.0, 10.0);
  std::array<REAL, trim_ndim> random_grid_nums;
  for (int i = 0; i < trim_ndim; i++) {
    random_grid_nums[i] = uniform_dist_m1(rng);
  }

  auto coeffs_data =
      trim_coefficient_values(random_grid_nums, particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
  auto lower_bounds = coeffs_data.get_lower_bounds();
  auto upper_bounds = coeffs_data.get_upper_bounds();
  auto grid_func_data = coeffs_data.get_grid_func_data();
  auto grid_func = coeffs_data.get_grid_func();
  auto trim_dims_vec = coeffs_data.get_trim_dims_vec();

  std::array<size_t, ndim> dims_arr;
  for (int i = 0; i < ndim; i++) {
    dims_arr[i] = dims_vec[i];
  }

  auto h_coords_arr = std::make_shared<NP::BufferDevice<REAL>>(
      particle_group->sycl_target, coords_vec);
  auto d_coords_arr = h_coords_arr->ptr;

  std::array<INT, trim_ndim> trim_dims_arr;
  for (int i = 0; i < trim_ndim; i++) {
    trim_dims_arr[i] = trim_dims_vec[i];
  }

  // Random number generator kernel
  std::uniform_int_distribution<INT> uniform_dist_0(0, dims_vec[0] - 1);
  std::uniform_int_distribution<INT> uniform_dist_1(0, dims_vec[1] - 1);
  std::uniform_real_distribution<REAL> uniform_dist_2(0.0, 1.0);

  auto rng_kernel0 = NP::host_per_particle_block_rng<INT>(
      rng_lambda_wrapper_int(uniform_dist_0, rng), 1);
  auto rng_kernel1 = NP::host_per_particle_block_rng<INT>(
      rng_lambda_wrapper_int(uniform_dist_1, rng), 1);
  auto trim_rng_kernel = NP::host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), trim_ndim);

  particle_loop(
      particle_group,
      [=](auto index, auto props, auto prop0_kernel, auto prop1_kernel,
          auto trim_indices, auto trim_kernel, auto expected_value) {
        auto index0 = prop0_kernel.at(index, 0);
        auto index1 = prop1_kernel.at(index, 0);

        auto indices = std::array<INT, ndim>{index0, index1};

        props.at(0) = d_coords_arr[index0];
        props.at(1) = d_coords_arr[dims_arr[0] + index1];
        auto coords = std::array<REAL, ndim>{props.at(0), props.at(1)};

        auto current_count = index.get_loop_linear_index();

        std::array<REAL, trim_ndim> real_trim_indices = {
            trim_kernel.at(index, 0), trim_kernel.at(index, 1),
            trim_kernel.at(index, 2)};

        trim_indices.at(0) = real_trim_indices[0];
        trim_indices.at(1) = real_trim_indices[1];
        trim_indices.at(2) = real_trim_indices[2];

        std::array<INT, trim_ndim> normalized_trim_indices =
            interp_utils::bin_uniform_indices(real_trim_indices, trim_dims_arr);

        auto result =
            grid_func(coords, normalized_trim_indices, random_grid_nums);

        expected_value.at(0) = result[0];
        expected_value.at(1) = result[1];
        expected_value.at(2) = result[2];
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<REAL>("PROPS")), NP::Access::read(rng_kernel0),
      NP::Access::read(rng_kernel1),
      NP::Access::write(NP::Sym<REAL>("TRIM_INDICES")),
      NP::Access::read(trim_rng_kernel),
      NP::Access::write(NP::Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")))
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
  auto extract_expected_value =
      extract<trim_ndim>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NP::NDLocalArray<REAL, 2>>(
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

  auto rng = std::mt19937(52234126 + rank);

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(NP::Sym<REAL>("PROPS"), ndim);
  particle_group->add_particle_dat(NP::Sym<REAL>("TRIM_INDICES"), trim_ndim);
  particle_group->add_particle_dat(
      NP::Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"), trim_ndim);

  // PANIC flag for TrimEval
  particle_group->remove_particle_dat(NP::Sym<INT>("REACTIONS_PANIC_FLAG"));
  particle_group->add_particle_dat(NP::Sym<INT>("REACTIONS_PANIC_FLAG"),
                                   trim_ndim);

  // Setup the mock data.
  std::uniform_real_distribution<REAL> uniform_dist_m1(0.0, 10.0);
  std::array<REAL, trim_ndim> random_grid_nums;
  for (int i = 0; i < trim_ndim; i++) {
    random_grid_nums[i] = uniform_dist_m1(rng);
  }

  auto coeffs_data =
      trim_coefficient_values(random_grid_nums, particle_group->sycl_target);
  auto dims_vec = coeffs_data.get_dims_vec();
  auto coords_vec = coeffs_data.get_coords_flat_vec();
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
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(0.0, 1.0);

  auto rng_kernel0 = NP::host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel1 = NP::host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);
  auto trim_rng_kernel = NP::host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), trim_ndim);

  particle_loop(
      particle_group,
      [=](auto index, auto props, auto prop0_kernel, auto prop1_kernel,
          auto trim_indices, auto trim_kernel, auto expected_value) {
        props.at(0) = prop0_kernel.at(index, 0);
        props.at(1) = prop1_kernel.at(index, 0);
        auto coords = std::array<REAL, ndim>{props.at(0), props.at(1)};

        std::array<REAL, trim_ndim> real_trim_indices = {
            trim_kernel.at(index, 0), trim_kernel.at(index, 1),
            trim_kernel.at(index, 2)};

        trim_indices.at(0) = real_trim_indices[0];
        trim_indices.at(1) = real_trim_indices[1];
        trim_indices.at(2) = real_trim_indices[2];

        std::array<INT, trim_ndim> normalized_trim_indices =
            interp_utils::bin_uniform_indices(real_trim_indices, trim_dims_arr);

        auto result =
            grid_func(coords, normalized_trim_indices, random_grid_nums);

        expected_value.at(0) = result[0];
        expected_value.at(1) = result[1];
        expected_value.at(2) = result[2];
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<REAL>("PROPS")), NP::Access::read(rng_kernel0),
      NP::Access::read(rng_kernel1),
      NP::Access::write(NP::Sym<REAL>("TRIM_INDICES")),
      NP::Access::read(trim_rng_kernel),
      NP::Access::write(NP::Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")))
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
  auto extract_expected_value =
      extract<trim_ndim>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NP::NDLocalArray<REAL, 2>>(
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

TEST(InterpolationTest, TRIM_DATA_ASYMMETRIC) {
  static constexpr int ndim = 2;
  static constexpr int trim_ndim = 3;

  auto particle_group = create_test_particle_group(1e3);

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;

  auto rng = std::mt19937(52234126 + rank);

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(NP::Sym<REAL>("PROPS"), ndim);
  particle_group->add_particle_dat(NP::Sym<REAL>("TRIM_INDICES"), trim_ndim);
  particle_group->add_particle_dat(
      NP::Sym<REAL>("EXPECTED_INTERPOLATION_VALUE"), trim_ndim);

  // PANIC flag for TrimEval
  particle_group->remove_particle_dat(NP::Sym<INT>("REACTIONS_PANIC_FLAG"));
  particle_group->add_particle_dat(NP::Sym<INT>("REACTIONS_PANIC_FLAG"),
                                   trim_ndim);

  // Setup the mock data.
  std::uniform_real_distribution<REAL> uniform_dist_m1(0.0, 10.0);
  std::array<REAL, trim_ndim> random_grid_nums;
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

  std::array<INT, trim_ndim> trim_dims_arr;
  for (int i = 0; i < trim_ndim; i++) {
    trim_dims_arr[i] = trim_dims_vec[i];
  }

  // Random number generator kernel
  std::uniform_real_distribution<REAL> uniform_dist_0(lower_bounds[0],
                                                      upper_bounds[0]);
  std::uniform_real_distribution<REAL> uniform_dist_1(lower_bounds[1],
                                                      upper_bounds[1]);
  std::uniform_real_distribution<REAL> uniform_dist_2(0.0, 1.0);

  auto rng_kernel0 = NP::host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_0, rng), 1);
  auto rng_kernel1 = NP::host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_1, rng), 1);
  auto trim_rng_kernel = NP::host_per_particle_block_rng<REAL>(
      rng_lambda_wrapper_real(uniform_dist_2, rng), trim_ndim);

  particle_loop(
      particle_group,
      [=](auto index, auto props, auto prop0_kernel, auto prop1_kernel,
          auto trim_indices, auto trim_kernel, auto expected_value) {
        props.at(0) = prop0_kernel.at(index, 0);
        props.at(1) = prop1_kernel.at(index, 0);
        auto coords = std::array<REAL, ndim>{props.at(0), props.at(1)};

        std::array<REAL, trim_ndim> real_trim_indices = {
            trim_kernel.at(index, 0), trim_kernel.at(index, 1),
            trim_kernel.at(index, 2)};

        trim_indices.at(0) = real_trim_indices[0];
        trim_indices.at(1) = real_trim_indices[1];
        trim_indices.at(2) = real_trim_indices[2];

        std::array<INT, trim_ndim> normalized_trim_indices =
            interp_utils::bin_uniform_indices(real_trim_indices, trim_dims_arr);

        auto result =
            grid_func(coords, normalized_trim_indices, random_grid_nums);

        expected_value.at(0) = result[0];
        expected_value.at(1) = result[1];
        expected_value.at(2) = result[2];
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<REAL>("PROPS")), NP::Access::read(rng_kernel0),
      NP::Access::read(rng_kernel1),
      NP::Access::write(NP::Sym<REAL>("TRIM_INDICES")),
      NP::Access::read(trim_rng_kernel),
      NP::Access::write(NP::Sym<REAL>("EXPECTED_INTERPOLATION_VALUE")))
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
  auto extract_expected_value =
      extract<trim_ndim>("EXPECTED_INTERPOLATION_VALUE");

  auto concat_data_calc = DataCalculator(pipeline);
  auto expect_data_calc = DataCalculator(extract_expected_value);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = concat_data_calc.get_data_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    auto calc_pre_req_data = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape);

    shape = expect_data_calc.get_data_size();
    auto expect_pre_req_data = std::make_shared<NP::NDLocalArray<REAL, 2>>(
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
