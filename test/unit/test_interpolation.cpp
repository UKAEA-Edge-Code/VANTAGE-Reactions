#include "include/mock_particle_group.hpp"
#include "reactions_lib/reaction_data/interpolate_data.hpp"
#include "include/mock_interpolation_data.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

inline void
diagnostic_output(const int &particle_count_, const int &dim_index_,
                  const int &num_points_,
                  Access::LocalMemoryInterlaced::Write<size_t> origin_indices_,
                  Access::LocalMemoryInterlaced::Write<int> vertices_,
                  Access::LocalMemoryInterlaced::Write<REAL> func_evals_,
                  size_t *dims_vec_ptr_, REAL *ranges_vec_ptr_) {
  for (int i = 0; i < num_points_; i++) {
    if (particle_count_ == 0) {
      for (int j = dim_index_; j >= 0; j--) {
        size_t temp_binary_repr =
            origin_indices_.at(j) + binary_extract(vertices_.at(i), j);
        REAL ranges_val = ranges_vec_ptr_[interp_utils::range_index_on_device(
            temp_binary_repr, size_t(j), dims_vec_ptr_)];
        printf("%ld, %e\t", temp_binary_repr, ranges_val);
      }
      printf("%e\n", func_evals_.at(i));
    }
  }
  if (particle_count_ == 0) {
    printf("\n");
  }
};

TEST(InterpolationTest, BINARY_SEARCH_INTERPOLATE) {
  auto test_values = coefficient_values_1D();
  auto dims_vec = test_values.get_dims_vec();
  auto ranges_vec = test_values.get_ranges_flat_vec();
  auto grid = test_values.get_coeffs_vec();

  REAL interp_point = 4.1e18;

  ranges_vec.insert(ranges_vec.begin(), -INF);
  ranges_vec.push_back(INF);

  auto left_most_index = interp_utils::calc_closest_point_index(
      interp_point, ranges_vec.data(), dims_vec[0] + 1);

  ASSERT_DOUBLE_EQ(left_most_index, 4);
}

TEST(InterpolationTest, BINARY_SEARCH_EXTRAPOLATE_UNDER) {
  auto test_values = coefficient_values_1D();
  auto dims_vec = test_values.get_dims_vec();
  auto ranges_vec = test_values.get_ranges_flat_vec();
  auto grid = test_values.get_coeffs_vec();

  REAL interp_point = 1.0e17;

  ranges_vec.insert(ranges_vec.begin(), -INF);
  ranges_vec.push_back(INF);

  auto left_most_index = interp_utils::calc_closest_point_index(
      interp_point, ranges_vec.data(), dims_vec[0] + 1);

  ASSERT_DOUBLE_EQ(left_most_index, 0);
}

TEST(InterpolationTest, BINARY_SEARCH_EXTRAPOLATE_OVER) {
  auto test_values = coefficient_values_1D();
  auto dims_vec = test_values.get_dims_vec();
  auto ranges_vec = test_values.get_ranges_flat_vec();
  auto grid = test_values.get_coeffs_vec();

  REAL interp_point = 1.0e19;
  ranges_vec.insert(ranges_vec.begin(), -INF);
  ranges_vec.push_back(INF);

  auto left_most_index = interp_utils::calc_closest_point_index(
      interp_point, ranges_vec.data(), dims_vec[0] + 1);

  ASSERT_DOUBLE_EQ(left_most_index, dims_vec[0]);
}

TEST(InterpolationTest, REACTION_DATA_1D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL expected_interp_value = std::log10(prop_interp_0);
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_1D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, 1> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < 1; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      // Slightly different assertion test since function for the grid values is
      // log10(x), and grid spacing is sufficiently wide that linear
      // interpolation will not give an approximation that's compatible with
      // EXPECT_DOUBLE_EQ.
      EXPECT_NEAR(results_dat[(ipart * dim_size)], expected_interp_value, 1e-2);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_2D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1;
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group,
      [=](auto prop) {
        prop.at(0) = prop_interp_0;
        prop.at(1) = prop_interp_1;
      },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_2D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, 1> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < 1; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      EXPECT_DOUBLE_EQ(results_dat[(ipart * dim_size)], expected_interp_value);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_3D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL prop_interp_2 = 2.3e4;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1 * prop_interp_2;
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group,
      [=](auto prop) {
        prop.at(0) = prop_interp_0;
        prop.at(1) = prop_interp_1;
        prop.at(2) = prop_interp_2;
      },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_3D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, 1> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < 1; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      EXPECT_DOUBLE_EQ(results_dat[(ipart * dim_size)], expected_interp_value);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_4D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL prop_interp_2 = 2.3e4;
  REAL prop_interp_3 = 3.2e7;
  REAL expected_interp_value =
      prop_interp_0 * prop_interp_1 * prop_interp_2 * prop_interp_3;
  static constexpr int ndim = 4;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group,
      [=](auto prop) {
        prop.at(0) = prop_interp_0;
        prop.at(1) = prop_interp_1;
        prop.at(2) = prop_interp_2;
        prop.at(3) = prop_interp_3;
      },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_4D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, 1> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < 1; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      auto relative_error =
          std::abs(results_dat[(ipart * dim_size)] - expected_interp_value) /
          std::abs(expected_interp_value);

      // Linear interpolation seems to be running out of steam with regards to
      // precision hence the check is to see if the calculated value is within
      // 1e-6 for relative error.
      EXPECT_NEAR(relative_error, 0.0, 1e-6);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_5D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL prop_interp_2 = 2.3e4;
  REAL prop_interp_3 = 3.2e7;
  REAL prop_interp_4 = 2.07;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1 * prop_interp_2 *
                               prop_interp_3 * prop_interp_4;
  static constexpr int ndim = 5;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group,
      [=](auto prop) {
        prop.at(0) = prop_interp_0;
        prop.at(1) = prop_interp_1;
        prop.at(2) = prop_interp_2;
        prop.at(3) = prop_interp_3;
        prop.at(4) = prop_interp_4;
      },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_5D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, 1> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < 1; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      auto relative_error =
          std::abs(results_dat[(ipart * dim_size)] - expected_interp_value) /
          std::abs(expected_interp_value);

      // Linear interpolation seems to be running out of steam with regards to
      // precision hence the check is to see if the calculated value is within
      // 1e-6 for relative error.
      EXPECT_NEAR(relative_error, 0.0, 1e-6);
    }
  }
}
