#include "include/mock_interpolation_data.hpp"
#include "include/mock_particle_group.hpp"
#include "reactions_lib/concatenator_data.hpp"
#include "reactions_lib/data_calculator.hpp"
#include "reactions_lib/reaction_data/extractor_data.hpp"
#include "reactions_lib/reaction_data/interpolate_data.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/containers/nd_local_array.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(InterpolationTest, REACTION_DATA_2D_PIPELINE) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1;
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e5);

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

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_2D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract);

  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);

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

      EXPECT_DOUBLE_EQ(results_dat[(ipart * dim_size)], expected_interp_value);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_3D_PIPELINE) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL prop_interp_2 = 2.3e4;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1 * prop_interp_2;
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e5);

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

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_3D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);

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

      EXPECT_DOUBLE_EQ(results_dat[(ipart * dim_size)], expected_interp_value);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_4D_PIPELINE) {
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

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP3"), 1);

  particle_loop(
      particle_group,
      [=](auto prop0, auto prop1, auto prop2, auto prop3) {
        prop0.at(0) = prop_interp_0;
        prop1.at(0) = prop_interp_1;
        prop2.at(0) = prop_interp_2;
        prop3.at(0) = prop_interp_3;
      },
      Access::write(Sym<REAL>("PROP0")), Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("PROP2")), Access::write(Sym<REAL>("PROP3")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_4D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto prop3_extract = extract<1>("PROP3");
  auto concatenator = ConcatenatorData(prop0_extract, prop1_extract,
                                       prop2_extract, prop3_extract);

  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);

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
        sycl_target, buffer_size, shape[1]);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

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

TEST(InterpolationTest, REACTION_DATA_5D_PIPELINE) {
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

  particle_group->add_particle_dat(Sym<REAL>("PROP0"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP3"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP4"), 1);

  particle_loop(
      particle_group,
      [=](auto prop0, auto prop1, auto prop2, auto prop3, auto prop4) {
        prop0.at(0) = prop_interp_0;
        prop1.at(0) = prop_interp_1;
        prop2.at(0) = prop_interp_2;
        prop3.at(0) = prop_interp_3;
        prop4.at(0) = prop_interp_4;
      },
      Access::write(Sym<REAL>("PROP0")), Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("PROP2")), Access::write(Sym<REAL>("PROP3")),
      Access::write(Sym<REAL>("PROP4")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_5D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto prop3_extract = extract<1>("PROP3");
  auto prop4_extract = extract<1>("PROP4");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract,
                       prop3_extract, prop4_extract);

  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);

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
        sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

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
