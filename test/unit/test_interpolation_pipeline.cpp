#include "include/mock_particle_group.hpp"
#include "reactions_lib/reaction_data/interpolate_data.hpp"
#include "include/mock_interpolation_data.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

// This is a prototype test for how I expect the pipelining to work for
// interpolation but I haven't gotten it to work yet.
TEST(InterpolationTest, REACTION_DATA_2D_PIPELINE) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1;
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  // This is a bit of a workaround for the fact that currently I can't use
  // PipelineData to pass a ConcatenatorData object that contains data from
  // multiple 1D ExtractorData objects and pass that to an InterpolateData
  // object.
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);

  particle_loop(
      particle_group,
      [=](auto prop1, auto prop2) {
        prop1.at(0) = prop_interp_0;
        prop2.at(0) = prop_interp_1;
      },
      Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("PROP2")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_2D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  // The input and output ndims being 2  for the InterpolateData (for this
  // specific unit test) is because PipelineData only presents the output ndims
  // of the last object in the pipeline when DataCalculator queries the
  // ndims of the passed PipelineData object. This is fine if either all objects
  // in the pipeline have the same output ndims OR if the last output ndims is
  // greater than the output ndims of any preceeding object in the pipeline.
  // For the current implementation of InterpolateData, I would
  // prefer input ndim be equal to the dimensionality of the grid.
  // and output ndim being 1 but this works fine for testing. (I'm just storing
  // the result in the 0th index of the array returned by
  // InterpolateData.calc_data and test against that).
  auto concatenator = ConcatenatorData<ExtractorData<1>, ExtractorData<1>>(ExtractorData<1>(Sym<REAL>("PROP1")), ExtractorData<1>(Sym<REAL>("PROP2")));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto pipeline_data_calc = DataCalculator<decltype(pipeline)>(pipeline);

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

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i,
                                    i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      // printf("particle (%d) in cell (%d): %e\n", ipart, i, results_dat[(ipart * dim_size)]);
      EXPECT_DOUBLE_EQ(results_dat[(ipart * dim_size)], expected_interp_value);
    }
  }
}
