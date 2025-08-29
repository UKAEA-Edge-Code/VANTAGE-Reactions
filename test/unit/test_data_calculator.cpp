#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(DataCalculator, custom_sources) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<TestReactionData, TestReactionData>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator<TestReactionData, TestReactionData>(
              TestReactionData(3.0), TestReactionData(4.0)));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                                          descendant_particles);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto source_density =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), i);
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), 3.0);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0), 4.0);
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(DataCalculator, mixed_multi_dim) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto energy_dat_0 = FixedRateData(0.1);
  auto energy_dat_1 = FixedRateData(1.5);

  // std::mt19937 rng = std::mt19937(std::random_device{}());
  // const double extents[1] = {1.0};
  auto rng_lambda = [&]() -> REAL { return 1.0; };
  auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, N_total);

  // auto constant_rate_cross_section = ConstantRateCrossSection(1.0);
  auto vel_dat = FilteredMaxwellianSampler<2>(2.0, rng_kernel);

  // (1D, 2D, 1D) order of ReactionData dimensions
  auto data_calc_obj = DataCalculator<decltype(energy_dat_0), decltype(vel_dat),
                                      decltype(energy_dat_1)>(
      energy_dat_0, vel_dat, energy_dat_1);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      particle_group->sycl_target, 0, data_calc_obj.get_data_size());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    data_calc_obj.fill_buffer(pre_req_data, particle_sub_group, i, i + 1);

    auto temp_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0]; ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      EXPECT_DOUBLE_EQ(temp_dat[(ipart * dim_size) + 0], 0.1);
      EXPECT_DOUBLE_EQ(temp_dat[(ipart * dim_size) + 1],
                       1); // FLUID_FLOW_SPEED x
      EXPECT_DOUBLE_EQ(temp_dat[(ipart * dim_size) + 2],
                       3); // FLUID_FLOW_SPEED y
      EXPECT_DOUBLE_EQ(temp_dat[(ipart * dim_size) + 3], 1.5);
    }
  }

  particle_group->domain->mesh->free();
}
