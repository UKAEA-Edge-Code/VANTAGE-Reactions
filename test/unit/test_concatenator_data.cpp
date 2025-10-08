#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ConcatenatorData, custom_sources) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_reaction = LinearReactionBase<
      0, TestReactionData, TestReactionDataCalcKernels<0>,
      DataCalculator<ConcatenatorData<TestReactionData, TestReactionData>>>(

      particle_group->sycl_target, 0, std::array<int, 0>{},
      TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
      DataCalculator<ConcatenatorData<TestReactionData, TestReactionData>>(
          ConcatenatorData(TestReactionData(3.0), TestReactionData(4.0))));

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
