#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_common.hpp"
#include <cmath>

using namespace VANTAGE::Reactions;

TEST(ReactionData, ArrheniusData) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto test_reaction =
      LinearReactionBase<0, ArrheniusData, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{},
          ArrheniusData(2.0, 1.5), TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<NP::ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);
  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);
    auto weight = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), i);
    const int nrow = weight->nrow;

    // The FLUID_TEMPERATURE is 2 for the mock group, so the rate coefficient
    // should be 2.0* 2**1.5
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 1.0 - 0.1 * 2.0 * std::pow(2, 1.5));
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
