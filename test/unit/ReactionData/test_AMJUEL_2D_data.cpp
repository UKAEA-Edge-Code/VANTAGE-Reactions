#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ReactionData, AMJUEL2DData) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();

  auto amjuel_data = AMJUEL2DData<2, 2>(
      3e12, 1.0, 1.0, 1.0,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}});

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DData<2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, amjuel_data,
          TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  // Rate calculated based on ne=3e18, T=2eV, with the evolved quantity
  // normalised to 3e12
  auto expected_rate = 3.880728735562758;
  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate); // 1e-15);
    }
  }

  particle_group->domain->mesh->free();
}