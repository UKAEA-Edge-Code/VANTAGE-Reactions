#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ReactionData, FixedCoefficientData) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction =
      LinearReactionBase<0, FixedCoefficientData, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{},
          FixedCoefficientData(2.0), TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);
  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    test_reaction.descendant_product_loop(particle_sub_group, i, i + 1, 0.1,
                                          descendant_particles);
    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    test_reaction.descendant_product_loop(particle_sub_group, i, i + 1, 0.1,
                                          descendant_particles);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);
    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.64);
    }
  }

  particle_group->domain->mesh->free();
}