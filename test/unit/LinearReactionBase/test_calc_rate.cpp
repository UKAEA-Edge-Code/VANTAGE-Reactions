#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(LinearReactionBase, calc_rate) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(
      particle_group, [=](auto ISTATE) { return (ISTATE[0] == 0); },
      Access::read(Sym<INT>("INTERNAL_STATE")));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 0;

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0, std::array<int, 0>{});

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);

    const int nrow = position->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(tot_reaction_rate->at(rowx, 0), 2 * test_rate)
          << "calc_rate did not set TOT_REACTION_RATE correctly...";
    }
  }

  particle_group->domain->mesh->free();
}