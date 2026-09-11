#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(LinearReactionBase, calc_rate) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_subgroup = particle_sub_group(
      particle_group, [=](auto ISTATE) { return (ISTATE[0] == 0); },
      NP::Access::read(NP::Sym<INT>("INTERNAL_STATE")));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 0;

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0, std::array<int, 0>{});

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_subgroup, i, i + 1);
    test_reaction.calculate_rates(particle_subgroup, i, i + 1);

    auto position = particle_group->get_cell(NP::Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(NP::Sym<REAL>("TOT_REACTION_RATE"), i);

    const int nrow = position->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(tot_reaction_rate->at(rowx, 0), 2 * test_rate)
          << "calc_rate did not set TOT_REACTION_RATE correctly...";
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(LinearReactionBase, calc_var_rate) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_subgroup = particle_sub_group(particle_group);

  auto test_reaction = TestReactionVarRate(particle_group->sycl_target, 0);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_subgroup, i, i + 1);
    test_reaction.calculate_rates(particle_subgroup, i, i + 1);

    auto position = particle_group->get_cell(NP::Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(NP::Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = position->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(tot_reaction_rate->at(rowx, 0),
                       2 * position->at(rowx, 0))
          << "calc_rate dP not set TOT_REACTION_RATE correctly...";
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
