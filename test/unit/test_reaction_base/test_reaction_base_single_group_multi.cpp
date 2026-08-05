#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(LinearReactionBase, single_group_multi_reaction) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);

  auto test_reaction1 =
      TestReaction<0>(particle_group->sycl_target, 1, 0, std::array<int, 0>{});

  auto test_reaction2 =
      TestReaction<0>(particle_group->sycl_target, 1, 0, std::array<int, 0>{});

  const INT num_products_per_parent = 1;

  auto test_reaction3 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, 2, 0, std::array<int, 1>{1});

  std::vector<std::shared_ptr<AbstractReaction>> reactions{};
  reactions.push_back(std::make_shared<TestReaction<0>>(test_reaction1));
  reactions.push_back(std::make_shared<TestReaction<0>>(test_reaction2));
  reactions.push_back(std::make_shared<TestReaction<1>>(test_reaction3));

  int cell_count = particle_group->domain->mesh->get_cell_count();
  int num_reactions = static_cast<int>(reactions.size());

  auto parent_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto particle_subgroup = particle_sub_group(particle_group);

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->calculate_rates(particle_subgroup, i, i + 1);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->apply(particle_subgroup, i, i + 1, 0.1,
                                 descendant_particles);
    }

    auto internal_state =
        particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      if (internal_state->at(rowx, 0) == 0) {
        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.6); //, 1e-12);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
