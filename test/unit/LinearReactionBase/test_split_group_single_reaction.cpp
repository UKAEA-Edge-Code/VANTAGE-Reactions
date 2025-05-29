#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(LinearReactionBase, split_group_single_reaction) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto loop = particle_loop(
      "set_internal_state", particle_group,
      [=](auto internal_state) { internal_state[0] = 2; },
      Access::write(Sym<INT>("INTERNAL_STATE")));

  auto particle_group_2 = create_test_particle_group(N_total);
  auto loop2 = particle_loop(
      "set_internal_state2", particle_group_2,
      [=](auto internal_state) { internal_state[0] = 3; },
      Access::write(Sym<INT>("INTERNAL_STATE")));

  loop->execute();
  loop2->execute();

  particle_group->add_particles_local(particle_group_2);

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction1 = TestReaction<0>(particle_group->sycl_target, 1, 2,
                                        std::array<int, 0>{});

  auto test_reaction2 = TestReaction<1>(particle_group->sycl_target, 2, 3,
                                        std::array<int, 1>{4});

  std::vector<std::shared_ptr<AbstractReaction>> reactions = {
      std::make_shared<TestReaction<0>>(test_reaction1),
      std::make_shared<TestReaction<1>>(test_reaction2)};
  std::vector<std::shared_ptr<ParticleSubGroup>> subgroups;

  auto parent_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  int cell_count = particle_group->domain->mesh->get_cell_count();
  int num_reactions = static_cast<int>(reactions.size());
  for (int i = 0; i < cell_count; i++) {
    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      auto sub_group_selector =
          make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
              Sym<INT>("INTERNAL_STATE"), (reaction + 2));
      auto particle_sub_group = sub_group_selector->make_marker_subgroup(
          std::make_shared<ParticleSubGroup>(particle_group));
      subgroups.push_back(particle_sub_group);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->run_rate_loop(subgroups[reaction], i, i + 1);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->descendant_product_loop(
          subgroups[reaction], i, i + 1, 0.1, descendant_particles);
    }

    auto internal_state =
        particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      if (internal_state->at(rowx, 0) == 2) {
        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.9);
      } else if (internal_state->at(rowx, 0) == 3) {
        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.8);
      }
    }
  }

  particle_group->domain->mesh->free();
  particle_group_2->domain->mesh->free();
  parent_particles->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}