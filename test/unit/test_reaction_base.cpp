#include "common_markers.hpp"
#include "compute_target.hpp"
#include "mock_reactions.hpp"
#include "particle_group.hpp"
#include "particle_spec.hpp"
#include "particle_sub_group.hpp"
#include "reaction_base.hpp"
#include "transformation_wrapper.hpp"
#include "typedefs.hpp"
#include <CL/sycl.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <ionisation_reactions/fixed_rate_ionisation.hpp>
#include <memory>

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

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 0,
      std::array<int, 0>{});

  test_reaction.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i);
    test_reaction.run_rate_loop(particle_sub_group, i);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);

    const int nrow = position->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(tot_reaction_rate->at(rowx, 0), 2 * test_rate)
          << "calc_rate did not set TOT_REACTION_RATE correctly...";
    }
  }

  particle_group->domain->mesh->free();
}

TEST(LinearReactionBase, calc_var_rate) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_reaction =
      TestReactionVarRate(particle_group->sycl_target,
                          Sym<REAL>("TOT_REACTION_RATE"), Sym<REAL>("POSITION"), 0);

  test_reaction.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i);
    test_reaction.run_rate_loop(particle_sub_group, i);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = position->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(tot_reaction_rate->at(rowx, 0), 2 * position->at(rowx, 0))
          << "calc_rate dP not set TOT_REACTION_RATE correctly...";
    }
  }

  particle_group->domain->mesh->free();
}

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

  auto test_reaction1 = TestReaction<0>(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 1, 2,
                                        std::array<int, 0>{});

  auto test_reaction2 = TestReaction<1>(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 2, 3,
                                        std::array<int, 1>{4});

  test_reaction1.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));
  test_reaction2.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  std::vector<std::shared_ptr<AbstractReaction>> reactions = {
      std::make_shared<TestReaction<0>>(test_reaction1),
      std::make_shared<TestReaction<1>>(test_reaction2)};
  std::vector<shared_ptr<ParticleSubGroup>> subgroups;

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
      reactions[reaction]->run_rate_loop(subgroups[reaction], i);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->descendant_product_loop(subgroups[reaction], i, 0.1,
                                                   descendant_particles);
    }

    auto internal_state =
        particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);
    auto weight =
        particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      if (internal_state->at(rowx, 0) == 2) {
        EXPECT_EQ(weight->at(rowx, 0), 0.9);
      } else if (internal_state->at(rowx, 0) == 3) {
        EXPECT_EQ(weight->at(rowx, 0), 0.8);
      }
    }
  }

  particle_group->domain->mesh->free();
  particle_group_2->domain->mesh->free();
  parent_particles->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(LinearReactionBase, single_group_multi_reaction) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);

  auto sub_group_selector =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("INTERNAL_STATE"), 0);

  auto particle_sub_group = sub_group_selector->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  auto test_reaction1 = TestReaction<0>(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 1, 0,
                                        std::array<int, 0>{});

  auto test_reaction2 = TestReaction<0>(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 1, 0,
                                        std::array<int, 0>{});

  const INT num_products_per_parent = 1;

  auto test_reaction3 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 2, 0,
      std::array<int, 1>{1});

  test_reaction1.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));
  test_reaction2.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));
  test_reaction3.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

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

    auto particle_sub_group =
        std::make_shared<ParticleSubGroup>(particle_group);

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->run_rate_loop(particle_sub_group, i);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->descendant_product_loop(particle_sub_group, i, 0.1,
                                                   descendant_particles);
    }

    auto internal_state =
        particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);
    auto weight =
        particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      if (internal_state->at(rowx, 0) == 0) {
        EXPECT_NEAR(weight->at(rowx, 0), 0.6, 1e-12);
      }
    }
  }

  particle_group->domain->mesh->free();
  parent_particles->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(IoniseReaction, calc_rate) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_reaction = FixedRateIonisation(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 1.0, 0);

  test_reaction.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.run_rate_loop(particle_sub_group, i);
    test_reaction.descendant_product_loop(particle_sub_group, i, 0.1,
                                          descendant_particles);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto weight =
        particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(weight->at(rowx, 0), 0.9);
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

/*
ReactionController :
loop over cells {
    loop over Reactions {
        generate reaction_sub_groups
    }
    loop over Reactions {
        run_rate_loop(...)
    }
    loop over Reactions {
        apply_kernel (currently descendant_products)
    }
    handle cell-wise products (need add_particles_local(DescendantProducts,
ParentGroup))
}

*/
