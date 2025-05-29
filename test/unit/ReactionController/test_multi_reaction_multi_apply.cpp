#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ReactionController, multi_reaction_multi_apply) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto particle_group_2 = create_test_particle_group(N_total);
  auto loop2 = particle_loop(
      "set_internal_state2", particle_group_2,
      [=](auto internal_state) { internal_state[0] = 2; },
      Access::write(Sym<INT>("INTERNAL_STATE")));

  loop2->execute();

  particle_group->add_particles_local(particle_group_2);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);
  auto reaction_controller = ReactionController(test_wrapper);

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 1;

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction1 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0,
      std::array<int, num_products_per_parent>{1});

  test_rate = 10.0; // example rate

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 2,
      std::array<int, num_products_per_parent>{3});

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction1));

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction2));

  auto reduction = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
      particle_group, [=](auto W, auto GA) { GA.fetch_add(0, 0, W[0]); },
      Access::read(Sym<REAL>("WEIGHT")), Access::add(reduction))
      ->execute();

  reaction_controller.apply_reactions(particle_group, 0.1);
  auto reduction_after = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
      particle_group, [=](auto W, auto GA) { GA.fetch_add(0, 0, W[0]); },
      Access::read(Sym<REAL>("WEIGHT")), Access::add(reduction_after))
      ->execute();

  auto merged_group_marking =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("INTERNAL_STATE"), 1);
  auto merged_group = merged_group_marking->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  auto merged_group_marking2 =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("INTERNAL_STATE"), 3);
  auto merged_group2 = merged_group_marking2->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 2);
    EXPECT_EQ(merged_group2->get_npart_cell(icell), 2);

    EXPECT_DOUBLE_EQ(reduction_after->get_cell(icell)->at(0, 0),
                     reduction->get_cell(icell)->at(0, 0)); //, 1e-12);
  }

  particle_group->domain->mesh->free();
  particle_group_2->domain->mesh->free();
}