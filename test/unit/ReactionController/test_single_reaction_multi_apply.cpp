#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ReactionController, single_reaction_multi_apply) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);
  auto reaction_controller = ReactionController(test_wrapper);
  REAL test_rate = 5.0;

  const INT num_products_per_parent = 1;

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0,
      std::array<int, num_products_per_parent>{1});

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction));

  auto merged_group_marking =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("INTERNAL_STATE"), 1);
  auto merged_group = merged_group_marking->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  auto reduction = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
      particle_group, [=](auto W, auto GA) { GA.fetch_add(0, 0, W[0]); },
      Access::read(Sym<REAL>("WEIGHT")), Access::add(reduction))
      ->execute();

  reaction_controller.apply_reactions(particle_group, 0.01);

  auto reduction_after = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
      particle_group, [=](auto W, auto GA) { GA.fetch_add(0, 0, W[0]); },
      Access::read(Sym<REAL>("WEIGHT")), Access::add(reduction_after))
      ->execute();

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 2);

    // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                reduction->get_cell(icell)->at(0, 0), 1e-12);
  }

  reaction_controller.apply_reactions(particle_group, 0.01);

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 4);

    // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                reduction->get_cell(icell)->at(0, 0), 1e-12);
  }

  // check that the TOT_REACTION_RATE buffer has been flushed between
  // applications
  auto parent_marking =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("INTERNAL_STATE"), 0);
  auto subgroup = std::make_shared<ParticleSubGroup>(particle_group);
  auto parent_subgroup = parent_marking->make_marker_subgroup(subgroup);

  auto test_la = std::make_shared<LocalArray<REAL>>(
      particle_group->sycl_target, parent_subgroup->get_npart_local(), 0);
  auto loop = particle_loop(
      "check_rate", parent_subgroup,
      [=](auto tot_reaction_rate, auto la, auto index) {
        auto idx = index.get_loop_linear_index();
        la.at(idx) = tot_reaction_rate[0];
      },
      Access::read(Sym<REAL>("TOT_REACTION_RATE")), Access::write(test_la),
      Access::read(ParticleLoopIndex()));
  loop->execute();
  auto test_vec = test_la->get();
  for (auto rate : test_vec) {

    EXPECT_DOUBLE_EQ(rate, 5.0); //, 1e-12);
  }

  particle_group->domain->mesh->free();
}