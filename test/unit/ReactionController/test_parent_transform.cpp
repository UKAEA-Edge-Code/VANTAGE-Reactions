#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ReactionController, parent_transform) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto parent_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto parent_transform_wrapper =
      std::make_shared<TransformationWrapper>(parent_transform);

  auto child_transform =
      make_transformation_strategy<NoOpTransformationStrategy>();

  auto child_transform_wrapper =
      std::make_shared<TransformationWrapper>(child_transform);

  auto reaction_controller =
      ReactionController(parent_transform_wrapper, child_transform_wrapper);

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction = TestReaction<0>(particle_group->sycl_target, 1, 0,
                                       std::array<int, 0>{});

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<0>>(test_reaction));

  auto reduction = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
      particle_group, [=](auto W, auto GA) { GA.fetch_add(0, 0, W[0]); },
      Access::read(Sym<REAL>("WEIGHT")), Access::add(reduction))
      ->execute();

  reaction_controller.apply_reactions(particle_group, 5e-15);

  auto reduction_after = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
      particle_group, [=](auto W, auto GA) { GA.fetch_add(0, 0, W[0]); },
      Access::read(Sym<REAL>("WEIGHT")), Access::add(reduction_after))
      ->execute();

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(particle_group->get_npart_cell(icell), 2);

    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                     reduction->get_cell(icell)->at(0, 0), 1e-12);
  }

  particle_group->domain->mesh->free();
}