#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_reaction_controller_functors.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <utility>

using namespace VANTAGE::Reactions;

TEST(ReactionController, single_reaction_multi_apply) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);
  auto reaction_controller = ReactionController(test_wrapper);
  NP::REAL test_rate = 5.0;

  const NP::INT num_products_per_parent = 1;

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0,
      std::array<int, num_products_per_parent>{1});

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction));

  auto merged_group =
      particle_sub_group(particle_group, InternalStateEquals(1),
                         NP::Access::read(NP::Sym<NP::INT>("INTERNAL_STATE")));

  auto reduction = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(particle_group, WeightReducer{},
                NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
                NP::Access::add(reduction))
      ->execute();

  reaction_controller.apply(particle_group, 0.01);

  auto reduction_after = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(particle_group, WeightReducer{},
                NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
                NP::Access::add(reduction_after))
      ->execute();

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 2);

    // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                reduction->get_cell(icell)->at(0, 0), 1e-12);
  }

  reaction_controller.apply(particle_group, 0.01);

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 4);

    // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                reduction->get_cell(icell)->at(0, 0), 1e-12);
  }

  // check that the TOT_REACTION_RATE buffer has been flushed between
  // applications
  auto parent_subgroup =
      particle_sub_group(particle_group, InternalStateEquals(0),
                         NP::Access::read(NP::Sym<NP::INT>("INTERNAL_STATE")));

  auto test_la = std::make_shared<NP::LocalArray<NP::REAL>>(
      particle_group->sycl_target, parent_subgroup->get_npart_local(), 0);
  auto loop = particle_loop(
      "check_rate", parent_subgroup,
      [=](auto tot_reaction_rate, auto la, auto index) {
        auto idx = index.get_loop_linear_index();
        la.at(idx) = tot_reaction_rate[0];
      },
      NP::Access::read(NP::Sym<NP::REAL>("TOT_REACTION_RATE")),
      NP::Access::write(test_la), NP::Access::read(NP::ParticleLoopIndex()));
  loop->execute();
  auto test_vec = test_la->get();
  for (auto rate : test_vec) {

    EXPECT_DOUBLE_EQ(rate, 5.0); //, 1e-12);
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
