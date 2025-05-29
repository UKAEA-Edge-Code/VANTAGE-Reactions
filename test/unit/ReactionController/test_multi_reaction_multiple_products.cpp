#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ReactionController, multi_reaction_multiple_products) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);

  auto reaction_controller = ReactionController(test_wrapper);
  reaction_controller.set_cell_block_size(2);

  REAL test_rate = 5.0;

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction1 = TestReaction<0>(particle_group->sycl_target, test_rate,
                                        0, std::array<int, 0>{});

  const INT num_products_per_parent = 2;

  test_rate = 10.0;

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0,
      std::array<int, num_products_per_parent>{1, 2});

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<0>>(test_reaction1));

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

  auto merged_species_1_marker =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("INTERNAL_STATE"), 1);
  auto merged_species_1 = merged_species_1_marker->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  auto merged_species_2_marker =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("INTERNAL_STATE"), 2);
  auto merged_species_2 = merged_species_2_marker->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_species_1->get_npart_cell(icell), 2);
    EXPECT_EQ(merged_species_2->get_npart_cell(icell), 2);

    // The 2/3 factor on reduction is due to test_reaction1 not producing
    // any descendant products but still reducing the weight of the parent
    // particles. This causes the final weight to be 2/3 that of the original
    // instead of equivalent.
    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                reduction->get_cell(icell)->at(0, 0) * (2.0 / 3.0), 1e-12);
  }

  particle_group->domain->mesh->free();
}