#include "../include/mock_reactions.hpp"
#include "../include/mock_particle_group.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ReactionData, EphemeralPropertiesReactionData) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_data = TestEphemeralVarData();

  auto expected_prop_names = std::vector<std::string>{
      get_default_map().at(default_properties.weight),
      get_default_map().at(default_properties.boundary_intersection_point),
      get_default_map().at(default_properties.boundary_intersection_normal)};

  auto test_prop_names = test_data.get_required_real_props();

  ASSERT_EQ(expected_prop_names.size(), test_prop_names.size());
  for (int i = 0; i < test_prop_names.size(); i++) {
    EXPECT_EQ(expected_prop_names[i], test_prop_names[i]);
  }

  auto expected_prop_names_ephemeral = std::vector<std::string>{
      get_default_map().at(default_properties.boundary_intersection_point),
      get_default_map().at(default_properties.boundary_intersection_normal)};

  auto test_prop_names_ephemeral =
      test_data.get_required_real_props_ephemeral();

  ASSERT_EQ(expected_prop_names_ephemeral.size(),
            test_prop_names_ephemeral.size());
  for (int i = 0; i < test_prop_names_ephemeral.size(); i++) {
    EXPECT_EQ(expected_prop_names_ephemeral[i], test_prop_names_ephemeral[i]);
  }

  auto test_reaction =
      LinearReactionBase<0, TestEphemeralVarData, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, test_data,
          TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto expected_rate = 0.5;

  // Add data to subgroup
  particle_sub_group->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_normal, 2);
  particle_sub_group->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_point, 2);
  particle_sub_group->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_metadata,
      BoundaryInteractionSpecification::intersection_metadata_ncomp);

  ASSERT_TRUE(contains_boundary_interaction_data(particle_sub_group));
  ASSERT_TRUE(contains_boundary_interaction_data(particle_sub_group, 2));
  particle_loop(
      "set_ephemeral_dat_loop_test", particle_sub_group,
      [=](auto point, auto normal) {
        point.at_ephemeral(0) = 2.0;
        normal.at_ephemeral(0) = 0.25;
      },
      Access::write(BoundaryInteractionSpecification::intersection_point),
      Access::write(BoundaryInteractionSpecification::intersection_normal))
      ->execute();

  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate);
    }
  }

  particle_group->domain->mesh->free();
}