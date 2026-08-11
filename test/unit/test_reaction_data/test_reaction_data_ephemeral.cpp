#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace VANTAGE::Reactions;

TEST(ReactionData, EphemeralPropertiesReactionData) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto test_data = TestEphemeralVarData();

  auto expected_prop_names = std::vector<std::string>{
      get_default_map().at(default_properties.weight),
      get_default_map().at(default_properties.boundary_intersection_point),
      get_default_map().at(default_properties.boundary_intersection_normal)};

  auto test_prop_names = test_data.get_required_real_props().to_string_vector();

  ASSERT_EQ(expected_prop_names.size(), test_prop_names.size());
  for (int i = 0; i < test_prop_names.size(); i++) {
    ASSERT_NE(std::find(test_prop_names.begin(), test_prop_names.end(),
                        expected_prop_names[i]),
              test_prop_names.end());
  }

  auto test_reaction =
      LinearReactionBase<0, TestEphemeralVarData, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, test_data,
          TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto expected_rate = 0.5;

  // Add data to subgroup
  particle_sub_group->add_ephemeral_dat(
      NP::BoundaryInteractionSpecification::intersection_normal, 2);
  particle_sub_group->add_ephemeral_dat(
      NP::BoundaryInteractionSpecification::intersection_point, 2);
  particle_sub_group->add_ephemeral_dat(
      NP::BoundaryInteractionSpecification::intersection_metadata,
      NP::BoundaryInteractionSpecification::intersection_metadata_ncomp);

  ASSERT_TRUE(contains_boundary_interaction_data(particle_sub_group));
  ASSERT_TRUE(contains_boundary_interaction_data(particle_sub_group, 2));
  particle_loop(
      "set_ephemeral_dat_loop_test", particle_sub_group,
      [=](auto point, auto normal) {
        point.at_ephemeral(0) = 2.0;
        normal.at_ephemeral(0) = 0.25;
      },
      NP::Access::write(
          NP::BoundaryInteractionSpecification::intersection_point),
      NP::Access::write(
          NP::BoundaryInteractionSpecification::intersection_normal))
      ->execute();

  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    auto rate =
        particle_group->get_cell(NP::Sym<NP::REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
