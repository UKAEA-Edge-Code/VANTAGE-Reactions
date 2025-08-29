#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ReactionData, FixedCoefficientData) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_reaction =
      LinearReactionBase<0, FixedCoefficientData, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{},
          FixedCoefficientData(2.0), TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);
  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                                          descendant_particles);
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                                          descendant_particles);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);
    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.64);
    }
  }

  particle_group->domain->mesh->free();
}

TEST(ReactionData, AMJUEL2DData) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto amjuel_data = AMJUEL2DData<2, 2>(
      3e12, 1.0, 1.0, 1.0,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}});

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DData<2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, amjuel_data,
          TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  // Rate calculated based on ne=3e18, T=2eV, with the evolved quantity
  // normalised to 3e12
  auto expected_rate = 3.880728735562758;
  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate); // 1e-15);
    }
  }

  particle_group->domain->mesh->free();
}

TEST(ReactionData, AMJUEL2DDataH3) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  REAL mass_amu = 1.0;
  REAL vel_norm = std::sqrt(
      2 * 1.60217663e-19 /
      (mass_amu * 1.66053904e-27)); // Makes the normalisation constant for the
                                    // energy equal to 1

  // Normalisation chosen to set the multiplicative constant in front of the
  // exp(sum...) to 1.0, assuming n = 3e18
  auto amjuel_data = AMJUEL2DDataH3<2, 2, 2>(
      3e12, 1.0, 1.0, 1.0, vel_norm, mass_amu,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}});

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DDataH3<2, 2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, amjuel_data,
          TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  REAL logT = std::log(2);
  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    auto vel = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      REAL logE = std::log(std::pow(vel->at(rowx, 0) - 1.0, 2) +
                           std::pow(vel->at(rowx, 1) - 3.0,
                                    2)); // Assuming vx = 1.0 and vy = 3.0
      REAL expected_rate = 1.0 + 0.02 * logE + 0.01 * logT + 0.02 * logE * logT;
      expected_rate = std::exp(expected_rate);
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate);
    }
  }

  particle_group->domain->mesh->free();
}

TEST(ReactionData, AMJUEL2DData_coronal) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Manipulating the normalisation quantities to trigger the coronal limit
  // calculation
  auto amjuel_data = AMJUEL2DData<2, 2>(
      3e6, 1e-6, 1.0, 1.0,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}});

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DData<2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, amjuel_data,
          TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  // Rate calculated based on ne=3e18, T=2eV, with the evolved quantity
  // normalised to 3e6, and density normalisation set to trigger the coronal
  // limit
  auto expected_rate = 2.737188973785161;
  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate); //, 1e-15);
    }
  }

  particle_group->domain->mesh->free();
}

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

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate);
    }
  }

  particle_group->domain->mesh->free();
}
