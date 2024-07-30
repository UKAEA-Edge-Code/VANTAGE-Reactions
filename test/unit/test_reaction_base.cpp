#pragma once
#include "common_markers.hpp"
#include "data_calculator.hpp"
#include "mock_reactions.hpp"
#include "reaction_base.hpp"
#include "transformation_wrapper.hpp"
#include <common_transformations.hpp>
#include <derived_reactions/electron_impact_ionisation.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <reaction_data/AMJUEL_2D_data.hpp>
#include <reaction_data/fixed_coefficient_data.hpp>
#include <reaction_data/fixed_rate_data.hpp>
#include <reaction_kernels/base_cx_kernels.hpp>
#include <stdexcept>
#include <transformation_wrapper.hpp>

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

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 0,
      std::array<int, 0>{}, particle_spec);

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

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction = TestReactionVarRate(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"),
      Sym<REAL>("POSITION"), 0, particle_spec);

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

TEST(ReactionData, FixedCoefficientData) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction =
      LinearReactionBase<0, FixedCoefficientData, TestReactionKernels<0>>(
          particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 0,
          std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
          std::vector<ParticleProp<INT>>{}, FixedCoefficientData(2.0),
          TestReactionKernels<0>(), particle_spec);

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);
  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i);
    test_reaction.descendant_product_loop(particle_sub_group, i, 0.1,
                                          descendant_particles);
    test_reaction.run_rate_loop(particle_sub_group, i);
    test_reaction.descendant_product_loop(particle_sub_group, i, 0.1,
                                          descendant_particles);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);
    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(weight->at(rowx, 0), 0.64);
    }
  }

  particle_group->domain->mesh->free();
}
TEST(ReactionData, AMJUEL2DData) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();

  auto amjuel_data = AMJUEL2DData<2, 2>(
      3e12, 1.0, 1.0, 1.0,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}});

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DData<2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 0,
          std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
          std::vector<ParticleProp<INT>>{}, amjuel_data,
          TestReactionKernels<0>(), particle_spec);

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  // Rate calculated based on ne=3e18, T=2eV, with the evolved quantity
  // normalised to 3e12
  auto expected_rate = 3.880728735562758;
  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_NEAR(rate->at(rowx, 0), expected_rate, 1e-15);
    }
  }

  particle_group->domain->mesh->free();
}
TEST(ReactionData, AMJUEL2DData_coronal) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();

  // Manipulating the normalisation quantities to trigger the coronal limit
  // calculation
  auto amjuel_data = AMJUEL2DData<2, 2>(
      3e6, 1e-6, 1.0, 1.0,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}});

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DData<2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 0,
          std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
          std::vector<ParticleProp<INT>>{}, amjuel_data,
          TestReactionKernels<0>(), particle_spec);

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  // Rate calculated based on ne=3e18, T=2eV, with the evolved quantity
  // normalised to 3e6, and density normalisation set to trigger the coronal
  // limit
  auto expected_rate = 2.737188973785161;
  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_NEAR(rate->at(rowx, 0), expected_rate, 1e-15);
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

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction1 = TestReaction<0>(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 1, 2,
                                        std::array<int, 0>{}, particle_spec);

  auto test_reaction2 = TestReaction<1>(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 2, 3,
                                        std::array<int, 1>{4}, particle_spec);

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
      reactions[reaction]->run_rate_loop(subgroups[reaction], i);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->descendant_product_loop(subgroups[reaction], i, 0.1,
                                                   descendant_particles);
    }

    auto internal_state =
        particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

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

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction1 = TestReaction<0>(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 1, 0,
                                        std::array<int, 0>{}, particle_spec);

  auto test_reaction2 = TestReaction<0>(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 1, 0,
                                        std::array<int, 0>{}, particle_spec);

  const INT num_products_per_parent = 1;

  auto test_reaction3 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 2, 0,
      std::array<int, 1>{1}, particle_spec);

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
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

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

  auto particle_spec = particle_group->get_particle_spec();

  auto test_data = FixedRateData(1.0);
  auto electron_species = Species("ELECTRON");
  auto target_species = Species("ION", 1.0);
  target_species.set_id(0);
  auto test_reaction = ElectronImpactIonisation<FixedRateData, FixedRateData>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_data,
      test_data, target_species, electron_species, particle_spec);

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

    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(weight->at(rowx, 0), 0.9);
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ChargeExchange, simple_beam_exchange) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();
  auto projectile_species = Species("ION", 1.2, 0.0, 0);
  auto target_species = Species("ION2", 2.0, 0.0, 1);

  auto test_reaction =
      LinearReactionBase<1, FixedRateData, CXReactionKernels<>,
                         DataCalculator<FixedRateData, FixedRateData>>(
          particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 0,
          std::array<int, 1>{1},
          std::vector<ParticleProp<REAL>>{
              ParticleProp<REAL>(Sym<REAL>("VELOCITY"), 2),
              ParticleProp<REAL>(Sym<REAL>("WEIGHT"), 1)},
          std::vector<ParticleProp<INT>>{
              ParticleProp<INT>(Sym<INT>{"INTERNAL_STATE"}, 1)},
          FixedRateData(1.0),
          CXReactionKernels<>(target_species, projectile_species),
          particle_spec,
          DataCalculator<FixedRateData, FixedRateData>(
              particle_spec, FixedRateData(-1.0), FixedRateData(1.0)));

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i);
    test_reaction.descendant_product_loop(particle_sub_group, i, 0.1,
                                          descendant_particles);

    auto weight = descendant_particles->get_cell(Sym<REAL>("WEIGHT"), i);
    auto vel_parent = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    auto vel_child = descendant_particles->get_cell(Sym<REAL>("VELOCITY"), i);
    auto id_child =
        descendant_particles->get_cell(Sym<INT>("INTERNAL_STATE"), i);

    auto target_source =
        particle_group->get_cell(Sym<REAL>("ION2_SOURCE_DENSITY"), i);
    auto projectile_source =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_DENSITY"), i);

    auto target_source_momentum =
        particle_group->get_cell(Sym<REAL>("ION2_SOURCE_MOMENTUM"), i);
    auto projectile_source_momentum =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_MOMENTUM"), i);

    auto target_source_energy =
        particle_group->get_cell(Sym<REAL>("ION2_SOURCE_ENERGY"), i);
    auto projectile_source_energy =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_ENERGY"), i);
    const int nrow = weight->nrow;
    const int parent_nrow = vel_parent->nrow;

    EXPECT_EQ(nrow, parent_nrow);

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(weight->at(rowx, 0), 0.1);
      EXPECT_EQ(vel_child->at(rowx, 0), -1.0);
      EXPECT_EQ(vel_child->at(rowx, 1), 1.0);
      EXPECT_EQ(id_child->at(rowx, 0), 1);
      EXPECT_EQ(target_source->at(rowx, 0), -0.1);
      EXPECT_EQ(projectile_source->at(rowx, 0), 0.1);
      EXPECT_EQ(target_source_momentum->at(rowx, 0), 0.1 * 2);
      EXPECT_EQ(target_source_momentum->at(rowx, 1), -0.1 * 2);
      EXPECT_EQ(projectile_source_momentum->at(rowx, 0),
                0.1 * 1.2 * vel_parent->at(rowx, 0));
      EXPECT_EQ(projectile_source_momentum->at(rowx, 1),
                0.1 * 1.2 * vel_parent->at(rowx, 1));
      EXPECT_EQ(target_source_energy->at(rowx, 0),
                -0.1 * 2); // -w*m*v_i^2 / 2
      EXPECT_EQ(projectile_source_energy->at(rowx, 0),
                0.1 * 0.6 *
                    (std::pow(vel_parent->at(rowx, 0), 2) +
                     std::pow(vel_parent->at(rowx, 1), 2))); // w*m*v^2 / 2
    }
  }

  particle_group->domain->mesh->free();
}

TEST(DataCalculator, custom_sources) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<TestReactionData, TestReactionData>>(

          particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 0,
          std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
          std::vector<ParticleProp<INT>>{}, TestReactionData(2.0),
          TestReactionDataCalcKernels<0>(), particle_spec,
          DataCalculator<TestReactionData, TestReactionData>(
              particle_spec, TestReactionData(3.0), TestReactionData(4.0)));

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

    auto source_density =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), i);
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(source_density->at(rowx, 0), 3.0);
      EXPECT_EQ(source_energy->at(rowx, 0), 4.0);
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(Species, getters) {
  auto test_species = Species();

  EXPECT_THROW(test_species.get_name(), std::logic_error);

  EXPECT_THROW(test_species.get_id(), std::logic_error);

  EXPECT_THROW(test_species.get_mass(), std::logic_error);

  EXPECT_THROW(test_species.get_charge(), std::logic_error);

  std::string test_species_name = "H";
  test_species.set_name(test_species_name);
  EXPECT_STREQ(test_species.get_name().c_str(), test_species_name.c_str());

  REAL test_species_id = 10.0;
  test_species.set_id(test_species_id);
  EXPECT_EQ(test_species.get_id(), test_species_id);

  REAL test_species_mass = 5.5;
  test_species.set_mass(test_species_mass);
  EXPECT_EQ(test_species.get_mass(), test_species_mass);

  REAL test_species_charge = 2.3;
  test_species.set_charge(test_species_charge);
  EXPECT_EQ(test_species.get_charge(), test_species_charge);
}
