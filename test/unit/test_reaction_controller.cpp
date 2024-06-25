#pragma once
#include "common_markers.hpp"
#include "common_transformations.hpp"
#include "merge_transformation.hpp"
#include "mock_reactions.hpp"
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include "transformation_wrapper.hpp"
#include <array>
#include <gtest/gtest.h>
#include <ionisation_reactions/amjuel_ionisation.hpp>
#include <ionisation_reactions/fixed_rate_ionisation.hpp>
#include <memory>
#include <particle_spec_builder.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ReactionController, single_reaction_multi_apply) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>(
          Sym<REAL>("POSITION"), Sym<REAL>("WEIGHT"), Sym<REAL>("VELOCITY"));

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);
  auto reaction_controller =
      ReactionController(test_wrapper, Sym<INT>("INTERNAL_STATE"));
  REAL test_rate = 5.0;

  const INT num_products_per_parent = 1;

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 0,
      std::array<int, num_products_per_parent>{1}, particle_spec);

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

    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                reduction->get_cell(icell)->at(0, 0), 1e-12);
  }

  reaction_controller.apply_reactions(particle_group, 0.01);

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 4);

    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                reduction->get_cell(icell)->at(0, 0), 1e-12);
  }

  particle_group->domain->mesh->free();
}

TEST(ReactionController, multi_reaction_multiple_products) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>(
          Sym<REAL>("POSITION"), Sym<REAL>("WEIGHT"), Sym<REAL>("VELOCITY"));

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);

  auto reaction_controller =
      ReactionController(test_wrapper, Sym<INT>("INTERNAL_STATE"));

  REAL test_rate = 5.0;

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction1 = TestReaction<0>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 0,
      std::array<int, 0>{}, particle_spec);

  const INT num_products_per_parent = 2;

  test_rate = 10.0;

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 0,
      std::array<int, num_products_per_parent>{1, 2}, particle_spec);

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
      make_transformation_strategy<MergeTransformationStrategy<2>>(
          Sym<REAL>("POSITION"), Sym<REAL>("WEIGHT"), Sym<REAL>("VELOCITY"));

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);
  auto reaction_controller =
      ReactionController(test_wrapper, Sym<INT>("INTERNAL_STATE"));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 1;

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction1 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 0,
      std::array<int, num_products_per_parent>{1}, particle_spec);

  test_rate = 10.0; // example rate

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 2,
      std::array<int, num_products_per_parent>{3}, particle_spec);

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

    EXPECT_NEAR(reduction_after->get_cell(icell)->at(0, 0),
                reduction->get_cell(icell)->at(0, 0), 1e-12);
  }

  particle_group->domain->mesh->free();
  particle_group_2->domain->mesh->free();
}

TEST(ReactionController, parent_transform) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto parent_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>(
          Sym<REAL>("POSITION"), Sym<REAL>("WEIGHT"), Sym<REAL>("VELOCITY"));

  auto parent_transform_wrapper =
      std::make_shared<TransformationWrapper>(parent_transform);

  auto child_transform =
      make_transformation_strategy<NoOpTransformationStrategy>();

  auto child_transform_wrapper =
      std::make_shared<TransformationWrapper>(child_transform);

  auto reaction_controller =
      ReactionController(parent_transform_wrapper, child_transform_wrapper,
                         Sym<INT>("INTERNAL_STATE"));

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction = TestReaction<0>(particle_group->sycl_target,
                                       Sym<REAL>("TOT_REACTION_RATE"), 1, 0,
                                       std::array<int, 0>{}, particle_spec);

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<0>>(test_reaction));

  auto reduction = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
      particle_group, [=](auto W, auto GA) { GA.fetch_add(0, 0, W[0]); },
      Access::read(Sym<REAL>("WEIGHT")), Access::add(reduction))
      ->execute();

  reaction_controller.apply_reactions(particle_group, 0.0);

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

TEST(ReactionController, ionisation_reaction) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto reaction_controller = ReactionController(Sym<INT>("INTERNAL_STATE"));

  auto particle_spec_builder = ParticleSpecBuilder();

  particle_spec_builder.add_particle_spec(particle_group->get_particle_spec());

  auto particle_spec = particle_spec_builder.get_particle_spec();

  auto ionise_reaction = FixedRateIonisation(particle_group->sycl_target,
                                             Sym<REAL>("TOT_REACTION_RATE"),
                                             1.0, 0, particle_spec);

  reaction_controller.add_reaction(
      std::make_shared<FixedRateIonisation>(ionise_reaction));

  reaction_controller.apply_reactions(particle_group, 1.5);

  auto test_removal_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
              Sym<REAL>("WEIGHT"), 1.0e-12)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), cellx);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(W->at(rowx, 0), 0.0);
    };
  };

  test_removal_wrapper.transform(particle_group);

  auto final_particle_num = particle_group->get_npart_local();

  EXPECT_EQ(final_particle_num, 0);

  particle_group->domain->mesh->free();
}

TEST(ReactionController, ionisation_reaction_accumulator) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto particle_spec = particle_group->get_particle_spec();

  auto ionise_reaction = FixedRateIonisation(particle_group->sycl_target,
                                             Sym<REAL>("TOT_REACTION_RATE"),
                                             1.0, 0, particle_spec);

  auto accumulator_transform = std::make_shared<CellwiseAccumulator<REAL>>(
      particle_group, std::vector<std::string>{"ELECTRON_SOURCE_DENSITY"});

  auto accumulator_transform_wrapper = std::make_shared<TransformationWrapper>(
      std::dynamic_pointer_cast<TransformationStrategy>(accumulator_transform));

  auto merge_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>(
          Sym<REAL>("POSITION"), Sym<REAL>("WEIGHT"), Sym<REAL>("VELOCITY"));

  auto merge_transform_wrapper =
      std::make_shared<TransformationWrapper>(merge_transform);

  auto reaction_controller = ReactionController(
      std::vector{accumulator_transform_wrapper, merge_transform_wrapper},
      std::vector<std::shared_ptr<TransformationWrapper>>{},
      Sym<INT>("INTERNAL_STATE"));
  reaction_controller.add_reaction(
      std::make_shared<FixedRateIonisation>(ionise_reaction));

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  std::vector<int> num_parts;
  for (int cellx = 0; cellx < num_cells; cellx++) {

    num_parts.push_back(particle_group->get_npart_cell(cellx));
  };

  reaction_controller.apply_reactions(particle_group, 0.5);

  auto accumulated_1d =
      accumulator_transform->get_cell_data("ELECTRON_SOURCE_DENSITY");
  for (int cellx = 0; cellx < num_cells; cellx++) {

    EXPECT_EQ(particle_group->get_npart_cell(cellx), 2);
    EXPECT_NEAR(accumulated_1d[cellx]->at(0, 0), num_parts[cellx], 1e-10);
  };

  particle_group->domain->mesh->free();
}

TEST(ReactionController, ionisation_reaction_amjuel) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto reaction_controller = ReactionController(Sym<INT>("INTERNAL_STATE"));

  // AMJUEL 2.1.5 reaction coeffecients
  // b0 -0.317385000000e+02 b1 0.114381800000e+02 b2 -0.383399800000e+01
  // b3 0.704669200000e+00 b4 -0.743148620000e-01 b5 0.415374900000e-02
  // b6 -0.948696700000e-04 b7 0.000000000000e-00 b8 0.000000000000e+00

  std::array<REAL, 9> b_coeffs = {
      -0.317385000000e+02, 0.114381800000e+02,  -0.383399800000e+01,
      0.704669200000e+00,  -0.743148620000e-01, 0.415374900000e-02,
      -0.948696700000e-04, 0.000000000000e-00,  0.000000000000e+00};

  // m^-3 to cm^-3
  REAL density_normalisation = 1e-6;

  auto particle_spec_builder = ParticleSpecBuilder();

  auto int_1d_props = Properties<INT>(std::vector<int>{
      default_properties.id, default_properties.internal_state});

  auto int_1d_positional_props =
      Properties<INT>(std::vector<int>{default_properties.cell_id});

  auto real_1d_props =
      Properties<REAL>(std::vector<int>{default_properties.tot_reaction_rate,
                                        default_properties.weight,
                                        default_properties.fluid_density,
                                        default_properties.fluid_temperature},
                       std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>{default_properties.temperature,
                                        default_properties.density,
                                        default_properties.source_energy,
                                        default_properties.source_density});

  auto real_2d_props =
      Properties<REAL>(std::vector<int>{default_properties.velocity},
                       std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>{default_properties.source_momentum});

  auto real_2d_positional_props =
      Properties<REAL>(std::vector<int>{default_properties.position});

  particle_spec_builder.add_particle_prop<INT>(int_1d_props);
  particle_spec_builder.add_particle_prop<INT>(int_1d_positional_props, 1,
                                               true);
  particle_spec_builder.add_particle_prop<REAL>(real_1d_props);
  particle_spec_builder.add_particle_prop<REAL>(real_2d_props, 2);
  particle_spec_builder.add_particle_prop<REAL>(real_2d_positional_props, 2,
                                                true);

  auto particle_spec = particle_spec_builder.get_particle_spec();

  auto ionise_reaction = IoniseReactionAMJUEL<9>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), 0,
      density_normalisation, b_coeffs, particle_spec);

  reaction_controller.add_reaction(
      std::make_shared<IoniseReactionAMJUEL<9>>(ionise_reaction));

  reaction_controller.apply_reactions(particle_group, 0.1);

  auto test_removal_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<REAL, EqualsComp>>(
              Sym<REAL>("WEIGHT"), 0.0)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  auto expected_weight = 0.0;

  // Hard-coded expected rate based on b_coeffs and fluid_density=3e18 and
  // fluid_temperature=2eV
  auto expected_rate = 26.993377387251336;

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), cellx);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), cellx);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(rate->at(rowx, 0), expected_rate);
      EXPECT_EQ(W->at(rowx, 0), expected_weight);
    };
  };

  test_removal_wrapper.transform(particle_group);

  auto final_particle_num = particle_group->get_npart_local();

  EXPECT_EQ(final_particle_num, 0);

  particle_group->domain->mesh->free();
}
