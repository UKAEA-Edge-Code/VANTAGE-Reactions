#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include <gtest/gtest.h>

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
                                        0, std::array<int, 0>{}, particle_spec);

  const INT num_products_per_parent = 2;

  test_rate = 10.0;

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0,
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
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);
  auto reaction_controller = ReactionController(test_wrapper);

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 1;

  auto particle_spec = particle_group->get_particle_spec();

  auto test_reaction1 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0,
      std::array<int, num_products_per_parent>{1}, particle_spec);

  test_rate = 10.0; // example rate

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 2,
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

    EXPECT_DOUBLE_EQ(reduction_after->get_cell(icell)->at(0, 0),
                     reduction->get_cell(icell)->at(0, 0)); //, 1e-12);
  }

  particle_group->domain->mesh->free();
  particle_group_2->domain->mesh->free();
}

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

    EXPECT_DOUBLE_EQ(reduction_after->get_cell(icell)->at(0, 0),
                     reduction->get_cell(icell)->at(0, 0)); //, 1e-12);
  }

  particle_group->domain->mesh->free();
}

TEST(ReactionController, ionisation_reaction) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto reaction_controller = ReactionController();

  auto particle_spec_builder = ParticleSpecBuilder(2);

  particle_spec_builder.add_particle_spec(particle_group->get_particle_spec());

  auto particle_spec = particle_spec_builder.get_particle_spec();

  auto test_data = FixedRateData(1.0);
  auto electron_species = Species("ELECTRON");
  auto target_species = Species("ION", 1.0, 1.0, 0);
  auto ionise_reaction = ElectronImpactIonisation<FixedRateData, FixedRateData>(
      particle_group->sycl_target, test_data, test_data, target_species,
      electron_species, particle_spec);

  reaction_controller.add_reaction(
      std::make_shared<ElectronImpactIonisation<FixedRateData, FixedRateData>>(
          ionise_reaction));

  reaction_controller.apply_reactions(particle_group, 1.5);

  auto test_removal_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
              Sym<REAL>("WEIGHT"), 1.0e-12)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int icell = 0; icell < num_cells; icell++) {
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), icell);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(W->at(rowx, 0), 0.0);
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

  auto test_data = FixedRateData(1.0);
  auto electron_species = Species("ELECTRON");
  auto target_species = Species("ION", 1.0, 1.0, 0);
  auto ionise_reaction = ElectronImpactIonisation<FixedRateData, FixedRateData>(
      particle_group->sycl_target, test_data, test_data, target_species,
      electron_species, particle_spec);

  auto accumulator_transform = std::make_shared<CellwiseAccumulator<REAL>>(
      particle_group, std::vector<std::string>{"ELECTRON_SOURCE_DENSITY"});

  auto accumulator_transform_wrapper = std::make_shared<TransformationWrapper>(
      std::dynamic_pointer_cast<TransformationStrategy>(accumulator_transform));

  auto merge_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto merge_transform_wrapper =
      std::make_shared<TransformationWrapper>(merge_transform);

  auto reaction_controller = ReactionController(
      std::vector{accumulator_transform_wrapper, merge_transform_wrapper},
      std::vector<std::shared_ptr<TransformationWrapper>>{});

  reaction_controller.add_reaction(
      std::make_shared<ElectronImpactIonisation<FixedRateData, FixedRateData>>(
          ionise_reaction));

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  std::vector<int> num_parts;
  for (int icell = 0; icell < num_cells; icell++) {

    num_parts.push_back(particle_group->get_npart_cell(icell));
  };

  reaction_controller.apply_reactions(particle_group, 0.5);

  auto accumulated_1d =
      accumulator_transform->get_cell_data("ELECTRON_SOURCE_DENSITY");
  for (int icell = 0; icell < num_cells; icell++) {

    EXPECT_EQ(particle_group->get_npart_cell(icell), 2);
    EXPECT_DOUBLE_EQ(accumulated_1d[icell]->at(0, 0),
                     num_parts[icell] * 0.5); //, 1e-10);
  };

  particle_group->domain->mesh->free();
}

TEST(ReactionController, ionisation_reaction_amjuel) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto reaction_controller = ReactionController();

  // AMJUEL H.2 2.1.5FJ reaction coeffecients
  // b0 -0.317385000000e+02 b1 0.114381800000e+02 b2 -0.383399800000e+01
  // b3 0.704669200000e+00 b4 -0.743148620000e-01 b5 0.415374900000e-02
  // b6 -0.948696700000e-04 b7 0.000000000000e-00 b8 0.000000000000e+00

  std::array<REAL, 9> b_coeffs = {
      -0.317385000000e+02, 0.114381800000e+02,  -0.383399800000e+01,
      0.704669200000e+00,  -0.743148620000e-01, 0.415374900000e-02,
      -0.948696700000e-04, 0.000000000000e-00,  0.000000000000e+00};

  auto particle_spec_builder = ParticleSpecBuilder(2);

  auto int_1d_props = Properties<INT>(std::vector<int>{
      default_properties.id, default_properties.internal_state});

  auto int_1d_positional_props =
      Properties<INT>(std::vector<int>{default_properties.cell_id});

  auto real_1d_props = Properties<REAL>(
      std::vector<int>{default_properties.tot_reaction_rate,
                       default_properties.weight,
                       default_properties.fluid_density,
                       default_properties.fluid_temperature},
      std::vector<Species>{Species("ELECTRON"), Species("ION")},
      std::vector<int>{
          default_properties.temperature, default_properties.density,
          default_properties.source_energy, default_properties.source_density});

  auto real_2d_props = Properties<REAL>(
      std::vector<int>{default_properties.velocity},
      std::vector<Species>{Species("ELECTRON"), Species("ION")},
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

  auto fixed_rate = FixedRateData(1.0);
  auto electron_species = Species("ELECTRON");
  auto target_species = Species("ION", 1.0, 1.0, 0);
  auto test_data = AMJUEL1DData<9>(1.0, 1.0, 1.0, 1.0, b_coeffs);
  auto ionise_reaction =
      ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData>(
          particle_group->sycl_target, test_data, fixed_rate, target_species,
          electron_species, particle_spec);

  reaction_controller.add_reaction(
      std::make_shared<
          ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData>>(
          ionise_reaction));

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

  for (int icell = 0; icell < num_cells; icell++) {
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), icell);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), icell);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate);
      EXPECT_DOUBLE_EQ(W->at(rowx, 0), expected_weight);
    };
  };

  test_removal_wrapper.transform(particle_group);

  auto final_particle_num = particle_group->get_npart_local();

  EXPECT_EQ(final_particle_num, 0);

  particle_group->domain->mesh->free();
}
