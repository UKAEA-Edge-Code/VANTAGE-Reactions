#include "common_markers.hpp"
#include "common_transformations.hpp"
#include "containers/cell_dat_const.hpp"
#include "merge_transformation.hpp"
#include "mock_reactions.hpp"
#include "particle_sub_group.hpp"
#include "transformation_wrapper.hpp"
#include "typedefs.hpp"
#include <gtest/gtest.h>
#include <memory>
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
          Sym<REAL>("P"), Sym<REAL>("COMPUTATIONAL_WEIGHT"), Sym<REAL>("V"));

  auto reaction_controller =
      ReactionController(child_transform, Sym<INT>("INTERNAL_STATE"));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 1;

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 0,
      std::array<int, num_products_per_parent>{1});

  test_reaction.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction));

  reaction_controller.apply_reactions(particle_group, 0.01);

  auto initial_weight_per_cell =
      reaction_controller.get_tot_weight_per_cell(1)->get_cell(0)->at(0, 0);

  auto merged_group_marking =
      make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(
          Sym<INT>("INTERNAL_STATE"), 1);
  auto merged_group = merged_group_marking->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 2);

    std::vector<INT> cells = {icell, icell};
    std::vector<INT> layers = {0, 1};

    auto merged_particles = merged_group->get_particles(cells, layers);
    for (int i = 0; i < 2; i++) {
      EXPECT_NEAR(merged_particles->at(Sym<REAL>("COMPUTATIONAL_WEIGHT"), i, 0),
                  initial_weight_per_cell / 2, 1e-12);
    }
  }

  reaction_controller.apply_reactions(particle_group, 0.01);

  merged_group = merged_group_marking->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  // TODO: Find a way to check the weights of the descendant
  // particles against the weight of the parent group after
  // one time-step and after two time-steps.
  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 4);
  }

  particle_group->domain->mesh->free(); // Explicit free? Yuck
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
          Sym<REAL>("P"), Sym<REAL>("COMPUTATIONAL_WEIGHT"), Sym<REAL>("V"));

  auto reaction_controller =
      ReactionController(child_transform, Sym<INT>("INTERNAL_STATE"));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 1;

  auto test_reaction1 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 0,
      std::array<int, num_products_per_parent>{1});

  test_rate = 10.0; // example rate

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate, 2,
      std::array<int, num_products_per_parent>{3});

  test_reaction1.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  test_reaction2.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction1));

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction2));

  reaction_controller.apply_reactions(particle_group, 0.1);

  auto initial_weight_per_cell =
      reaction_controller.get_tot_weight_per_cell(1)->get_cell(0)->at(0, 0);
  auto initial_weight_per_cell2 =
      reaction_controller.get_tot_weight_per_cell(3)->get_cell(0)->at(0, 0);

  auto merged_group_marking =
      make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(
          Sym<INT>("INTERNAL_STATE"), 1);
  auto merged_group = merged_group_marking->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  auto merged_group_marking2 =
      make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(
          Sym<INT>("INTERNAL_STATE"), 3);
  auto merged_group2 = merged_group_marking2->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 2);
    EXPECT_EQ(merged_group2->get_npart_cell(icell), 2);

    std::vector<INT> cells = {icell, icell};
    std::vector<INT> layers = {0, 1};

    auto merged_particles = merged_group->get_particles(cells, layers);
    auto merged_particles2 = merged_group2->get_particles(cells, layers);
    for (int i = 0; i < 2; i++) {
      EXPECT_NEAR(merged_particles->at(Sym<REAL>("COMPUTATIONAL_WEIGHT"), i, 0),
                  initial_weight_per_cell / 2, 1e-12);
      EXPECT_NEAR(
          merged_particles2->at(Sym<REAL>("COMPUTATIONAL_WEIGHT"), i, 0),
          initial_weight_per_cell2 / 2, 1e-12);
    }
  }

  particle_group->domain->mesh->free();
  particle_group_2->domain->mesh->free();
}

TEST(ReactionController, ionisation_reaction) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  auto reaction_controller = ReactionController(Sym<INT>("INTERNAL_STATE"));

  auto ionise_reaction = IoniseReaction(particle_group->sycl_target,
                                        Sym<REAL>("TOT_REACTION_RATE"), 0);

  ionise_reaction.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  reaction_controller.add_reaction(
      std::make_shared<IoniseReaction>(ionise_reaction));

  reaction_controller.apply_reactions(particle_group, 1.5);

  auto test_removal_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<EqualsComp<REAL>, REAL>>(
              Sym<REAL>("COMPUTATIONAL_WEIGHT"), 0.0)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto W = particle_group->get_cell(Sym<REAL>("COMPUTATIONAL_WEIGHT"), cellx);
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
