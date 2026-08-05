#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_reaction_controller_functors.hpp"
#include "reactions_lib/reaction_controller.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/particle_sub_group/particle_sub_group.hpp>
#include <utility>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ReactionController, ionisation_reaction) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto reaction_controller = ReactionController();

  auto particle_spec_builder = ParticleSpecBuilder(2);

  particle_spec_builder.add_particle_spec(particle_group->get_particle_spec());

  auto test_data = FixedRateData(1.0);
  auto electron_species = Species("ELECTRON");
  auto target_species = Species("ION", 1.0, 1.0, 0);
  auto ionise_reaction = ElectronImpactIonisation<FixedRateData, FixedRateData>(
      particle_group->sycl_target, test_data, test_data, target_species,
      electron_species);

  reaction_controller.add_reaction(
      std::make_shared<ElectronImpactIonisation<FixedRateData, FixedRateData>>(
          ionise_reaction));

  reaction_controller.apply(particle_group, 1.5);

  auto accessor = Access::read(Sym<REAL>("WEIGHT"));

  auto test_removal_wrapper = std::make_shared<TransformationWrapper>(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_direct_marking_strategy("small", SmallWeightMarker{}, accessor)},
      make_lambda_transformation_strategy("remove", RemoveSubgroupTransform{}));

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int icell = 0; icell < num_cells; icell++) {
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), icell);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(W->at(rowx, 0), 0.0);
    };
  };

  test_removal_wrapper->transform(particle_group);

  auto final_particle_num = particle_group->get_npart_local();

  EXPECT_EQ(final_particle_num, 0);

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ReactionController, ionisation_reaction_accumulator) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto test_data = FixedRateData(1.0);
  auto electron_species = Species("ELECTRON");
  auto target_species = Species("ION", 1.0, 1.0, 0);
  auto ionise_reaction = ElectronImpactIonisation<FixedRateData, FixedRateData>(
      particle_group->sycl_target, test_data, test_data, target_species,
      electron_species);

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

  reaction_controller.apply(particle_group, 0.5);

  auto accumulated_1d =
      accumulator_transform->get_cell_data("ELECTRON_SOURCE_DENSITY");
  for (int icell = 0; icell < num_cells; icell++) {

    EXPECT_EQ(particle_group->get_npart_cell(icell), 2);
    EXPECT_DOUBLE_EQ(accumulated_1d[icell]->at(0, 0),
                     num_parts[icell] * 0.5); //, 1e-10);
  };

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
