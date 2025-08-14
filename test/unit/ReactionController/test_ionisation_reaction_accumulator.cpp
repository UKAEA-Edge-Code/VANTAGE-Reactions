#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ReactionController, ionisation_reaction_accumulator) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto particle_spec = particle_group->get_particle_spec();

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