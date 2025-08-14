#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

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
      electron_species);

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