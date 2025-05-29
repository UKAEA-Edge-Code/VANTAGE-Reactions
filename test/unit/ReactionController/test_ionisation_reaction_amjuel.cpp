#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace NESO::Particles;
using namespace Reactions;

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
          electron_species);

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