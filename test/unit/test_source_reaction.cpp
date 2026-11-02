#include "include/mock_particle_group.hpp"
#include "include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(SourceReaction, PreservesMarkerWeightAndCreatesProduct) {
  auto particle_group = create_test_particle_group(1000);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);
  auto marker_species = Species("MARKER", 1.0, 0.0, 0);
  auto product_species = Species("PRODUCT", 2.0, 0.0, 1);
  auto velocity_data = FixedArrayData<2>({3.0, 4.0});
  auto reaction = SourceReaction<FixedArrayData<2>>(
      particle_group->sycl_target, velocity_data, marker_species,
      product_species);
  auto descendants = std::make_shared<NP::ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  const int cell_count = particle_group->domain->mesh->get_cell_count();
  for (int cell = 0; cell < cell_count; cell++) {
    reaction.calculate_rates(particle_sub_group, cell, cell + 1);
    reaction.apply(particle_sub_group, cell, cell + 1, 0.1, descendants);

    auto parent_weight =
        particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), cell);
    auto product_weight = descendants->get_cell(NP::Sym<REAL>("WEIGHT"), cell);
    auto product_velocity =
        descendants->get_cell(NP::Sym<REAL>("VELOCITY"), cell);

    for (int row = 0; row < product_weight->nrow; row++) {
      EXPECT_DOUBLE_EQ(parent_weight->at(row, 0), 1.0);
      EXPECT_DOUBLE_EQ(product_weight->at(row, 0), 0.1);
      EXPECT_DOUBLE_EQ(product_velocity->at(row, 0), 3.0);
      EXPECT_DOUBLE_EQ(product_velocity->at(row, 1), 4.0);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(SourceReaction, OptionalSourceFeedback) {
  auto particle_group = create_test_particle_group(1000);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);
  auto marker_species = Species("MARKER", 1.0, 0.0, 0);
  auto product_species = Species("PRODUCT", 2.0, 0.0, 1);
  auto properties_map = PropertiesMap();
  properties_map[default_properties.source_momentum] = "ION_SOURCE_MOMENTUM";
  properties_map[default_properties.source_energy] = "ION_SOURCE_ENERGY";
  auto velocity_data = FixedArrayData<2>({1.0, 1.0});
  auto reaction = SourceReaction<FixedArrayData<2>, 2, true>(
      particle_group->sycl_target, velocity_data, marker_species,
      product_species, properties_map.get_map());
  auto descendants = std::make_shared<NP::ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  reaction.calculate_rates(particle_sub_group, 0, 1);
  reaction.apply(particle_sub_group, 0, 1, 0.1, descendants);

  auto parent_weight = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), 0);
  auto parent_velocity = particle_group->get_cell(NP::Sym<REAL>("VELOCITY"), 0);
  auto source_momentum =
      particle_group->get_cell(NP::Sym<REAL>("ION_SOURCE_MOMENTUM"), 0);
  auto source_energy =
      particle_group->get_cell(NP::Sym<REAL>("ION_SOURCE_ENERGY"), 0);

  for (int row = 0; row < parent_weight->nrow; row++) {
    EXPECT_DOUBLE_EQ(parent_weight->at(row, 0), 1.0);
    EXPECT_DOUBLE_EQ(source_momentum->at(row, 0),
                     0.1 * 2.0 * (parent_velocity->at(row, 0) - 1.0));
    EXPECT_DOUBLE_EQ(source_momentum->at(row, 1),
                     0.1 * 2.0 * (parent_velocity->at(row, 1) - 1.0));
    EXPECT_NEAR(source_energy->at(row, 0),
                0.5 * 0.1 * 2.0 *
                    (parent_velocity->at(row, 0) * parent_velocity->at(row, 0) +
                     parent_velocity->at(row, 1) * parent_velocity->at(row, 1) -
                     2.0),
                1.0e-14);
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
