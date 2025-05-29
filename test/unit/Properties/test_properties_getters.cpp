#include "../include/mock_particle_properties.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

// Testing getters for Properties struct
TEST(Properties, properties_getters) {
  auto default_props_obj = PropertiesTest::real_props;

  auto expected_simple_props =
      std::vector<int>{default_properties.position,
                       default_properties.velocity,
                       default_properties.tot_reaction_rate,
                       default_properties.weight,
                       default_properties.fluid_density,
                       default_properties.fluid_temperature,
                       default_properties.fluid_flow_speed};

  auto expected_species = std::vector<Species>{Species("ELECTRON")};

  auto expected_species_props = std::vector<int>{
      default_properties.temperature,    default_properties.density,
      default_properties.flow_speed,     default_properties.source_energy,
      default_properties.source_density, default_properties.source_momentum};

  auto returned_simple_props = default_props_obj.get_simple_props();
  auto returned_species = default_props_obj.get_species();
  auto returned_species_props = default_props_obj.get_species_props();

  EXPECT_EQ(expected_simple_props.size(), returned_simple_props.size());
  for (int i = 0; i < returned_simple_props.size(); i++) {
    EXPECT_EQ(expected_simple_props[i], returned_simple_props[i]);
  }

  EXPECT_EQ(expected_species.size(), returned_species.size());
  for (int i = 0; i < returned_species.size(); i++) {
    EXPECT_STREQ(expected_species[i].get_name().c_str(),
                 returned_species[i].get_name().c_str());
  }

  EXPECT_EQ(expected_species_props.size(), returned_species_props.size());
  for (int i = 0; i < returned_species_props.size(); i++) {
    EXPECT_EQ(expected_species_props[i], returned_species_props[i]);
  }

  auto expected_all_props =
      std::vector<int>{default_properties.position,
                       default_properties.velocity,
                       default_properties.tot_reaction_rate,
                       default_properties.weight,
                       default_properties.fluid_density,
                       default_properties.fluid_temperature,
                       default_properties.fluid_flow_speed,
                       default_properties.temperature,
                       default_properties.density,
                       default_properties.flow_speed,
                       default_properties.source_energy,
                       default_properties.source_density,
                       default_properties.source_momentum};

  auto returned_all_props = default_props_obj.get_props();

  EXPECT_EQ(expected_all_props.size(), returned_all_props.size());
  for (int i = 0; i < expected_all_props.size(); i++) {
    EXPECT_EQ(expected_all_props[i], returned_all_props[i]);
  }
}