#include "../include/mock_particle_properties.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

// Testing required_species_prop_index call for Properties struct
TEST(Properties, species_prop_index) {
  // Properties object with a species property but no simple property
  auto empty_species_props_obj =
      Properties<REAL>(std::vector<int>(default_properties.velocity));

if (std::getenv("TEST_NESOASSERT") != nullptr) {
  EXPECT_THROW(empty_species_props_obj.species_prop_index(
                   "ELECTRON", default_properties.density),
               std::logic_error);
}

  auto real_props_obj = PropertiesTest::real_props;

  std::vector<int> real_props = {
      default_properties.temperature,    default_properties.density,
      default_properties.flow_speed,     default_properties.source_energy,
      default_properties.source_density, default_properties.source_momentum};

  size_t simple_props_size = real_props_obj.get_simple_props().size();

  for (int i = 0; i < real_props.size(); i++) {
    auto real_prop_index =
        real_props_obj.species_prop_index("ELECTRON", real_props[i]);
    EXPECT_EQ(real_prop_index, simple_props_size + i);
  }
}