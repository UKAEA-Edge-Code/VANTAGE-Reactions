#include "../include/mock_particle_properties.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

// Testing required_species_prop_names call for Properties struct
TEST(Properties, species_prop_names) {
  // Properties object with a simple property but no species property
  auto empty_species_props_obj =
      Properties<INT>(std::vector<int>{default_properties.internal_state});

  ASSERT_EQ(empty_species_props_obj.species_prop_names(), std::vector<std::string>());

  auto real_props_obj = PropertiesTest::real_props;

  auto real_props = {
      default_properties.temperature,    default_properties.density,
      default_properties.flow_speed,     default_properties.source_energy,
      default_properties.source_density, default_properties.source_momentum};

  std::vector<std::string> species_real_props = {};
  for (auto prop : real_props) {
    species_real_props.push_back(get_default_map().at(prop));
  }

  auto real_prop_names = real_props_obj.species_prop_names();

  EXPECT_EQ(real_prop_names.size(), species_real_props.size());

  for (int i = 0; i < real_prop_names.size(); i++) {
    std::string electron_species_name = "ELECTRON_" + species_real_props[i];
    EXPECT_STREQ(real_prop_names[i].c_str(), electron_species_name.c_str());
  }

  auto custom_species_props_obj = Properties<REAL>(
      std::vector<Species>{Species("ION")},
      std::vector<int>{default_properties.density,
                       PropertiesTest::custom_props.test_custom_prop1,
                       PropertiesTest::custom_props.test_custom_prop2});

  std::vector<std::string> species_custom_props = {
      "ION_DENSITY", "ION_TEST_PROP1", "ION_TEST_PROP2"};

  auto custom_prop_names = custom_species_props_obj.species_prop_names(
      PropertiesTest::custom_prop_map);

  EXPECT_EQ(species_custom_props.size(), custom_prop_names.size());

  for (int i = 0; i < custom_prop_names.size(); i++) {
    EXPECT_STREQ(custom_prop_names[i].c_str(), species_custom_props[i].c_str());
  }
}