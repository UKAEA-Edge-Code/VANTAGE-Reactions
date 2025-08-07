#include "../include/mock_particle_properties.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(Properties, modify_property_map) {
  auto test_map_obj = PropertiesMap(PropertiesTest::custom_prop_map);
  test_map_obj.at(PropertiesTest::custom_props.test_custom_prop2) =
      "TEST_PROP3";
  auto test_map = test_map_obj.get_map();

  EXPECT_STREQ(
      test_map.at(PropertiesTest::custom_props.test_custom_prop2).c_str(),
      "TEST_PROP3");
}