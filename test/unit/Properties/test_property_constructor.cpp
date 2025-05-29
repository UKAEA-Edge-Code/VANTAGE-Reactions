#include "../include/mock_particle_properties.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(Properties, property_constructor) {
  auto test_map = PropertiesTest::custom_prop_map;
  ASSERT_THROW(properties_map(test_map).at(
                   PropertiesTest::custom_props.test_custom_prop2 + 1),
               std::out_of_range);
}