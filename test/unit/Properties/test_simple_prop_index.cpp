#include "../include/mock_particle_properties.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

// Testing required_simple_prop_index call for Properties struct
TEST(Properties, simple_prop_index) {
  // Properties object with a species property but no simple property
  auto empty_simple_props_obj =
      Properties<REAL>(std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>(default_properties.density));

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(empty_simple_props_obj.simple_prop_index(default_properties.id),
                  std::logic_error);
  }

  auto int_props_obj = PropertiesTest::int_props;

  std::vector<int> int_props = {default_properties.id,
                                default_properties.internal_state,
                                default_properties.cell_id};

  auto int_prop_names = int_props_obj.simple_prop_names();

  for (int i = 0; i < int_props.size(); i++) {
    auto int_prop_index = int_props_obj.simple_prop_index(int_props[i]);
    EXPECT_EQ(int_prop_index, i);
  }

  auto real_props_obj = PropertiesTest::real_props;

  std::vector<int> real_props = {default_properties.position,
                                 default_properties.velocity,
                                 default_properties.tot_reaction_rate,
                                 default_properties.weight,
                                 default_properties.fluid_density,
                                 default_properties.fluid_temperature,
                                 default_properties.fluid_flow_speed};

  for (int i = 0; i < real_props.size(); i++) {
    auto real_prop_index = real_props_obj.simple_prop_index(real_props[i]);
    EXPECT_EQ(real_prop_index, i);
  }
}