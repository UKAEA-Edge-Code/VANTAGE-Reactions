#include "../include/mock_particle_properties.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

// Testing required_simple_prop_names call for Properties struct
TEST(Properties, simple_prop_names) {
  // Properties object with a species property but no simple property
  auto empty_simple_props_obj =
      Properties<REAL>(std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>(default_properties.density));

  ASSERT_EQ(empty_simple_props_obj.simple_prop_names(), std::vector<std::string>());

  auto int_props_obj = PropertiesTest::int_props;

  std::vector<std::string> simple_int_props = {};
  for (auto prop : int_props_obj.get_simple_props()) {
    simple_int_props.push_back(get_default_map().at(prop));
  }
  auto int_prop_names = int_props_obj.simple_prop_names();

  EXPECT_EQ(int_prop_names.size(), simple_int_props.size());

  for (int i = 0; i < int_prop_names.size(); i++) {
    EXPECT_STREQ(int_prop_names[i].c_str(), simple_int_props[i].c_str());
  }

  auto custom_simple_props_obj = Properties<REAL>(std::vector<int>{
      default_properties.weight, PropertiesTest::custom_props.test_custom_prop1,
      PropertiesTest::custom_props.test_custom_prop2});

  std::vector<std::string> simple_custom_props = {"WEIGHT", "TEST_PROP1",
                                                  "TEST_PROP2"};

  auto custom_prop_names = custom_simple_props_obj.simple_prop_names(
      PropertiesTest::custom_prop_map);

  EXPECT_EQ(custom_prop_names.size(), simple_custom_props.size());

  for (int i = 0; i < custom_prop_names.size(); i++) {
    EXPECT_STREQ(custom_prop_names[i].c_str(), simple_custom_props[i].c_str());
  }

  auto real_props_obj = PropertiesTest::real_props;

  auto real_props = {default_properties.position,
                     default_properties.velocity,
                     default_properties.tot_reaction_rate,
                     default_properties.weight,
                     default_properties.fluid_density,
                     default_properties.fluid_temperature,
                     default_properties.fluid_flow_speed};

  std::vector<std::string> simple_real_props = {};
  for (auto prop : real_props) {
    simple_real_props.push_back(get_default_map().at(prop));
  }

  auto real_prop_names = real_props_obj.simple_prop_names();

  EXPECT_EQ(real_prop_names.size(), simple_real_props.size());

  for (int i = 0; i < real_prop_names.size(); i++) {
    EXPECT_STREQ(real_prop_names[i].c_str(), simple_real_props[i].c_str());
  }

  // Since default_properties.weight will not be a key in the fully
  // custom_full_prop_map
  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(custom_simple_props_obj.simple_prop_names(
                    PropertiesTest::custom_prop_no_weight_map),
                std::logic_error);
  }

  auto custom_full_simple_props_obj = Properties<REAL>(
      std::vector<int>{PropertiesTest::custom_props.test_custom_prop1,
                       PropertiesTest::custom_props.test_custom_prop2});

  auto custom_full_prop_names = custom_full_simple_props_obj.simple_prop_names(
      PropertiesTest::custom_prop_map);
  EXPECT_EQ(custom_full_prop_names.size(), 2);
}