#pragma once
#include "common_markers.hpp"
#include "common_transformations.hpp"
#include "merge_transformation.hpp"
#include "mock_reactions.hpp"
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include "transformation_wrapper.hpp"
#include <array>
#include <gtest/gtest.h>
#include <ionisation_reactions/amjuel_ionisation.hpp>
#include <ionisation_reactions/fixed_rate_ionisation.hpp>
#include <memory>
#include <particle_spec_builder.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <stdexcept>

using namespace NESO::Particles;
using namespace Reactions;

namespace PropertiesTest {
auto int_props = Properties<INT>(
    std::vector<int>{default_properties.id, default_properties.internal_state,
                     default_properties.cell_id});

auto real_props = Properties<REAL>(
    std::vector<int>{
        default_properties.position, default_properties.velocity,
        default_properties.tot_reaction_rate, default_properties.weight,
        default_properties.fluid_density, default_properties.fluid_temperature},
    std::vector<Species>{Species("ELECTRON")},
    std::vector<int>{default_properties.temperature, default_properties.density,
                     default_properties.source_energy,
                     default_properties.source_density,
                     default_properties.source_momentum});

struct custom_properties_enum : standard_properties_enum {
public:
  enum {
    test_custom_prop1 = default_properties.fluid_temperature + 1,
    test_custom_prop2
  };
};

auto custom_props = custom_properties_enum();

struct custom_prop_map_struct : properties_map {
  custom_prop_map_struct() {
    properties_map::extend_map(custom_props.test_custom_prop1, "TEST_PROP1");
    properties_map::extend_map(custom_props.test_custom_prop2, "TEST_PROP2");
  }
};

const auto custom_prop_map = custom_prop_map_struct().get_map();
} // namespace PropertiesTest

// Testing required_simple_prop_names call for Properties struct
TEST(Properties, simple_prop_names) {
  // Properties object with a species property but no simple property
  auto empty_simple_props_obj =
      Properties<REAL>(std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>(default_properties.density));

  EXPECT_THROW(empty_simple_props_obj.simple_prop_names(), std::logic_error);

  auto int_props_obj = PropertiesTest::int_props;

  std::vector<std::string> simple_int_props = {};
  for (auto prop : int_props_obj.get_simple_props()) {
    simple_int_props.push_back(default_map.at(prop));
  }
  auto int_prop_names = int_props_obj.simple_prop_names();

  EXPECT_EQ(int_prop_names.size(), simple_int_props.size());

  for (int i = 0; i < int_prop_names.size(); i++) {
    EXPECT_EQ(int_prop_names[i], simple_int_props[i]);
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
    EXPECT_EQ(custom_prop_names[i], simple_custom_props[i]);
  }

  auto real_props_obj = PropertiesTest::real_props;

  auto real_props = {default_properties.position,
                     default_properties.velocity,
                     default_properties.tot_reaction_rate,
                     default_properties.weight,
                     default_properties.fluid_density,
                     default_properties.fluid_temperature};

  std::vector<std::string> simple_real_props = {};
  for (auto prop : real_props) {
    simple_real_props.push_back(default_map.at(prop));
  }

  auto real_prop_names = real_props_obj.simple_prop_names();

  EXPECT_EQ(real_prop_names.size(), simple_real_props.size());

  for (int i = 0; i < real_prop_names.size(); i++) {
    EXPECT_EQ(real_prop_names[i], simple_real_props[i]);
  }
}

// Testing required_simple_prop_index call for Properties struct
TEST(Properties, simple_prop_index) {
  // Properties object with a species property but no simple property
  auto empty_simple_props_obj =
      Properties<REAL>(std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>(default_properties.density));

  EXPECT_THROW(empty_simple_props_obj.simple_prop_index(default_properties.id),
               std::logic_error);

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
                                 default_properties.fluid_temperature};

  for (int i = 0; i < real_props.size(); i++) {
    auto real_prop_index = real_props_obj.simple_prop_index(real_props[i]);
    EXPECT_EQ(real_prop_index, i);
  }
}

// Testing required_species_prop_names call for Properties struct
TEST(Properties, species_prop_names) {
  // Properties object with a simple property but no species property
  auto empty_species_props_obj =
      Properties<INT>(std::vector<int>{default_properties.internal_state});

  EXPECT_THROW(empty_species_props_obj.species_prop_names(), std::logic_error);

  auto real_props_obj = PropertiesTest::real_props;

  auto real_props = {default_properties.temperature, default_properties.density,
                     default_properties.source_energy,
                     default_properties.source_density,
                     default_properties.source_momentum};

  std::vector<std::string> species_real_props = {};
  for (auto prop : real_props) {
    species_real_props.push_back(default_map.at(prop));
  }

  auto real_prop_names = real_props_obj.species_prop_names();

  EXPECT_EQ(real_prop_names.size(), species_real_props.size());

  for (int i = 0; i < real_prop_names.size(); i++) {
    EXPECT_EQ(real_prop_names[i], "ELECTRON_" + species_real_props[i]);
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
    EXPECT_EQ(custom_prop_names[i], species_custom_props[i]);
  }
}

// Testing required_species_prop_index call for Properties struct
TEST(Properties, species_prop_index) {
  // Properties object with a species property but no simple property
  auto empty_species_props_obj =
      Properties<REAL>(std::vector<int>(default_properties.velocity));

  EXPECT_THROW(empty_species_props_obj.species_prop_index(
                   "ELECTRON", default_properties.density),
               std::logic_error);

  auto real_props_obj = PropertiesTest::real_props;

  std::vector<int> real_props = {
      default_properties.temperature, default_properties.density,
      default_properties.source_energy, default_properties.source_density,
      default_properties.source_momentum};

  size_t simple_props_size = real_props_obj.get_simple_props().size();

  for (int i = 0; i < real_props.size(); i++) {
    auto real_prop_index =
        real_props_obj.species_prop_index("ELECTRON", real_props[i]);
    EXPECT_EQ(real_prop_index, simple_props_size + i);
  }
}

// Testing getters for Properties struct
TEST(Properties, properties_getters) {
  auto default_props_obj = PropertiesTest::real_props;

  auto expected_simple_props =
      std::vector<int>{default_properties.position,
                       default_properties.velocity,
                       default_properties.tot_reaction_rate,
                       default_properties.weight,
                       default_properties.fluid_density,
                       default_properties.fluid_temperature};

  auto expected_species = std::vector<Species>{Species("ELECTRON")};

  auto expected_species_props = std::vector<int>{
      default_properties.temperature, default_properties.density,
      default_properties.source_energy, default_properties.source_density,
      default_properties.source_momentum};

  auto returned_simple_props = default_props_obj.get_simple_props();
  auto returned_species = default_props_obj.get_species();
  auto returned_species_props = default_props_obj.get_species_props();

  EXPECT_EQ(expected_simple_props.size(), returned_simple_props.size());
  for (int i = 0; i < returned_simple_props.size(); i++) {
    EXPECT_EQ(expected_simple_props[i], returned_simple_props[i]);
  }

  EXPECT_EQ(expected_species.size(), returned_species.size());
  for (int i = 0; i < returned_species.size(); i++) {
    EXPECT_EQ(expected_species[i].get_name(), returned_species[i].get_name());
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
                       default_properties.temperature,
                       default_properties.density,
                       default_properties.source_energy,
                       default_properties.source_density,
                       default_properties.source_momentum};

  auto returned_all_props = default_props_obj.get_props();

  EXPECT_EQ(expected_all_props.size(), returned_all_props.size());
  for (int i = 0; i < expected_all_props.size(); i++) {
    EXPECT_EQ(expected_all_props[i], returned_all_props[i]);
  }
}