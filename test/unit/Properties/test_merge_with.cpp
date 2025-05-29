#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

// Testing merge_with for Properties struct
TEST(Properties, merge_with) {

  auto simple_props1 = std::vector<int>{
      default_properties.position, default_properties.velocity,
      default_properties.tot_reaction_rate, default_properties.weight};

  auto species1 = std::vector<Species>{Species("ELECTRON")};

  auto species_props1 = std::vector<int>{default_properties.temperature,
                                         default_properties.density};

  auto properties1 = Properties<REAL>(simple_props1, species1, species_props1);

  auto simple_props2 = std::vector<int>{default_properties.position,
                                        default_properties.fluid_density};

  auto species2 = std::vector<Species>{Species("ION"), Species("ELECTRON")};

  auto species_props2 = std::vector<int>{default_properties.density,
                                         default_properties.source_density};

  auto properties2 = Properties<REAL>(simple_props2, species2, species_props2);

  auto merge_props = properties1.merge_with(properties2);

  auto expected_simple_props = std::vector<int>{
      default_properties.position, default_properties.velocity,
      default_properties.tot_reaction_rate, default_properties.weight,
      default_properties.fluid_density};

  auto expected_species =
      std::vector<Species>{Species("ELECTRON"), Species("ION")};

  auto expected_species_props = std::vector<int>{
      default_properties.temperature, default_properties.density,
      default_properties.source_density};

  auto returned_simple_props = merge_props.get_simple_props();
  auto returned_species = merge_props.get_species();
  auto returned_species_props = merge_props.get_species_props();

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

  auto expected_all_props = std::vector<int>{
      default_properties.position,          default_properties.velocity,
      default_properties.tot_reaction_rate, default_properties.weight,
      default_properties.fluid_density,     default_properties.temperature,
      default_properties.density,           default_properties.source_density};

  auto returned_all_props = merge_props.get_props();

  EXPECT_EQ(expected_all_props.size(), returned_all_props.size());
  for (int i = 0; i < expected_all_props.size(); i++) {
    EXPECT_EQ(expected_all_props[i], returned_all_props[i]);
  }
}