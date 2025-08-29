#include "include/mock_particle_group.hpp"
#include "include/mock_particle_properties.hpp"
#include "include/mock_reactions.hpp"
#include "reactions_lib/particle_properties_map.hpp"
#include "reactions_lib/reaction_kernel_pre_reqs.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(Properties, property_constructor) {
  auto test_map = PropertiesTest::custom_prop_map;
  ASSERT_THROW(PropertiesMap(test_map).at(
                   PropertiesTest::custom_props.test_custom_prop2 + 1),
               std::out_of_range);
}

TEST(Properties, modify_property_map) {
  auto test_map_obj = PropertiesMap(PropertiesTest::custom_prop_map);
  test_map_obj.at(PropertiesTest::custom_props.test_custom_prop2) =
      "TEST_PROP3";
  auto test_map = test_map_obj.get_map();

  EXPECT_STREQ(
      test_map.at(PropertiesTest::custom_props.test_custom_prop2).c_str(),
      "TEST_PROP3");
}

// Testing required_simple_prop_names call for Properties struct
TEST(Properties, simple_prop_names) {
  // Properties object with a species property but no simple property
  auto empty_simple_props_obj =
      Properties<REAL>(std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>(default_properties.density));

  ASSERT_EQ(empty_simple_props_obj.simple_prop_names(),
            std::vector<std::string>());

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

// Testing required_simple_prop_index call for Properties struct
TEST(Properties, simple_prop_index) {
  // Properties object with a species property but no simple property
  auto empty_simple_props_obj =
      Properties<REAL>(std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>(default_properties.density));

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(
        empty_simple_props_obj.simple_prop_index(default_properties.id),
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

// Testing required_species_prop_names call for Properties struct
TEST(Properties, species_prop_names) {
  // Properties object with a simple property but no species property
  auto empty_species_props_obj =
      Properties<INT>(std::vector<int>{default_properties.internal_state});

  ASSERT_EQ(empty_species_props_obj.species_prop_names(),
            std::vector<std::string>());

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

TEST(Properties, properties_map_setting) {
  std::map<int, std::string> test_property_map = {
      {0, "TEST_PROP1"}, {1, "TEST_PROP2"}, {2, "TEST_PROP3"}};

  struct test_reaction_data : ReactionDataBase<> {
    test_reaction_data(std::map<int, std::string> property_map_)
        : ReactionDataBase(property_map_) {}

    std::map<int, std::string> get_property_map() {
      return this->properties_map;
    }
  };

  auto test_reaction_data_base = test_reaction_data(test_property_map);

  auto returned_property_map = test_reaction_data_base.get_property_map();

  auto count = 0;
  for (auto prop : returned_property_map) {
    EXPECT_EQ(count, prop.first);
    count++;
    char test_prop[11];
    std::sprintf(test_prop, "TEST_PROP%d", count);
    EXPECT_STREQ(test_prop, prop.second.c_str());
  }
}

TEST(Properties, full_use_properties_map) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_prop1 = ParticleProp(Sym<REAL>("TEST_PROP1"), 1);
  auto test_prop2 = ParticleProp(Sym<REAL>("TEST_PROP2"), 1);

  auto test_prop1_dat = ParticleDat<REAL>(
      particle_group->sycl_target, Sym<REAL>("TEST_PROP1"), 1, cell_count);
  auto test_prop2_dat = ParticleDat<REAL>(
      particle_group->sycl_target, Sym<REAL>("TEST_PROP2"), 1, cell_count);

  particle_group->add_particle_dat(test_prop1_dat);
  particle_group->add_particle_dat(test_prop2_dat);

  particle_loop(
      particle_group,
      [=](auto test_prop1_write, auto test_prop2_write) {
        test_prop1_write[0] = 3.0e18;
        test_prop2_write[0] = 2.0;
      },
      Access::write(Sym<REAL>("TEST_PROP1")),
      Access::write(Sym<REAL>("TEST_PROP2")))
      ->execute();

  particle_group->remove_particle_dat(Sym<REAL>("FLUID_DENSITY"));
  particle_group->remove_particle_dat(Sym<REAL>("FLUID_TEMPERATURE"));

  auto test_prop_map = get_default_map();
  test_prop_map[default_properties.fluid_density] = "TEST_PROP1";
  test_prop_map[default_properties.fluid_temperature] = "TEST_PROP2";

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto amjuel_data = AMJUEL2DData<2, 2>(
      3e12, 1.0, 1.0, 1.0,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}},
      test_prop_map);

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DData<2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, amjuel_data,
          TestReactionKernels<0>(test_prop_map));

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  // Rate calculated based on ne=3e18, T=2eV, with the evolved quantity
  // normalised to 3e12
  auto expected_rate = 3.880728735562758;
  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate); // 1e-15);
    }
  }

  particle_group->domain->mesh->free();
}
