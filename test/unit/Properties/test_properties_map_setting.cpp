#include <reactions.hpp>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

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