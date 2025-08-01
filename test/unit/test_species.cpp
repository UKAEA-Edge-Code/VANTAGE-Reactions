#include <reactions.hpp>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(Species, getters) {
    auto test_species = Species();

    if (std::getenv("TEST_NESOASSERT") != nullptr) {
        EXPECT_THROW(test_species.get_name(), std::logic_error);
    
        EXPECT_THROW(test_species.get_id(), std::logic_error);
    
        EXPECT_THROW(test_species.get_mass(), std::logic_error);
    
        EXPECT_THROW(test_species.get_charge(), std::logic_error);
    }
  
    std::string test_species_name = "H";
    test_species.set_name(test_species_name);
    EXPECT_STREQ(test_species.get_name().c_str(), test_species_name.c_str());
  
    INT test_species_id = 10;
    test_species.set_id(test_species_id);
    EXPECT_EQ(test_species.get_id(), test_species_id);
  
    REAL test_species_mass = 5.5;
    test_species.set_mass(test_species_mass);
    EXPECT_DOUBLE_EQ(test_species.get_mass(), test_species_mass);
  
    REAL test_species_charge = 2.3;
    test_species.set_charge(test_species_charge);
    EXPECT_DOUBLE_EQ(test_species.get_charge(), test_species_charge);
}
