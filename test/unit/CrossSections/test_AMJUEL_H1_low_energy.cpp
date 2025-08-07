#include <reactions.hpp>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(CrossSections, AMJUEL_H1_low_energy) {  
    REAL mass_amu = 1.0;
    REAL vel_norm = std::sqrt(
        2 * 1.60217663e-19 /
        (mass_amu * 1.66053904e-27)); // Makes the normalisation constant for the
    // energy equal to 1
    REAL E_max = 1.0e6;
    auto cs = AMJUELFitCrossSection<2, 2, 0>(
        vel_norm, 1e-4, mass_amu, std::array<REAL, 2>{-1.0, -0.1},
        std::array<REAL, 2>{-2.0, -0.2}, std::array<REAL, 0>{}, 2.0, 1e4, E_max);

    EXPECT_NEAR(cs.get_value_at(10.0), std::exp(-1.0 - 0.1 * std::log(100.0)),
                1e-14);
    EXPECT_NEAR(cs.get_value_at(0.1), std::exp(-2.0 - 0.2 * std::log(0.01)),
                1e-14);
}