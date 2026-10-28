#include "include/test_common.hpp"
#include "include/test_extern_templates.hpp"

using namespace VANTAGE::Reactions;

TEST(CrossSections, AMJUEL_H1_bulk) {

  REAL mass_amu = 1.0;
  REAL vel_norm = std::sqrt(
      2 * 1.60217663e-19 /
      (mass_amu * 1.66053904e-27)); // Makes the normalisation constant for the
  // energy equal to 1
  REAL E_max = 1.0e6;
  auto cs = AMJUELFitCrossSection<2, 0, 0>(
      vel_norm, 1e-4, mass_amu, std::array<REAL, 2>{-1.0, -0.1},
      std::array<REAL, 0>{}, std::array<REAL, 0>{}, 2.0, 1e4, E_max);

  EXPECT_NEAR(cs.get_max_rate_val(),
              std::exp(-1.0 - 0.1 * std::log(E_max)) * std::sqrt(E_max), 1e-12);
  EXPECT_NEAR(cs.get_value_at(1e4), cs.get_max_rate_val() / 1e4, 1e-12);
  EXPECT_NEAR(cs.get_value_at(10.0), std::exp(-1.0 - 0.1 * std::log(100.0)),
              1e-12);
  EXPECT_FALSE(
      cs.accept_reject(0.1, 0.5, cs.get_value_at(0.1), cs.get_max_rate_val()));
}

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

TEST(CrossSections, AMJUEL_H1_high_energy) {

  REAL mass_amu = 1.0;
  REAL vel_norm = std::sqrt(
      2 * 1.60217663e-19 /
      (mass_amu * 1.66053904e-27)); // Makes the normalisation constant for the
  // energy equal to 1
  REAL E_max = 1.0e6;
  auto cs = AMJUELFitCrossSection<2, 2, 2>(
      vel_norm, 1e-4, mass_amu, std::array<REAL, 2>{-1.0, -0.1},
      std::array<REAL, 2>{-2.0, -0.2}, std::array<REAL, 2>{-3.0, -0.3}, 2.0,
      1e4, E_max);

  EXPECT_NEAR(cs.get_value_at(10.0), std::exp(-1.0 - 0.1 * std::log(100.0)),
              1e-14);
  EXPECT_NEAR(cs.get_value_at(200), std::exp(-3.0 - 0.3 * std::log(40000.0)),
              1e-14);
  EXPECT_NEAR(cs.get_value_at(1e4), cs.get_max_rate_val() / 1e4, 1e-14);
}

TEST(CrossSections, sigma_v_zero_relative_speed) {
  auto constant_rate_cs = ConstantRateCrossSection(2.5);
  EXPECT_DOUBLE_EQ(constant_rate_cs.sigma_v(0.0), 2.5);
  EXPECT_DOUBLE_EQ(constant_rate_cs.sigma_v(4.0), 2.5);

  auto zero_rate_cs = ConstantRateCrossSection(0.0);
  EXPECT_DOUBLE_EQ(zero_rate_cs.sigma_v(0.0), 0.0);

  auto constant_cs = ConstantCrossSection(2.5, 10.0);
  EXPECT_DOUBLE_EQ(constant_cs.sigma_v(0.0), 0.0);
  EXPECT_DOUBLE_EQ(constant_cs.sigma_v(4.0), 10.0);

  auto ipl_cs = IPLCrossSection(2.5, 1.0, 0.5, 10.0);
  EXPECT_DOUBLE_EQ(ipl_cs.sigma_v(0.0), 0.0);
  EXPECT_DOUBLE_EQ(ipl_cs.sigma_v(4.0), 5.0);

  REAL mass_amu = 1.0;
  REAL vel_norm = std::sqrt(2 * 1.60217663e-19 / (mass_amu * 1.66053904e-27));
  auto amjuel_cs = AMJUELFitCrossSection<2, 2, 0>(
      vel_norm, 1e-4, mass_amu, std::array<REAL, 2>{-1.0, -0.1},
      std::array<REAL, 2>{-2.0, -0.2}, std::array<REAL, 0>{}, 2.0, 1e4, 1.0e6);
  EXPECT_DOUBLE_EQ(amjuel_cs.get_value_at(0.0), 0.0);
  EXPECT_DOUBLE_EQ(amjuel_cs.sigma_v(0.0), 0.0);
}