#ifndef REACTIONS_AMJUEL_FIT_CS_H
#define REACTIONS_AMJUEL_FIT_CS_H
#include "../reaction_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief A struct that defines a general H.1 AMJUEL cross section fit, with
 * left and right asymptotic treatment. Assumes monotonically decreasing
 * cross-sections, and takes as the maximum value the evaluated rate at
 * some maximum lab frame impact energy.
 *
 * @tparam num_coeffs Number of coefficients in the bulk of the validity range
 * (usually 9).
 * @tparam num_l_coeffs Number of fit coefficients in the left (low-energy)
 * asymptotic fit (usually 3).
 * @tparam num_r_coeffs Number of fit coefficients in the right (high-energy)
 * asymptotic fit (usually 3).
 */
template <size_t num_coeffs, size_t num_l_coeffs, size_t num_r_coeffs>
struct AMJUELFitCrossSection : public AbstractCrossSection {

  /**
   * @brief Constructor for AMJUELFitCrossSection.
   *
   * @param vel_norm Velocity normalisation in m/s
   * @param cs_norm Cross-section normalisation in m^2
   * @param mass_amu Reduced mass of the collision partners in the H.1 reaction
   * in amus
   * @param coeffs Bulk fit coefficients
   * @param l_coeffs Left asymptote coefficients (size 0 if no low-energy
   * treatment)
   * @param r_coeffs Right asymptote coefficients (size 0 if no high-energy
   * treatment)
   * @param lab_E_min Energy value below which the left asymptote fit is used
   * (if there are any coefficients)
   * @param lab_E_max Energy value above which the right asymptote fit is used
   * (if there are any coefficients)
   * @param max_E Highest energy for which the cross-section is evaluated. This
   * is where the maximum value of the rate is assumed to be. After this value,
   * the cross section is of the form max_val/v_r.
   */
  AMJUELFitCrossSection(REAL vel_norm, REAL cs_norm, REAL mass_amu,
                        std::array<REAL, num_coeffs> coeffs,
                        std::array<REAL, num_l_coeffs> l_coeffs,
                        std::array<REAL, num_r_coeffs> r_coeffs, REAL lab_E_min,
                        REAL lab_E_max, REAL max_E)
      : cs_norm(cs_norm), mult_const(vel_norm * vel_norm * mass_amu *
                                     1.66053904e-27 / (2 * 1.60217663e-19)),
        coeffs(coeffs), l_coeffs(l_coeffs), r_coeffs(r_coeffs),
        lab_E_min(lab_E_min), lab_E_max(lab_E_max) {

    // Make sure that the uninitialised value is never returned in the upcoming
    // call
    this->max_E = 1e128;
    this->max_val = this->get_value_at(std::sqrt(max_E / this->mult_const)) *
                    std::sqrt(max_E / this->mult_const);
    this->max_E = max_E;
  };

  /**
   * @brief Get value of H.1 AMJUEL cross section at given relative velocity of
   * projectile and target
   *
   * @param relative_vel Relative projectile and target velocity (in normalised
   * units)
   * @return Value of the cross section (in normalised units) at the given
   * velocity value, obeying the fit asymptotic rules
   */
  REAL get_value_at(const REAL &relative_vel) const {

    REAL E = this->mult_const * relative_vel * relative_vel;

    REAL logE = Kernel::log(E);
    if (E >= this->max_E) {
      return this->max_val / relative_vel;
    };

    bool left_asymptote = E <= this->lab_E_min && num_l_coeffs > 0;
    bool right_asymptote = E >= this->lab_E_max && num_r_coeffs > 0;

    REAL sum_E = 0;
    if (left_asymptote) {

      for (int i = 0; i < num_l_coeffs; i++) {

        sum_E += this->l_coeffs[i] * std::pow(logE, i);
      }
    } else if

        (right_asymptote) {

      for (int i = 0; i < num_r_coeffs; i++) {

        sum_E += this->r_coeffs[i] * std::pow(logE, i);
      }
    }

    else {
      for (int i = 0; i < num_coeffs; i++) {

        sum_E += this->coeffs[i] * std::pow(logE, i);
      }
    }
    return Kernel::exp(sum_E) * 1e-4 / this->cs_norm;
  };

  /**
   * @brief Returns maximum value of the rate sigma*v of for this cross-section.
   *
   * @return REAL-valued maximum value.
   */
  REAL get_max_rate_val() const { return this->max_val; };

private:
  REAL max_val;
  REAL mult_const;
  REAL cs_norm;
  REAL max_E;
  REAL lab_E_min;
  REAL lab_E_max;
  std::array<REAL, num_coeffs> coeffs;
  std::array<REAL, num_l_coeffs> l_coeffs;
  std::array<REAL, num_r_coeffs> r_coeffs;
};
}; // namespace VANTAGE::Reactions
#endif