#pragma once
#include <neso_particles.hpp>
#include <reaction_data.hpp>

using namespace NESO::Particles;

/**
 * struct AMJUELFitCrossSection - General H.1 AMJUEL cross section fit, with
 * left and right asymptotic treatment. Assumes monotonically decreasing
 * cross-sections, and takes as the maximum value the evaluated cross-section at
 * some minimum lab frame impact energy.
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
   * @brief General H.1 AMJUEL cross section fit, with optional left and right
   * asymptotic fits.
   *
   * @param vel_norm Velocity normalisation in m/s
   * @param cs_norm Cross-section normalisation in m^2
   * @param mass_amu Mass of the ion in the H.1 reaction
   * @param coeffs Bulk fit coefficients
   * @param l_coeffs Left asymptote coefficients (size 0 if no low-energy
   * treatment)
   * @param r_coeffs Right asymptote coefficients (size 0 if no high-energy
   * treatment)
   * @param lab_E_min Energy value below which the left asymptote fit is used
   * (if there are any coefficients)
   * @param lab_E_max Energy value above which the right asymptote fit is used
   * (if there are any coefficients)
   * @param min_E Lowest energy for which the cross-section is evaluated. This
   * is where the maximum value of the cross-section is assumed to be.
   */
  AMJUELFitCrossSection(REAL vel_norm, REAL cs_norm, REAL mass_amu,
                        std::array<REAL, num_coeffs> coeffs,
                        std::array<REAL, num_l_coeffs> l_coeffs,
                        std::array<REAL, num_r_coeffs> r_coeffs, REAL lab_E_min,
                        REAL lab_E_max, REAL min_E)
      : cs_norm(cs_norm), mult_const(vel_norm * vel_norm * mass_amu *
                                     1.66053904e-27 / (2 * 1.60217663e-19)),
        coeffs(coeffs), l_coeffs(l_coeffs), r_coeffs(r_coeffs),
        lab_E_min(lab_E_min), lab_E_max(lab_E_max) {

    // Make sure that the uninitialised value is never returned in the upcoming
    // call
    this->min_E = -1.0;
    this->max_val = this->get_value_at(std::sqrt(min_E / this->mult_const));
    this->min_E = min_E;
  };

  /**
   * @brief Get value of H.1 AMJUEL cross section at given relative velocity of
   * projectile and target
   *
   * @param relative_vel Relative projectile and target velocity (in normalised
   * units)
   * @return Value of the cross section (in normalised units) at the givent
   * velocity value, obeying the fit asymptotic rules
   */
  REAL get_value_at(const REAL &relative_vel) const {

    REAL E = this->mult_const * relative_vel * relative_vel;

    REAL logE = std::log(E);
    if (E <= this->min_E) {
      return this->max_val;
    };

    bool left_asymptote = E <= this->lab_E_min && num_l_coeffs > 0;
    bool right_asymptote = E >= this->lab_E_max && num_r_coeffs > 0;

    REAL sum_E = 0;
    if (left_asymptote) {

      for (int i; i < num_l_coeffs; i++) {

        sum_E += this->l_coeffs[i] * std::pow(logE, i);
      }
    } else if

        (right_asymptote) {

      for (int i; i < num_r_coeffs; i++) {

        sum_E += this->r_coeffs[i] * std::pow(logE, i);
      }
    }

    else {
      for (int i; i < num_coeffs; i++) {

        sum_E += this->coeffs[i] * std::pow(logE, i);
      }
    }
    return std::exp(sum_E) * 1e-4 / this->cs_norm;
  };

  REAL get_max_rate_val() const { return this->max_val; };

private:
  REAL max_val;
  REAL mult_const;
  REAL cs_norm;
  REAL min_E;
  REAL lab_E_min;
  REAL lab_E_max;
  std::array<REAL, num_coeffs> coeffs;
  std::array<REAL, num_l_coeffs> l_coeffs;
  std::array<REAL, num_r_coeffs> r_coeffs;
};
