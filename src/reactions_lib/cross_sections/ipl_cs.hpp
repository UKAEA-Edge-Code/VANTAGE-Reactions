#ifndef REACTIONS_IPL_CS_H
#define REACTIONS_IPL_CS_H
#include "../cross_section_abstract.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief An inverse power law cross-section, proportional to v_rel^{-b}, with b
 * < 1. The rate is unbounded, so a default maximum velocity needs to be
 * suplied. NOTE: Ideally, the default maximum rate call wouldn't be used.
 */
struct IPLCrossSection : public AbstractCrossSection {

  IPLCrossSection() = default;
  /**
   * @brief Constructor for IPLCrossSection.
   *
   * @param reference_sigma Reference cross-section
   * @param reference_vel Reference velocity value
   * @param power Inverse power law power (<1)
   * @param default_max_vel The maximum allowed relative velocity for the
   * get_max_rate_val() call
   */
  IPLCrossSection(REAL reference_sigma, REAL reference_vel, REAL power,
                  REAL default_max_vel)
      : mult_const(reference_sigma * std::pow(reference_vel, power)),
        power(power), default_max_vel(default_max_vel) {};

  /**
   * @brief Returns the cross-section value at given relative velocity
   *
   * @param relative_vel Relative velocity of projectile and target
   * @return REAL-valued cross-section
   */
  REAL get_value_at(const REAL &relative_vel) const {
    return this->mult_const / Kernel::pow(relative_vel, this->power);
  };

  /**
   * @brief Returns maximum value of the rate sigma*v of for this cross-section.
   *
   * @return REAL-valued constant
   */
  REAL get_max_rate_val() const {
    return this->mult_const /
           Kernel::pow(this->default_max_vel, 1 - this->power);
  };

  /**
   * @brief Get the greedy maximum value of sigma*v_r where sigma is this
   * cross-section evaluated at v_r and v_r is the relative speed of the
   * projectile and target. This should (in most cases) be implemented to return
   * values less than or equal to the one with get_max_rate_val()
   *
   * @param relative_vel Magnitude of relative velocity of the projectile and
   * target
   * @return REAL-valued maximum rate
   */
  REAL get_max_rate_val_greedy(const REAL &relative_vel) const {
    return this->mult_const / Kernel::pow(relative_vel, 1 - this->power);
  };

private:
  REAL mult_const;
  REAL power;
  REAL default_max_vel;
};
}; // namespace VANTAGE::Reactions
#endif
