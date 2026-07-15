#ifndef REACTIONS_CONSTANT_CS_H
#define REACTIONS_CONSTANT_CS_H
#include "../cross_section_abstract.hpp"
#include <limits>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief A constant cross section. The rate is unbounded, so a default maximum
 * velocity needs to be suplied. NOTE: Ideally, the default maximum rate call
 * wouldn't be used.
 */
struct ConstantCrossSection : public AbstractCrossSection {

  ConstantCrossSection() = default;
  /**
   * @brief Constructor for ConstantCrossSection.
   *
   * @param constant_sigma Constant collision rate
   * @param default_max_vel The maximum allowed relative velocity for the
   * get_max_rate_val() call
   */
  ConstantCrossSection(REAL constant_sigma, REAL default_max_vel)
      : constant_sigma(constant_sigma), default_max_vel(default_max_vel) {};

  /**
   * @brief Returns the cross-section value at given relative velocity
   *
   * @param relative_vel Relative velocity of projectile and target
   * @return REAL-valued cross-section
   */
  REAL get_value_at(const REAL &relative_vel) const {
    return this->constant_sigma;
  };

  /**
   * @brief Returns maximum value of the rate sigma*v of for this cross-section.
   *
   * @return REAL-valued constant
   */
  REAL get_max_rate_val() const {
    return this->constant_sigma * this->default_max_vel;
  };

  /**
   * @brief Get the greedy maximum value of sigma*v_r.
   *
   * @param relative_vel Magnitude of relative velocity of the projectile and
   * target
   * @return REAL-valued maximum rate
   */
  REAL get_max_rate_val(REAL relative_vel) const {
    return this->constant_sigma * relative_vel;
  };

private:
  REAL constant_sigma;
  REAL default_max_vel;
};
}; // namespace VANTAGE::Reactions
#endif
