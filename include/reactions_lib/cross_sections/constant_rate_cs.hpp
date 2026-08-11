#ifndef REACTIONS_CONSTANT_RATE_CS_H
#define REACTIONS_CONSTANT_RATE_CS_H
#include "../reaction_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <limits>

namespace VANTAGE::Reactions {

/**
 * @brief A struct that defines a cross section evaluating to K/v_r
 * where K is a constant rate and v_r is the relative velocity. Leads to always
 * accepting in rejection algorithms weighted by this cross-section
 */
struct ConstantRateCrossSection : public AbstractCrossSection {

  ConstantRateCrossSection() = default;
  /**
   * @brief Constructor for ConstantRateCrossSection.
   *
   * @param constant_sigma_v Constant collision rate
   */
  ConstantRateCrossSection(NP::REAL constant_sigma_v)
      : constant_sigma_v(constant_sigma_v) {};

  /**
   * @brief Returns the cross-section value at given relative velocity
   *
   * @param relative_vel Relative velocity of projectile and target
   * @return NP::REAL-valued cross-section = K/v_r
   */
  NP::REAL get_value_at(const NP::REAL &relative_vel) const {
    return this->constant_sigma_v / relative_vel;
  };

  /**
   * @brief Returns maximum value of the rate sigma*v of for this cross-section.
   * This is constant in this class.
   *
   * @return NP::REAL-valued constant (plus floating point error to account for
   * potential use in explicit rejection methods).
   */
  NP::REAL get_max_rate_val() const {
    // Avoid potential comparison issues with
    // sigma*v by raising the max rate slightly - might cause the occasional
    // spurious rejection in explicit rejection methods
    return this->constant_sigma_v +
           10 * std::numeric_limits<NP::REAL>::epsilon();
  };

  /**
   * @brief Always accepts the relative velocity, regardless of uniform random
   * number
   *
   * @param relative_vel Relative velocity of projectile and target
   * @param uniform_rand Uniformly distributed random number
   * @return true
   */
  bool accept_reject(NP::REAL relative_vel, NP::REAL uniform_rand,
                     NP::REAL value_at, NP::REAL max_rate_val) const {
    return true;
  }

private:
  NP::REAL constant_sigma_v;
};
}; // namespace VANTAGE::Reactions
#endif
