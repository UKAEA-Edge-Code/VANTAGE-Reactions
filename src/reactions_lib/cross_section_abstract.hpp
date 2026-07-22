#ifndef REACTIONS_CROSS_SECTION_ABSTRACT_H
#define REACTIONS_CROSS_SECTION_ABSTRACT_H
#include <limits>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief An abstract base class for cross-section objects.
 * All classes derived from this class should be device copyable in order to be
 * used within ReactionData classes.
 */
struct AbstractCrossSection {

  /**
   * @brief Get the value of the cross section for a given relative velocity
   * value of projectile and target
   *
   * @param relative_vel Magnitude of relative velocity of target and projectile
   * @return REAL-valued cross-section at requested relative vel magnitude
   */
  REAL get_value_at(const REAL &relative_vel) const {
    // This function should never actually be called. If it is called and we do
    // not have a return value then the calling function will receive an
    // undefined value. By setting a value we at least know what the returned
    // value is and can pick one that is detectable.
    return std::numeric_limits<REAL>::lowest();
  };

  /**
   * @brief Get the maximum value of sigma*v_r where sigma is this cross-section
   * evaluated at v_r and v_r is the relative speed of the projectile and target
   *
   * @return REAL-valued maximum rate
   */
  REAL get_max_rate_val() const {
    // This function should never actually be called. If it is called and we do
    // not have a return value then the calling function will receive an
    // undefined value. By setting a value we at least know what the returned
    // value is and can pick one that is detectable.
    return std::numeric_limits<REAL>::lowest();
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
    // TODO: check if this still causes issues (see workarounds below)
    return this->get_max_rate_val();
  };
  /**
   * @brief Accept-reject function for when this cross-section is used in
   * rejection methods. Accepts if the uniform random number on (0,1) is less
   * than the ratio of sigma*v evaluated at a given relative speed to the
   * maximum value of sigma*v.
   *
   * @param relative_vel Magnitude of relative velocity of the projectile and
   * target
   * @param uniform_rand Uniformly sampled random number on (0,1)
   * @param value_at Value of cross section for a given relative velocity value
   * of projectile and target (NOTE this is currently a workaround due to the
   * limitation on calling get_value_at(...) inside this function.)
   * @param max_rate_val Maximum value of sigma*v_r (NOTE this is currently a
   * workaround due to the limitation on calling get_max_rate_val(...) inside
   * this function.)
   * @return true if relative_vel value is accepted, false otherwise
   */
  bool accept_reject(REAL relative_vel, REAL uniform_rand, REAL value_at,
                     REAL max_rate_val) const {
    return uniform_rand < (value_at * relative_vel / max_rate_val);
  }
};

}; // namespace VANTAGE::Reactions
#endif
