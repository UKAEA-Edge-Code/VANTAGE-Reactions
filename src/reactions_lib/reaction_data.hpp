#pragma once
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <stdexcept>

using namespace NESO::Particles;
namespace Reactions {

/**
 * struct AbstractCrossSection - Abstract base class for cross-section objects.
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
/**
 * @brief Base reaction data object.
 *
 * @param required_int_props Properties<INT> object containing information
 * regarding the required INT-based properties for the reaction data.
 * @param required_real_props Properties<REAL> object containing information
 * regarding the required REAL-based properties for the reaction data.
 * @param required_int_props_ephemeral Properties<INT> object containing
 * information regarding the required INT-based ephemeral properties for the
 * reaction data.
 * @param required_real_props_ephemeral Properties<REAL> object containing
 * information regarding the required REAL-based ephemeral properties for the
 * reaction data.
 * @param properties_map A std::map<int, std::string> object to be used when
 * retrieven property names, i.e. remapping default property names
 * get_required_int_props(...)).
 */
template <size_t dim = 1, typename RNG_TYPE = HostPerParticleBlockRNG<REAL>>
struct ReactionDataBase {

  using RNG_KERNEL_TYPE = RNG_TYPE;
  ReactionDataBase(Properties<INT> required_int_props,
                   Properties<REAL> required_real_props,
                   Properties<INT> required_int_props_ephemeral,
                   Properties<REAL> required_real_props_ephemeral,
                   std::map<int, std::string> properties_map = default_map)
      : required_int_props(required_int_props),
        required_real_props(required_real_props),
        required_int_props_ephemeral(required_int_props_ephemeral),
        required_real_props_ephemeral(required_real_props_ephemeral),
        properties_map(properties_map) {
    auto rng_lambda = [&]() -> REAL { return 0; };
    this->rng_kernel = std::make_shared<RNG_TYPE>(rng_lambda, 0);
  }

  ReactionDataBase(std::map<int, std::string> properties_map = default_map)
      : ReactionDataBase(Properties<INT>(), Properties<REAL>(),
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  ReactionDataBase(Properties<INT> required_int_props,
                   std::map<int, std::string> properties_map = default_map)
      : ReactionDataBase(required_int_props, Properties<REAL>(),
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  ReactionDataBase(Properties<REAL> required_real_props,
                   std::map<int, std::string> properties_map = default_map)
      : ReactionDataBase(Properties<INT>(), required_real_props,
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  ReactionDataBase(Properties<INT> required_int_props,
                   Properties<REAL> required_real_props,
                   std::map<int, std::string> properties_map = default_map)
      : ReactionDataBase(required_int_props, required_real_props,
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}
  /**
   * @brief Return all required integer property names, including ephemeral
   * properties
   *
   */
  std::vector<std::string> get_required_int_props() {
    auto names = this->required_int_props.get_prop_names(this->properties_map);
    auto ephemeral_names =
        this->required_int_props_ephemeral.get_prop_names(this->properties_map);
    names.insert(names.end(), ephemeral_names.begin(), ephemeral_names.end());
    return names;
  }

  /**
   * @brief Return all required real property names, including ephemeral
   * properties
   *
   */
  std::vector<std::string> get_required_real_props() {
    auto names = this->required_real_props.get_prop_names(this->properties_map);
    auto ephemeral_names = this->required_real_props_ephemeral.get_prop_names(
        this->properties_map);
    names.insert(names.end(), ephemeral_names.begin(), ephemeral_names.end());
    return names;
  }

  /**
   * @brief Return names of required ephemeral integer properties
   *
   */
  std::vector<std::string> get_required_int_props_ephemeral() {
    return this->required_int_props_ephemeral.get_prop_names(
        this->properties_map);
  }
  /**
   * @brief Return names of required ephemeral real properties
   *
   */
  std::vector<std::string> get_required_real_props_ephemeral() {
    return this->required_real_props_ephemeral.get_prop_names(
        this->properties_map);
  }

  void set_rng_kernel(std::shared_ptr<RNG_TYPE> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }

  std::shared_ptr<RNG_TYPE> get_rng_kernel() { return this->rng_kernel; }

  static constexpr size_t get_dim() { return dim; }

protected:
  Properties<INT> required_int_props;
  Properties<REAL> required_real_props;
  Properties<INT> required_int_props_ephemeral;
  Properties<REAL> required_real_props_ephemeral;
  std::shared_ptr<RNG_TYPE> rng_kernel;
  std::map<int, std::string> properties_map;
};

/**
 * @brief Base reaction data object to be used on SYCL devices.
 */
template <size_t dim = 1, typename RNG_TYPE = HostPerParticleBlockRNG<REAL>>
struct ReactionDataBaseOnDevice {
  using RNG_KERNEL_TYPE = RNG_TYPE;
  ReactionDataBaseOnDevice() = default;

  /**
   * @brief Function to calculate the reaction data.
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction data calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction data calculation.
   * @param rng_kernel The random number generator kernel potentially used in
   * the calculation
   */
  std::array<REAL, dim>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename RNG_TYPE::KernelType &rng_kernel) const {
    return std::array<REAL, dim>{0.0};
  }

  static constexpr size_t get_dim() { return dim; }
};
}; // namespace Reactions
