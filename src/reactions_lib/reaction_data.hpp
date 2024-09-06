#pragma once
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <stdexcept>

//TODO: Generalise cross-section get_max_rate_val()
using namespace NESO::Particles;

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
  virtual REAL get_value_at(const REAL &relative_vel) const {};

  /**
   * @brief Get the maximum value of sigma*v_r where sigma is this cross-section
   * evaluated at v_r and v_r is the relative speed of the projectile and target
   *
   * @return REAL-valued maximum rate
   */
  virtual REAL get_max_rate_val() const {};

  /**
   * @brief Accept-reject function for when this cross-section is used in
   * rejection methods. Accepts if the uniform random number on (0,1) is less
   * than the ratio of sigma*v evaluated at a given relative speed to the
   * maximum value of sigma*v.
   *
   * @param relative_vel Magnitude of relative velocity of the projectile and
   * target
   * @param uniform_rand Uniformly sampled random number on (0,1)
   * @return true if relative_vel value is accepted, false otherwise
   */
  virtual bool accept_reject(REAL relative_vel, REAL uniform_rand) const {
    return uniform_rand < this->get_value_at(relative_vel) * relative_vel /
                              this->get_max_rate_val();
  }
};
/**
 * @brief Base reaction data object.
 */
template <size_t dim = 1, typename RNG_TYPE = HostPerParticleBlockRNG<REAL>>
struct ReactionDataBase {

  using RNG_KERNEL_TYPE = RNG_TYPE;
  ReactionDataBase(Properties<INT> required_int_props,
                   Properties<REAL> required_real_props)
      : required_int_props(required_int_props),
        required_real_props(required_real_props) {
    auto rng_lambda = [&]() -> REAL { return 0; };
    this->rng_kernel = std::make_shared<RNG_TYPE>(rng_lambda, 0);
  }

  ReactionDataBase()
      : ReactionDataBase(Properties<INT>(), Properties<REAL>()) {}

  ReactionDataBase(Properties<INT> required_int_props)
      : ReactionDataBase(required_int_props, Properties<REAL>()) {}

  ReactionDataBase(Properties<REAL> required_real_props)
      : ReactionDataBase(Properties<INT>(), required_real_props) {}

  std::vector<std::string> get_required_int_props() {
    std::vector<std::string> prop_names;
    try {
      prop_names = this->required_int_props.get_prop_names();
    } catch (std::logic_error) {
    }
    return prop_names;
  }

  std::vector<std::string> get_required_real_props() {
    std::vector<std::string> prop_names;
    try {
      prop_names = this->required_real_props.get_prop_names();
    } catch (std::logic_error) {
    }
    return prop_names;
  }
  void set_rng_kernel(std::shared_ptr<RNG_TYPE> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }

  std::shared_ptr<RNG_TYPE> get_rng_kernel() { return this->rng_kernel; }

  static constexpr size_t get_dim() { return dim; }

protected:
  Properties<INT> required_int_props;
  Properties<REAL> required_real_props;
  std::shared_ptr<RNG_TYPE> rng_kernel;
};

/**
 * @brief Base reaction data object to be used on SYCL devices.
 */
template <size_t dim = 1, typename RNG_TYPE = HostPerParticleBlockRNG<REAL>>
struct ReactionDataBaseOnDevice {
  using RNG_KERNEL_TYPE = RNG_TYPE;
  ReactionDataBaseOnDevice() = default;

  /**
   * @brief Virtual function to calculate the reaction data.
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
  virtual std::array<REAL, dim>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Read<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename RNG_TYPE::KernelType &rng_kernel) const {
    return std::array<REAL, dim>{0.0};
  }

  static constexpr size_t get_dim() { return dim; }
};
