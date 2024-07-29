#pragma once
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <stdexcept>

// TODO: Update docs
using namespace NESO::Particles;

/**
 * @brief Base reaction data object.
 */

struct ReactionDataBase {

  ReactionDataBase() {
    auto rng_lambda = [&]() -> REAL { return 0; };
    this->rng_kernel = host_kernel_rng<REAL>(rng_lambda, 0);
  }
  ReactionDataBase(Properties<INT> required_int_props)
      : required_int_props(required_int_props) {
    auto rng_lambda = [&]() -> REAL { return 0; };
    this->rng_kernel = host_kernel_rng<REAL>(rng_lambda, 0);
  }

  ReactionDataBase(Properties<REAL> required_real_props)
      : required_real_props(required_real_props) {
    auto rng_lambda = [&]() -> REAL { return 0; };
    this->rng_kernel = host_kernel_rng<REAL>(rng_lambda, 0);
  }

  ReactionDataBase(Properties<INT> required_int_props,
                   Properties<REAL> required_real_props)
      : required_int_props(required_int_props),
        required_real_props(required_real_props) {
    auto rng_lambda = [&]() -> REAL { return 0; };
    this->rng_kernel = host_kernel_rng<REAL>(rng_lambda, 0);
  }


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
  void set_rng_kernel(std::shared_ptr<KernelRNG<REAL>> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }

  std::shared_ptr<KernelRNG<REAL>> get_rng_kernel() { return this->rng_kernel; }

protected:
  Properties<INT> required_int_props;
  Properties<REAL> required_real_props;
  std::shared_ptr<KernelRNG<REAL>> rng_kernel;
};

/**
 * @brief Base reaction data object to be used on SYCL devices.
 */
struct ReactionDataBaseOnDevice {
  ReactionDataBaseOnDevice() = default;

  /**
   * @brief Virtual function to calculate the reaction rate.
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   */
  virtual REAL
  calc_rate(const Access::LoopIndex::Read &index,
            const Access::SymVector::Read<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            const Access::KernelRNG::Read<REAL> &rng_kernel) const {
    return 0.0;
  }
};
