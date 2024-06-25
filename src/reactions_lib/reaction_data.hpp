#pragma once
#include <neso_particles.hpp>
#include <neso_particles/containers/sym_vector.hpp>

using namespace NESO::Particles;

/**
 * @brief Base reaction data object.
 */

struct ReactionDataBase {

  ReactionDataBase() = default;

  /**
   * @brief Virtual getters functions that can be overidden by an implementation
   * in a derived struct.
   */

  virtual std::vector<std::string> get_required_int_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }

  virtual std::vector<std::string> get_required_real_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }
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
            const Access::SymVector::Read<REAL> &req_real_props) const {
    return 0.0;
  }
};
