#pragma once
#include <neso_particles.hpp>

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
  virtual const int get_num_simple_int_props() { return 0; }
  virtual const int get_num_simple_real_props() { return 0; }

  virtual const int get_num_species_int_props() { return 0; }
  virtual const int get_num_species_real_props() { return 0; }

  virtual std::vector<std::string> get_required_simple_int_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }
  virtual std::vector<std::string> get_required_simple_real_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }

  virtual std::vector<std::string> get_required_species_int_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }
  virtual std::vector<std::string> get_required_species_real_props() {
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
   * @param req_simple_prop_ints Vector of symbols for simple integer-valued
   * properties that need to be used for the reaction rate calculation.
   * @param req_simple_prop_reals Vector of symbols for simple real-valued
   * properties that need to be used for the reaction rate calculation.
   * @param req_species_prop_ints Vector of symbols for species-dependent
   * integer-valued properties that need to be used for the reaction rate
   * calculation.
   * @param req_species_prop_reals Vector of symbols for species-dependent
   * real-valued properties that need to be used for the reaction rate
   * calculation.
   */
  virtual REAL
  calc_rate(Access::LoopIndex::Read &index,
            Access::SymVector::Read<INT> &req_simple_prop_ints,
            Access::SymVector::Read<REAL> &req_simple_prop_reals,
            Access::SymVector::Read<INT> &req_species_prop_ints,
            Access::SymVector::Read<REAL> &req_species_prop_reals) const {
    return 0.0;
  }
};
