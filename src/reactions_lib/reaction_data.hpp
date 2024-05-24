#pragma once
#include "containers/sym_vector.hpp"
#include "loop/particle_loop_index.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;

/**
 * @brief Base reaction data object.
 */

struct ReactionDataBase {

  ReactionDataBase() = default;

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

struct ReactionDataBaseOnDevice {
  ReactionDataBaseOnDevice() = default;

  virtual REAL calc_rate(Access::LoopIndex::Read &index,
                         Access::SymVector::Read<INT> &req_simple_prop_ints,
                         Access::SymVector::Read<REAL> &req_simple_prop_reals,
                         Access::SymVector::Read<INT> &req_species_prop_ints,
                         Access::SymVector::Read<REAL> &req_species_prop_reals) const {
    return 0.0;
  }

};
