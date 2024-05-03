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

  virtual REAL calc_rate(Access::LoopIndex::Read &index,
                         Access::SymVector::Read<INT> &req_part_ints,
                         Access::SymVector::Read<REAL> &req_part_reals,
                         Access::SymVector::Read<INT> &req_field_ints,
                         Access::SymVector::Read<REAL> &req_field_reals) const {
    return 0.0;
  }

  virtual const int get_num_particle_int_props() { return 0; }
  virtual const int get_num_particle_real_props() { return 0; }

  virtual const int get_num_field_int_props() { return 0; }
  virtual const int get_num_field_real_props() { return 0; }

  virtual const char **get_required_particle_int_props() {
    static const char *required_prop_names[] = {};
    return required_prop_names;
  }
  virtual const char **get_required_particle_real_props() {
    static const char *required_prop_names[] = {};
    return required_prop_names;
  }

  virtual const char **get_required_field_int_props() {
    static const char *required_prop_names[] = {};
    return required_prop_names;
  }
  virtual const char **get_required_field_real_props() {
    static const char *required_prop_names[] = {};
    return required_prop_names;
  }
};
