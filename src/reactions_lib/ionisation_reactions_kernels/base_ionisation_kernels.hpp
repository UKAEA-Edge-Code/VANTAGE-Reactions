#pragma once
#include "containers/sym_vector.hpp"
#include "ionisation_reactions_kernels/base_ionisation_kernels.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <reaction_kernels.hpp>
#include <vector>

#define PARTICLE_REAL_PROPS X(velocity), X(weight)

#define FIELD_REAL_PROPS                                                       \
  X(electron_density), X(source_density), X(source_momentum), X(source_energy)

using namespace NESO::Particles;

template <INT num_products_per_parent>
struct IoniseReactionKernels
    : public ReactionKernelsBase<num_products_per_parent> {
  IoniseReactionKernels() = default;

  void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::DescendantProducts::Write &descendant_products,
                       Access::SymVector::Write<INT> &req_particle_ints,
                       Access::SymVector::Write<REAL> &req_particle_reals,
                       Access::SymVector::Write<INT> &req_field_ints,
                       Access::SymVector::Write<REAL> &req_field_reals,
                       const std::array<int, 0> &out_states,
                       Access::LocalArray::Read<REAL> &pre_req_data,
                       double dt) const {
    // auto k_V_0 = write_req_reals.at(0, index, 0);
    // auto k_V_1 = write_req_reals.at(0, index, 1);
    auto k_V_0 = req_particle_reals.at(velocity, index, 0);
    auto k_V_1 = req_particle_reals.at(velocity, index, 1);

    const REAL vsquared = (k_V_0 * k_V_0) + (k_V_1 * k_V_1);

    REAL k_n_scale = 1.0; // / test_reaction_data.get_n_to_SI();
    REAL inv_k_dt = 1.0 / dt;

    auto nE = req_field_reals.at(electron_density, index, 0);

    // Set SOURCE_DENSITY
    req_field_reals.at(source_density, index, 0) +=
        nE * modified_weight * k_n_scale * inv_k_dt;

    // Get SOURCE_DENSITY for SOURCE_MOMENTUM calc
    auto k_SD = req_field_reals.at(source_density, index, 0);
    req_field_reals.at(source_momentum, index, 0) += k_SD * k_V_0;
    req_field_reals.at(source_momentum, index, 1) += k_SD * k_V_1;

    // Set SOURCE_ENERGY
    req_field_reals.at(source_energy, index, 0) += k_SD * vsquared * 0.5;

    req_particle_reals.at(weight, index, 0) -= modified_weight;
  }

public:
#define X(M) M
  enum { PARTICLE_REAL_PROPS, NUM_PARTICLE_REAL_PROPS };
#undef X
  const int get_num_particle_real_props() { return NUM_PARTICLE_REAL_PROPS; }
#define X(M) #M
  const char *required_particle_real_prop_names[NUM_PARTICLE_REAL_PROPS] = {
      PARTICLE_REAL_PROPS};
#undef X
  const char **get_required_particle_real_props() {
    return required_particle_real_prop_names;
  }
#define X(M) M
  enum { FIELD_REAL_PROPS, NUM_FIELD_REAL_PROPS };
#undef X
  const int get_num_field_real_props() { return NUM_FIELD_REAL_PROPS; }
#define X(M) #M
  const char *required_field_real_prop_names[NUM_FIELD_REAL_PROPS] = {
      FIELD_REAL_PROPS};
#undef X
  const char **get_required_field_real_props() {
    return required_field_real_prop_names;
  }
};
#undef PARTICLE_REAL_PROPS
#undef FIELD_REAL_PROPS