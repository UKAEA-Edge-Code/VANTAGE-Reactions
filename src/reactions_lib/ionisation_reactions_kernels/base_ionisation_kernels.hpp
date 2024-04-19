#pragma once
#include "containers/sym_vector.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

#define BASE_IONISATION_PROPS \
  X(velocity), X(electron_density), X(source_density),\
  X(source_momentum), X(source_energy), X(weight)

using namespace NESO::Particles;
using namespace Reactions;

template <INT num_products_per_parent>
struct IoniseReactionKernels
    : public ReactionKernelsBase<num_products_per_parent> {
  IoniseReactionKernels() = default;

  void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::DescendantProducts::Write &descendant_products,
                       Access::SymVector::Read<INT> &read_req_ints,
                       Access::SymVector::Read<REAL> &read_req_reals,
                       Access::SymVector::Write<INT> &write_req_ints,
                       Access::SymVector::Write<REAL> &write_req_reals,
                       const std::array<int, 0> &out_states,
                       Access::LocalArray::Read<REAL> &pre_req_data,
                       double dt) const {
    // auto k_V_0 = write_req_reals.at(0, index, 0);
    // auto k_V_1 = write_req_reals.at(0, index, 1);
    auto k_V_0 = write_req_reals.at(velocity, index, 0);
    auto k_V_1 = write_req_reals.at(velocity, index, 1);

    const REAL vsquared = (k_V_0 * k_V_0) + (k_V_1 * k_V_1);

    REAL k_n_scale = 1.0; // / test_reaction_data.get_n_to_SI();
    REAL inv_k_dt = 1.0 / dt;

    auto nE = write_req_reals.at(electron_density, index, 0);

    // Set SOURCE_DENSITY
    write_req_reals.at(source_density, index, 0) +=
        nE * modified_weight * k_n_scale * inv_k_dt;

    // Get SOURCE_DENSITY for SOURCE_MOMENTUM calc
    auto k_SD = write_req_reals.at(source_density, index, 0);
    write_req_reals.at(source_momentum, index, 0) += k_SD * k_V_0;
    write_req_reals.at(source_momentum, index, 1) += k_SD * k_V_1;

    // Set SOURCE_ENERGY
    write_req_reals.at(source_energy, index, 0) += k_SD * vsquared * 0.5;

    write_req_reals.at(weight, index, 0) -= modified_weight;
  }

public:
  #define X(M) M
    enum { BASE_IONISATION_PROPS, NUM_PROPS };
  #undef X
  const int get_num_props() {
    return NUM_PROPS;
  }
  #define X(M) #M
    const char* required_prop_names[NUM_PROPS] = { BASE_IONISATION_PROPS };
  #undef X
  const char** get_required_properties() {
    return required_prop_names;
  }
};
#undef BASE_IONISATION_PROPS