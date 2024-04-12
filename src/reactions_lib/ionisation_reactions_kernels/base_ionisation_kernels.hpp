#pragma once
#include "containers/sym_vector.hpp"
#include "particle_properties_map.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

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

  void set_var_indices(std::vector<int> &indices) {
    using namespace ParticlePropertiesIndices;

    std::array<prop_inds, 6> props = {
        prop_inds::velocity,       prop_inds::electron_density,
        prop_inds::source_density, prop_inds::source_momentum,
        prop_inds::source_energy,  prop_inds::weight};

    int unmatched_count = 0;
    for (int i = 0; i < indices.size(); i++) {
      if (std::find(props.begin(), props.end(), indices.at(i)) == props.end()) {
        unmatched_count += 1;
      }
      switch (indices.at(i)) {
      case prop_inds::velocity:
        velocity = i;
        break;
      case prop_inds::electron_density:
        electron_density = i;
        break;
      case prop_inds::source_density:
        source_density = i;
        break;
      case prop_inds::source_momentum:
        source_momentum = i;
        break;
      case prop_inds::source_energy:
        source_energy = i;
        break;
      case prop_inds::weight:
        weight = i;
        break;
      }
    }
    NESOASSERT((unmatched_count != props.size()),
               "Sym_vector indices don't map to indices of particle properties "
               "required by IoniseReactionKernels...");
  }

public:
  int velocity, electron_density, source_density, source_momentum,
      source_energy, weight;
};