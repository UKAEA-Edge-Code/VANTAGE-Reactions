#pragma once
#include "particle_spec.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <ionisation_reactions_data/fixed_rate_ionisation_data.hpp>
#include <ionisation_reactions_kernels/base_ionisation_kernels.hpp>
#include <memory>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;

struct FixedRateIonisation
    : public LinearReactionBase<0, FixedRateIonisationData,
                                IoniseReactionKernels> {
  FixedRateIonisation() = default;

  FixedRateIonisation(SYCLTargetSharedPtr sycl_target_,
                      Sym<REAL> total_reaction_rate_, REAL rate_,
                      int in_states_)
      : LinearReactionBase<0, FixedRateIonisationData, IoniseReactionKernels>(
            sycl_target_, total_reaction_rate_, in_states_,
            std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
            std::vector<ParticleProp<INT>>{}, FixedRateIonisationData(rate_),
            IoniseReactionKernels<0>()) {}
};