#pragma once
#include "particle_spec.hpp"
#include "typedefs.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <ionisation_reactions_data/amjuel_ionisation_data.hpp>
#include <ionisation_reactions_kernels/base_ionisation_kernels.hpp>
#include <memory>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;

template <int num_coeffs>
struct IoniseReactionAMJUEL
    : public LinearReactionBase<0, IoniseReactionAMJUELData<num_coeffs>,
                                IoniseReactionKernels> {

  IoniseReactionAMJUEL() = default;

  IoniseReactionAMJUEL(SYCLTargetSharedPtr sycl_target_,
                       Sym<REAL> total_reaction_rate_, int in_states_,
                       REAL density_normalisation,
                       std::array<REAL, num_coeffs> coeffs)
      : LinearReactionBase<0, IoniseReactionAMJUELData<num_coeffs>,
                           IoniseReactionKernels>(
            sycl_target_, total_reaction_rate_,
            std::vector<Sym<REAL>>{
                Sym<REAL>("V"), Sym<REAL>("ELECTRON_TEMPERATURE"),
                Sym<REAL>("ELECTRON_DENSITY"), Sym<REAL>("SOURCE_ENERGY"),
                Sym<REAL>("SOURCE_MOMENTUM"), Sym<REAL>("SOURCE_DENSITY"),
                Sym<REAL>("COMPUTATIONAL_WEIGHT"),
                Sym<REAL>("FLUID_TEMPERATURE"), Sym<REAL>("FLUID_DENSITY")},
            std::vector<Sym<REAL>>{
                Sym<REAL>("V"), Sym<REAL>("ELECTRON_TEMPERATURE"),
                Sym<REAL>("ELECTRON_DENSITY"), Sym<REAL>("SOURCE_ENERGY"),
                Sym<REAL>("SOURCE_MOMENTUM"), Sym<REAL>("SOURCE_DENSITY"),
                Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<INT>>(), std::vector<Sym<INT>>(), in_states_,
            std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
            std::vector<ParticleProp<INT>>{},
            IoniseReactionAMJUELData<num_coeffs>(density_normalisation, coeffs),
            IoniseReactionKernels<0>()) {}
};