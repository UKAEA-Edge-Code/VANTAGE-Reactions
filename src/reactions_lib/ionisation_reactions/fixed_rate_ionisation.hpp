#pragma once
#include <cmath>
#include <gtest/gtest.h>
#include <ionisation_reactions_data/fixed_rate_ionisation_data.hpp>
#include <ionisation_reactions_kernels/base_ionisation_kernels.hpp>
#include <memory>
#include <neso_particles.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernel_pre_reqs.hpp>
#include <reaction_kernels.hpp>
#include <string_view>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;

/**
 * @brief Struct defining a fixed-rate ionisation reaction.
 *
 * @param sycl_target Compute device used by the instance.
 * @param total_reaction_rate_ Symbol index for a ParticleDat that's used to
 * track the cumulative weighted reaction rate modification imposed on all of
 * the particles in the ParticleSubGroup passed to run_rate_loop(...).
 * @param rate_ REAL-valued rate to be used in reaction rate calculation.
 * @param in_states_ Integer specifying the ID of the species on
 * which the derived reaction is acting on.
 */
struct FixedRateIonisation
    : public LinearReactionBase<0, FixedRateIonisationData,
                                IoniseReactionKernels<2, 2>> {

  FixedRateIonisation() = default;

  FixedRateIonisation(SYCLTargetSharedPtr sycl_target_,
                      Sym<REAL> total_reaction_rate_, REAL rate_,
                      int in_states_, const ParticleSpec& particle_spec)
      : LinearReactionBase<0, FixedRateIonisationData, IoniseReactionKernels<2, 2>>(
            sycl_target_, total_reaction_rate_, in_states_,
            std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
            std::vector<ParticleProp<INT>>{}, FixedRateIonisationData(rate_),
            IoniseReactionKernels<2, 2>(Species("ELECTRON")), particle_spec) {}
};