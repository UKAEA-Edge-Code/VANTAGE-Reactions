#pragma once
#include <cmath>
#include <gtest/gtest.h>
#include <ionisation_reactions_data/amjuel_ionisation_data.hpp>
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
 * @brief Struct defining a 1D AMJUEL-based ionisation reaction.
 *
 * @tparam num_coeffs The number of coefficients needed for a 1D AMJUEL reaction
 * rate calculation.
 * @param sycl_target Compute device used by the instance.
 * @param total_reaction_rate_ Symbol index for a ParticleDat that's used to
 * track the cumulative weighted reaction rate modification imposed on all of
 * the particles in the ParticleSubGroup passed to run_rate_loop(...).
 * @param in_states_ Integer specifying the ID of the species on
 * which the derived reaction is acting on.
 * @param density_normalisation A normalisation constant to be used in a 1D
 * AMJUEL reaction rate calculation.
 * @param coeffs A real-valued array of coefficients to be used in a 1D AMJUEL
 * reaction rate calculation.
 */
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
            sycl_target_, total_reaction_rate_, in_states_,
            std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
            std::vector<ParticleProp<INT>>{},
            IoniseReactionAMJUELData<num_coeffs>(density_normalisation, coeffs),
            IoniseReactionKernels(std::vector<Species>{Species("ELECTRON")})) {}
};