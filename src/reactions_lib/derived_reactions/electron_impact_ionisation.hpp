#pragma once
#include <data_calculator.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_data/fixed_rate_data.hpp>
#include <reaction_kernel_pre_reqs.hpp>
#include <reaction_kernels.hpp>
#include <reaction_kernels/base_ionisation_kernels.hpp>

using namespace NESO::Particles;
using namespace Reactions;
// TODO docs
/**
 * @brief Struct defining a fixed-rate ionisation reaction.
 *
 * @param sycl_target Compute device used by the instance.
 * @param total_reaction_rate_ Symbol index for a ParticleDat that's used to
 * track the cumulative weighted reaction rate modification imposed on all of
 * the particles in the ParticleSubGroup passed to run_rate_loop(...).
 * @param rate_ REAL-valued rate to be used in reaction rate calculation.
 * @param energy_rate_ REAL-valued rate to be used in reaction energy loss rate
 * calculation.
 * @param in_states_ Integer specifying the ID of the species on
 * which the derived reaction is acting on.
 * @param particle_spec ParticleSpec object containing particle properties to
 * use to construct sym_vectors.
 */
template <typename RateData, typename EnergyRateData, int ndim = 2>
struct ElectronImpactIonisation
    : public LinearReactionBase<0, RateData, IoniseReactionKernels<ndim>,
                                DataCalculator<EnergyRateData>> {

  ElectronImpactIonisation(SYCLTargetSharedPtr sycl_target_,
                           Sym<REAL> total_reaction_rate, RateData rate_data,
                           EnergyRateData energy_rate_data,
                           Species target_species, Species electron_species,
                           const ParticleSpec &particle_spec)
      : LinearReactionBase<0, RateData, IoniseReactionKernels<ndim>,
                           DataCalculator<EnergyRateData>>(
            sycl_target_, total_reaction_rate, target_species.get_id(),
            std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
            std::vector<ParticleProp<INT>>{}, rate_data,
            IoniseReactionKernels<ndim>(target_species, electron_species,
                                        electron_species),
            DataCalculator<EnergyRateData>(particle_spec, energy_rate_data),
            particle_spec) {}
};
