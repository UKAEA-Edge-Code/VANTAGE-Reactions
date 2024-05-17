#pragma once
#include "particle_properties_map.hpp"
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
#include <reaction_kernel_pre_reqs.hpp>
#include <reaction_kernels.hpp>
#include <string_view>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;
using namespace ParticlePropertiesIndices;

namespace FIXED_RATE_IONISATION {
constexpr int num_products_per_parent = 0;

const std::array<std::string, 1> species = {"ELECTRON"};

constexpr std::array<int, 2> required_particle_real_props = {velocity,
                                                                    weight};

constexpr std::array<int, 4> required_field_real_props = {
    density, source_density, source_momentum, source_energy};

const auto required_particle_real_props_obj =
    RequiredParticleRealProperties<required_particle_real_props.size()>(
        required_particle_real_props);

const auto required_species_field_real_props_obj =
    RequiredSpeciesFieldProperties<std::size(species),
                                   required_field_real_props.size()>(
        species, required_field_real_props);

} // namespace FIXED_RATE_IONISATION

struct FixedRateIonisation
    : public LinearReactionBase<
          FIXED_RATE_IONISATION::num_products_per_parent,
          FixedRateIonisationData, IoniseReactionKernels> {

  FixedRateIonisation() = default;

  FixedRateIonisation(SYCLTargetSharedPtr sycl_target_,
                      Sym<REAL> total_reaction_rate_, REAL rate_,
                      int in_states_)
      : LinearReactionBase<
            FIXED_RATE_IONISATION::num_products_per_parent,
            FixedRateIonisationData, IoniseReactionKernels>(
            sycl_target_, total_reaction_rate_, in_states_,
            std::array<int, FIXED_RATE_IONISATION::num_products_per_parent>{}, std::vector<ParticleProp<REAL>>{},
            std::vector<ParticleProp<INT>>{}, FixedRateIonisationData(rate_),
            IoniseReactionKernels<
                FIXED_RATE_IONISATION::num_products_per_parent>(
                FIXED_RATE_IONISATION::required_particle_real_props_obj,
                FIXED_RATE_IONISATION::required_species_field_real_props_obj)) {}
};