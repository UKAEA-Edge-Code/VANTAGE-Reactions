#pragma once
#include "particle_properties_map.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <ionisation_reactions_data/amjuel_ionisation_data.hpp>
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

namespace AMJUEL_IONISATION_DATA {
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
    RequiredSpeciesFieldProperties<std::size(species), required_field_real_props.size()>(species, required_field_real_props);

} // namespace AMJUEL_IONISATION_DATA

// using namespace AMJUEL_IONISATION_DATA;

template <int num_coeffs>
struct IoniseReactionAMJUEL
    : public LinearReactionBase<AMJUEL_IONISATION_DATA::num_products_per_parent,
                                IoniseReactionAMJUELData<num_coeffs>,
                                IoniseReactionKernels> {

  IoniseReactionAMJUEL() = default;

  IoniseReactionAMJUEL(SYCLTargetSharedPtr sycl_target_,
                       Sym<REAL> total_reaction_rate_, int in_states_,
                       REAL density_normalisation,
                       std::array<REAL, num_coeffs> coeffs)
      : LinearReactionBase<AMJUEL_IONISATION_DATA::num_products_per_parent,
                           IoniseReactionAMJUELData<num_coeffs>,
                           IoniseReactionKernels>(
            sycl_target_, total_reaction_rate_, in_states_,
            std::array<int, AMJUEL_IONISATION_DATA::num_products_per_parent>{},
            std::vector<ParticleProp<REAL>>{}, std::vector<ParticleProp<INT>>{},
            IoniseReactionAMJUELData<num_coeffs>(density_normalisation, coeffs),
            IoniseReactionKernels<AMJUEL_IONISATION_DATA::num_products_per_parent>(
                AMJUEL_IONISATION_DATA::required_particle_real_props_obj,
                AMJUEL_IONISATION_DATA::required_species_field_real_props_obj
            )) {}
};