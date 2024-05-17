#pragma once
#include "typedefs.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <particle_properties_map.hpp>
#include <reaction_kernel_pre_reqs.hpp>
#include <reaction_kernels.hpp>
#include <string_view>
#include <vector>

using namespace NESO::Particles;
using namespace ParticlePropertiesIndices;

template <INT num_products_per_parent>
struct IoniseReactionKernelsOnDevice
      : public ReactionKernelsBaseOnDevice<num_products_per_parent> {
    IoniseReactionKernelsOnDevice() = default;

    void
    feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                    Access::DescendantProducts::Write &descendant_products,
                    Access::SymVector::Write<INT> &req_particle_ints,
                    Access::SymVector::Write<REAL> &req_particle_reals,
                    Access::SymVector::Write<INT> &req_field_ints,
                    Access::SymVector::Write<REAL> &req_field_reals,
                    const std::array<int, num_products_per_parent> &out_states,
                    Access::LocalArray::Read<REAL> &pre_req_data,
                    double dt) const {
      // auto k_V_0 = write_req_reals.at(0, index, 0);
      // auto k_V_1 = write_req_reals.at(0, index, 1);
      // Instead of enums, perhaps strings that are mapped to indices
      // and then use aliases to refer to the mapped indices in a more readable
      // manner?
      auto k_V_0 = req_particle_reals.at(velocity_ind, index, 0);
      auto k_V_1 = req_particle_reals.at(velocity_ind, index, 1);

      const REAL vsquared = (k_V_0 * k_V_0) + (k_V_1 * k_V_1);

      REAL k_n_scale = 1.0; // / test_reaction_data.get_n_to_SI();
      REAL inv_k_dt = 1.0 / dt;

      auto nE = req_field_reals.at(electron_density_ind, index, 0);

      // Set SOURCE_DENSITY
      req_field_reals.at(electron_source_density_ind, index, 0) +=
          nE * modified_weight * k_n_scale * inv_k_dt;

      // Get SOURCE_DENSITY for SOURCE_MOMENTUM calc
      auto k_SD = req_field_reals.at(electron_source_density_ind, index, 0);
      req_field_reals.at(electron_source_momentum_ind, index, 0) +=
          k_SD * k_V_0;
      req_field_reals.at(electron_source_momentum_ind, index, 1) +=
          k_SD * k_V_1;

      // Set SOURCE_ENERGY
      req_field_reals.at(electron_source_energy_ind, index, 0) +=
          k_SD * vsquared * 0.5;

      req_particle_reals.at(weight_ind, index, 0) -= modified_weight;
    }
    
    int velocity_ind, electron_density_ind, electron_source_density_ind,
      electron_source_momentum_ind, electron_source_energy_ind, weight_ind;
  };


template <INT num_products_per_parent>

// TODO Species will be passed as typename parameter pack
// template <INT num_products_per_parent>
struct IoniseReactionKernels : public ReactionKernelsBase {

  IoniseReactionKernels(
      const RequiredParticleRealProperties<2> required_particle_real_props_,
      const RequiredSpeciesFieldProperties<1, 4> required_species_field_real_props_)
      : ReactionKernelsBase(),
        required_particle_real_props(required_particle_real_props_),
        required_species_field_real_props(required_species_field_real_props_) {
    auto species_names =
        this->required_species_field_real_props.get_species_names();

    this->ionise_reaction_kernels_on_device.velocity_ind =
        this->required_particle_real_props.required_particle_real_index(
            velocity);
    this->ionise_reaction_kernels_on_device.electron_density_ind =
        this->required_species_field_real_props
            .required_species_field_real_index(species_names[0], density);
    this->ionise_reaction_kernels_on_device.electron_source_density_ind =
        this->required_species_field_real_props
            .required_species_field_real_index(species_names[0],
                                               source_density);
    this->ionise_reaction_kernels_on_device.electron_source_momentum_ind =
        this->required_species_field_real_props
            .required_species_field_real_index(species_names[0],
                                               source_momentum);
    this->ionise_reaction_kernels_on_device.electron_source_energy_ind =
        this->required_species_field_real_props
            .required_species_field_real_index(species_names[0], source_energy);
    this->ionise_reaction_kernels_on_device.weight_ind =
        this->required_particle_real_props.required_particle_real_index(weight);
  };

private:
  IoniseReactionKernelsOnDevice<num_products_per_parent> ionise_reaction_kernels_on_device;

  RequiredParticleRealProperties<2> required_particle_real_props;
  RequiredSpeciesFieldProperties<1, 4> required_species_field_real_props;

public:
  const int get_num_particle_real_props() {
    return std::size(this->required_particle_real_props.get_req_props());
  }

  std::vector<std::string> get_required_particle_real_props() {
    auto result = this->required_particle_real_props.required_particle_real_prop_names();
    std::vector<std::string> result_vec(result.begin(), result.end());
    return result_vec;
  }

  const int get_num_field_real_props() {
    return std::size(this->required_species_field_real_props.get_req_props());
  }

  std::vector<std::string> get_required_field_real_props() {
    auto result = this->required_species_field_real_props.required_species_field_real_prop_names();
    std::vector<std::string> result_vec(result.begin(), result.end());
    return result_vec;
  }

  IoniseReactionKernelsOnDevice<num_products_per_parent> get_on_device_obj() {
    return this->ionise_reaction_kernels_on_device;
  }
};
