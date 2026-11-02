#ifndef REACTIONS_ABSORPTION_KERNELS_H
#define REACTIONS_ABSORPTION_KERNELS_H
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <array>

namespace VANTAGE::Reactions {

/**
 * @brief Device type for general absorption kernels
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 */
template <int ndim_velocity>
struct GeneralAbsorptionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<0> {
  GeneralAbsorptionKernelsOnDevice() = default;
  GeneralAbsorptionKernelsOnDevice(REAL mass) : mass(mass) {}

  /**
   * @brief General absorption feedback kernel for calculating and applying
   * background field modifications from the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param accessors Bundled loop index, input/descendant properties, and
   * prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles
   * @param dt The current time step size.
   */
  void feedback_kernel(REAL &modified_weight,
                       SingleReactionKernelAccessors &accessors,
                       const std::array<int, 0> &out_states, double dt) const {

    std::array<REAL, ndim_velocity> k_V_pre;
    REAL delta_vsquared = 0.0;

    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V_pre[vdim] = accessors.req_real_props.at(this->velocity_ind,
                                                  accessors.index, vdim);
      delta_vsquared += k_V_pre[vdim] * k_V_pre[vdim];
    }

    // SOURCE_DENSITY calc
    accessors.req_real_props.at(this->source_density_ind, accessors.index, 0) +=
        modified_weight;

    // SOURCE_MOMENTUM calc
    for (int sm_dim = 0; sm_dim < ndim_velocity; sm_dim++) {
      accessors.req_real_props.at(this->source_momentum_ind, accessors.index,
                                  sm_dim) +=
          this->mass * modified_weight * k_V_pre[sm_dim];
    }

    // Set SOURCE_ENERGY
    accessors.req_real_props.at(this->source_energy_ind, accessors.index, 0) +=
        modified_weight * this->mass * delta_vsquared * 0.5;

    accessors.req_real_props.at(this->weight_ind, accessors.index, 0) -=
        modified_weight;
  }

public:
  INT velocity_ind, source_density_ind, source_momentum_ind, source_energy_ind,
      weight_ind;
  REAL mass;
};

/**
 * @brief Host type for general absorption kernels
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 */
template <int ndim_velocity = 2>
struct GeneralAbsorptionKernels
    : public ReactionKernelsBase<
          GeneralAbsorptionKernelsOnDevice<ndim_velocity>> {
  using Base =
      ReactionKernelsBase<GeneralAbsorptionKernelsOnDevice<ndim_velocity>>;

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 5> required_simple_real_props = {
      props.weight, props.source_density, props.velocity, props.source_momentum,
      props.source_energy};

  /**
   * @brief Constructor for GeneralAbsorptionKernels.
   *
   * @param absorbed_species Species object corresponding to the absorbed
   * particle
   * @param properties_map (Optional) A PropertiesMap object to be
   * used when remapping property names.
   */
  GeneralAbsorptionKernels(const Species &absorbed_species,
                           const PropertiesMap properties_map = PropertiesMap())
      : Base(Properties<REAL>(required_simple_real_props), 0, properties_map) {
    this->on_device_obj = GeneralAbsorptionKernelsOnDevice<ndim_velocity>(
        absorbed_species.get_mass());
    this->index_on_device_object();
  }

  /** @brief Index inputs from the argument pack. */
  void index_on_device_object() override {
    this->on_device_obj->velocity_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    this->on_device_obj->source_density_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.source_density));
    this->on_device_obj->source_momentum_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.source_momentum));
    this->on_device_obj->source_energy_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.source_energy));
    this->on_device_obj->weight_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.weight));
  }
};

// Extern template declarations, so consumers do not re-instantiate what the
// library already provides.

extern template class GeneralAbsorptionKernels<2>;

extern template class GeneralAbsorptionKernels<3>;

}; // namespace VANTAGE::Reactions
#endif
