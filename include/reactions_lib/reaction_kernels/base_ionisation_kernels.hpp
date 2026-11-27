#ifndef REACTIONS_BASE_IONISATION_KERNELS_H
#define REACTIONS_BASE_IONISATION_KERNELS_H
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <array>

#include <vector>

namespace VANTAGE::Reactions {

/**
 * @brief Device type for ionisation kernels
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 * @tparam ndim_source_momentum The number of dimensions for electron
 * source momentum property.
 * @tparam has_momentum_req_data The boolean specifying whether a
 * projectile momentum req_data is available.
 */
template <int ndim_velocity, int ndim_source_momentum,
          bool has_momentum_req_data>
struct IoniseReactionKernelsOnDevice : public ReactionKernelsBaseOnDevice<0> {
  IoniseReactionKernelsOnDevice() = default;
  IoniseReactionKernelsOnDevice(REAL target_mass) : target_mass(target_mass) {}

  /**
   * @brief Feedback kernel for calculating and applying
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
    std::array<REAL, ndim_velocity> k_V;
    REAL vsquared = 0.0;

    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = accessors.req_real_props.at(this->velocity_ind,
                                              accessors.index, vdim);
      vsquared += k_V[vdim] * k_V[vdim];
    }

    // Set SOURCE_DENSITY
    accessors.req_real_props.at(this->electron_source_density_ind,
                                accessors.index, 0) += modified_weight;

    accessors.req_real_props.at(this->target_source_density_ind,
                                accessors.index, 0) += modified_weight;

    // SOURCE_MOMENTUM calc
    for (int sm_dim = 0; sm_dim < ndim_source_momentum; sm_dim++) {
      accessors.req_real_props.at(this->target_source_momentum_ind,
                                  accessors.index, sm_dim) +=
          this->target_mass * modified_weight * k_V[sm_dim];
    }

    // Treat momentum sources when a projectile momentum req_data is available
    if (has_momentum_req_data) {

      for (int sm_dim = 0; sm_dim < ndim_source_momentum; sm_dim++) {
        accessors.req_real_props.at(this->target_source_momentum_ind,
                                    accessors.index, sm_dim) +=
            accessors.pre_req_data.at(accessors.index.get_loop_linear_index(),
                                      1) *
            dt;
        accessors.req_real_props.at(this->projectile_source_momentum_ind,
                                    accessors.index, sm_dim) -=
            accessors.pre_req_data.at(accessors.index.get_loop_linear_index(),
                                      1) *
            dt;
      }
    }

    // Set SOURCE_ENERGY
    accessors.req_real_props.at(this->target_source_energy_ind, accessors.index,
                                0) +=
        modified_weight * this->target_mass * vsquared * 0.5;

    accessors.req_real_props.at(this->projectile_source_energy_ind,
                                accessors.index, 0) -=
        accessors.pre_req_data.at(accessors.index.get_loop_linear_index(), 0) *
        dt;

    accessors.req_real_props.at(this->weight_ind, accessors.index, 0) -=
        modified_weight;
  }

public:
  INT velocity_ind, electron_source_density_ind, projectile_source_energy_ind,
      projectile_source_momentum_ind, target_source_density_ind,
      target_source_momentum_ind, target_source_energy_ind, weight_ind;
  REAL target_mass;
};

/**
 * @brief Host type for ionisation kernels
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 * @tparam ndim_source_momentum Optional number of dimensions for electron
 * source momentum property (default value of ndim_veloctiy)
 * @tparam has_momentum_req_data Optional boolean specifying whether a
 * projectile momentum req_data is available (default value of false)
 */
template <int ndim_velocity = 2, int ndim_source_momentum = ndim_velocity,
          bool has_momentum_req_data = false>
struct IoniseReactionKernels
    : public ReactionKernelsBase<IoniseReactionKernelsOnDevice<
          ndim_velocity, ndim_source_momentum, has_momentum_req_data>> {
  using Base = ReactionKernelsBase<IoniseReactionKernelsOnDevice<
      ndim_velocity, ndim_source_momentum, has_momentum_req_data>>;

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.weight, props.velocity};

  constexpr static std::array<int, 3> required_species_real_props = {
      props.source_density, props.source_energy, props.source_momentum};
  /**
   * @brief Constructor for IonisationReactionKernels.
   *
   * @param target_species Species object representing the ionisation target
   * (and the corresponding ion field!)
   * @param electron_species Species object representing the electrons
   * @param projectile_species Species object representing the projectile
   * species
   * @param properties_map (Optional) A PropertiesMap object to be
   * used when remapping property names.
   */
  IoniseReactionKernels(const Species &target_species,
                        const Species &electron_species,
                        const Species &projectile_species,
                        const PropertiesMap properties_map = PropertiesMap())
      : Base(Properties<REAL>(required_simple_real_props,
                              std::vector<Species>{target_species,
                                                   electron_species,
                                                   projectile_species},
                              required_species_real_props),
             has_momentum_req_data ? 2 : 1, properties_map),
        target_species(target_species), electron_species(electron_species),
        projectile_species(projectile_species) {
    static_assert(
        (ndim_velocity >= ndim_source_momentum),
        "Number of dimension for VELOCITY must be greater than or "
        "equal to number of dimensions for ELECTRON_SOURCE_MOMENTUM.");
    this->on_device_obj =
        IoniseReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum,
                                      has_momentum_req_data>(
            target_species.get_mass());
    this->index_on_device_object();
  }

  /** @brief Index inputs from the argument pack. */
  void index_on_device_object() override {
    this->on_device_obj->velocity_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    this->on_device_obj->electron_source_density_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->electron_species,
            this->get_properties_map().at(props.source_density)));
    this->on_device_obj->target_source_density_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->target_species,
            this->get_properties_map().at(props.source_density)));
    this->on_device_obj->target_source_momentum_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->target_species,
            this->get_properties_map().at(props.source_momentum)));
    this->on_device_obj->target_source_energy_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->target_species,
            this->get_properties_map().at(props.source_energy)));
    this->on_device_obj->projectile_source_momentum_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->projectile_species,
            this->get_properties_map().at(props.source_momentum)));
    this->on_device_obj->projectile_source_energy_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->projectile_species,
            this->get_properties_map().at(props.source_energy)));
    this->on_device_obj->weight_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.weight));
  }

private:
  Species target_species, electron_species, projectile_species;
};

// Extern template declarations, so consumers do not re-instantiate what the
// library already provides.

extern template class IoniseReactionKernels<2>;

extern template class IoniseReactionKernels<3>;

}; // namespace VANTAGE::Reactions
#endif