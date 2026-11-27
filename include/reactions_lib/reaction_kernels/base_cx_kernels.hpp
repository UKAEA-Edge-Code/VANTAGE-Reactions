#ifndef REACTIONS_BASE_CX_KERNELS_H
#define REACTIONS_BASE_CX_KERNELS_H
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <array>

#include <vector>

namespace VANTAGE::Reactions {

/**
 * @brief Device type for charge-exchange kernels
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 * @tparam ndim_source_momentum The number of dimensions for
 * source momentum property.
 */
template <int ndim_velocity, int ndim_source_momentum>
struct CXReactionKernelsOnDevice : public ReactionKernelsBaseOnDevice<1> {
  CXReactionKernelsOnDevice() = default;
  CXReactionKernelsOnDevice(REAL target_mass, REAL projectile_mass)
      : target_mass(target_mass), projectile_mass(projectile_mass) {}

  /**
   * @brief CX scattering kernel - assumes that pre_req_data stores ion
   * velcocities sampled from the ion distribution and sets the product's
   * velocity components to those values
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param accessors Bundled loop index, input/descendant properties, and
   * prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles
   * @param dt The current time step size.
   */
  void scattering_kernel(REAL &modified_weight,
                         SingleReactionKernelAccessors &accessors,
                         const std::array<int, 1> &out_states,
                         double dt) const {
    for (int dimx = 0; dimx < ndim_velocity; dimx++) {
      accessors.descendant_products.at_real(
          accessors.index, 0, this->descendant_velocity_ind, dimx) =
          accessors.pre_req_data.at(accessors.index.get_loop_linear_index(),
                                    dimx);
    }
  }

  /**
   * @brief CX weight kernel - simply sets the product's weight to the weight
   * change due to the reaction
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param accessors Bundled loop index, input/descendant properties, and
   * prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles
   * @param dt The current time step size.
   */
  void weight_kernel(REAL &modified_weight,
                     SingleReactionKernelAccessors &accessors,
                     const std::array<int, 1> &out_states, double dt) const {
    accessors.descendant_products.at_real(
        accessors.index, 0, this->descendant_weight_ind, 0) = modified_weight;
  }

  /**
   * @brief CX transformation kernel - simply sets the product's ID the target
   * ID
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param accessors Bundled loop index, input/descendant properties, and
   * prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles
   * @param dt The current time step size.
   */
  void transformation_kernel(REAL &modified_weight,
                             SingleReactionKernelAccessors &accessors,
                             const std::array<int, 1> &out_states,
                             double dt) const {
    accessors.descendant_products.at_int(accessors.index, 0,
                                         this->descendant_internal_state_ind,
                                         0) = out_states[0];
  }

  /**
   * @brief CX feedback kernel for calculating and applying
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
                       const std::array<int, 1> &out_states, double dt) const {

    std::array<REAL, ndim_velocity> k_V, k_Vi;
    REAL vsquared = 0.0;
    REAL visquared = 0.0;

    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = accessors.req_real_props.at(this->velocity_ind,
                                              accessors.index, vdim);
      k_Vi[vdim] = accessors.pre_req_data.at(
          accessors.index.get_loop_linear_index(), vdim);
      vsquared += k_V[vdim] * k_V[vdim];
      visquared += k_Vi[vdim] * k_Vi[vdim];
    }

    // Set SOURCE_DENSITY
    accessors.req_real_props.at(this->projectile_source_density_ind,
                                accessors.index, 0) += modified_weight;

    accessors.req_real_props.at(this->target_source_density_ind,
                                accessors.index, 0) -= modified_weight;

    // SOURCE_MOMENTUM calc
    for (int sm_dim = 0; sm_dim < ndim_source_momentum; sm_dim++) {
      accessors.req_real_props.at(this->target_source_momentum_ind,
                                  accessors.index, sm_dim) -=
          this->target_mass * modified_weight * k_Vi[sm_dim];
      accessors.req_real_props.at(this->projectile_source_momentum_ind,
                                  accessors.index, sm_dim) +=
          this->projectile_mass * modified_weight * k_V[sm_dim];
    }

    // Set SOURCE_ENERGY
    accessors.req_real_props.at(this->target_source_energy_ind, accessors.index,
                                0) -=
        modified_weight * this->target_mass * visquared * 0.5;

    accessors.req_real_props.at(this->projectile_source_energy_ind,
                                accessors.index, 0) +=
        modified_weight * this->projectile_mass * vsquared * 0.5;

    accessors.req_real_props.at(this->weight_ind, accessors.index, 0) -=
        modified_weight;
  }

public:
  INT velocity_ind, projectile_source_density_ind, projectile_source_energy_ind,
      projectile_source_momentum_ind, target_source_density_ind,
      target_source_momentum_ind, target_source_energy_ind, weight_ind;
  INT descendant_internal_state_ind, descendant_velocity_ind,
      descendant_weight_ind;
  REAL target_mass, projectile_mass;
};

/**
 * @brief Host type for charge-exchange kernels
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 * @tparam ndim_source_momentum Optional number of dimensions for
 * source momentum property (default value of ndim_velocity)
 */
template <int ndim_velocity = 2, int ndim_source_momentum = ndim_velocity>
struct CXReactionKernels
    : public ReactionKernelsBase<
          CXReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum>> {
  using Base = ReactionKernelsBase<
      CXReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum>>;

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.weight, props.velocity};

  constexpr static std::array<int, 3> required_species_real_props = {
      props.source_density, props.source_energy,
      props.source_momentum}; // namespace BASE_CX_KERNEL

  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};
  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};
  /**
   * @brief Constructor for CXReactionKernels.
   *
   * @param target_species Species object representing the charge exchange
   * target - the ingoing ion and outgoing neutral
   * @param projectile_species Species object representing the projectile
   * species - the outgoing ion and ingoing neutral
   * @param properties_map (Optional) A PropertiesMap object to be
   * used when remapping property names.
   */
  CXReactionKernels(const Species &target_species,
                    const Species &projectile_species,
                    const PropertiesMap properties_map = PropertiesMap())
      : Base(Properties<REAL>(
                 required_simple_real_props,
                 std::vector<Species>{target_species, projectile_species},
                 required_species_real_props),
             Properties<INT>(required_descendant_simple_int_props),
             Properties<REAL>(required_descendant_simple_real_props),
             ndim_velocity, ndim_velocity, properties_map),
        target_species(target_species), projectile_species(projectile_species) {
    static_assert((ndim_velocity >= ndim_source_momentum),
                  "Number of dimension for VELOCITY must be greater than or "
                  "equal to number of dimensions for SOURCE_MOMENTUM.");
    this->on_device_obj =
        CXReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum>(
            target_species.get_mass(), projectile_species.get_mass());
    this->index_on_device_object();
  }

  /** @brief Index inputs and descendants from the argument pack. */
  void index_on_device_object() override {
    this->on_device_obj->velocity_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.velocity));
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
    this->on_device_obj->projectile_source_density_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->projectile_species,
            this->get_properties_map().at(props.source_density)));
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
    this->on_device_obj->descendant_internal_state_ind =
        this->argument_pack.required_descendant_int_props.find_index(
            this->get_properties_map().at(props.internal_state));
    this->on_device_obj->descendant_velocity_ind =
        this->argument_pack.required_descendant_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    this->on_device_obj->descendant_weight_ind =
        this->argument_pack.required_descendant_real_props.find_index(
            this->get_properties_map().at(props.weight));
  }

private:
  Species target_species, projectile_species;
};

// Extern template declarations, so consumers do not re-instantiate what the
// library already provides.

extern template class CXReactionKernels<2>;

extern template class CXReactionKernels<3>;

}; // namespace VANTAGE::Reactions
#endif