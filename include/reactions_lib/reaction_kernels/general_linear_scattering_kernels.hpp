#ifndef REACTIONS_LINEAR_SCATTERING_KERNELS_H
#define REACTIONS_LINEAR_SCATTERING_KERNELS_H
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <array>

namespace VANTAGE::Reactions {

/**
 * @brief Device type for general linear scattering kernels
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 * @tparam with_sources If true will attempt to write to source properties
 * (defaults to true)
 * @tparam with_weight_feedback If true will subtract reacted weight from the
 * parent particle (defaults to true)
 */
template <int ndim_velocity, bool with_sources, bool with_weight_feedback>
struct LinearScatteringKernelsOnDevice : public ReactionKernelsBaseOnDevice<1> {
  LinearScatteringKernelsOnDevice() = default;
  LinearScatteringKernelsOnDevice(REAL mass) : mass(mass) {}

  /**
   * @brief General scattering kernel - assumes that pre_req_data stores
   * post-scattering velocities
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
   * @brief Linear scattering weight kernel - simply sets the product's weight
   * to the weight change due to the reaction
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
   * @brief Linear scattering transformation kernel - simply sets the product's
   * ID the target ID
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
   * @brief Linear scattering feedback kernel for calculating and applying
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

    if constexpr (with_sources) {
      std::array<REAL, ndim_velocity> k_V_pre, k_V_post;
      REAL delta_vsquared = 0.0;

      for (int vdim = 0; vdim < ndim_velocity; vdim++) {
        k_V_pre[vdim] = accessors.req_real_props.at(this->velocity_ind,
                                                    accessors.index, vdim);
        k_V_post[vdim] = accessors.pre_req_data.at(
            accessors.index.get_loop_linear_index(), vdim);
        delta_vsquared +=
            k_V_pre[vdim] * k_V_pre[vdim] - k_V_post[vdim] * k_V_post[vdim];
      }

      // SOURCE_MOMENTUM calc
      for (int sm_dim = 0; sm_dim < ndim_velocity; sm_dim++) {
        accessors.req_real_props.at(this->source_momentum_ind, accessors.index,
                                    sm_dim) +=
            this->mass * modified_weight * (k_V_pre[sm_dim] - k_V_post[sm_dim]);
      }

      // Set SOURCE_ENERGY
      accessors.req_real_props.at(this->source_energy_ind, accessors.index,
                                  0) +=
          modified_weight * this->mass * delta_vsquared * 0.5;
    }

    if constexpr (with_weight_feedback) {
      accessors.req_real_props.at(this->weight_ind, accessors.index, 0) -=
          modified_weight;
    }
  }

public:
  INT velocity_ind, source_momentum_ind, source_energy_ind, weight_ind;
  INT descendant_internal_state_ind, descendant_velocity_ind,
      descendant_weight_ind;
  REAL mass;
};

/**
 * @brief Host type for linear scattering kernels - general kernels with
 * post-collision velocities defined by data calculator outputs
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 * @tparam with_sources If true will track sources (defaults to true)
 * @tparam with_weight_feedback If true will subtract reacted weight from the
 * parent particle (defaults to true)
 */
template <int ndim_velocity = 2, bool with_sources = true,
          bool with_weight_feedback = true>
struct LinearScatteringKernels
    : public ReactionKernelsBase<LinearScatteringKernelsOnDevice<
          ndim_velocity, with_sources, with_weight_feedback>> {
  using Base = ReactionKernelsBase<LinearScatteringKernelsOnDevice<
      ndim_velocity, with_sources, with_weight_feedback>>;

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props_with_sources =
      {props.source_momentum, props.source_energy};

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.weight, props.velocity};
  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};
  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};
  /**
   * @brief Constructor for LinearScatteringKernels.
   *
   * @param scattered_species Species object corresponding to the scattered
   * particle
   * @param properties_map (Optional) A PropertiesMap object to be
   * used when remapping property names.
   */
  LinearScatteringKernels(const Species &scattered_species,
                          const PropertiesMap properties_map = PropertiesMap())
      : Base((with_sources) ? Properties<REAL>(required_simple_real_props)
                                  .merge_with(Properties<REAL>(
                                      required_simple_real_props_with_sources))
                            : Properties<REAL>(required_simple_real_props),
             Properties<INT>(required_descendant_simple_int_props),
             Properties<REAL>(required_descendant_simple_real_props),
             ndim_velocity, ndim_velocity, properties_map) {
    this->on_device_obj =
        LinearScatteringKernelsOnDevice<ndim_velocity, with_sources,
                                        with_weight_feedback>(
            scattered_species.get_mass());
    this->index_on_device_object();
  }

  /** @brief Index inputs and descendants from the argument pack. */
  void index_on_device_object() override {
    this->on_device_obj->velocity_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    if constexpr (with_sources) {
      this->on_device_obj->source_momentum_ind =
          this->argument_pack.required_real_props.find_index(
              this->get_properties_map().at(props.source_momentum));
      this->on_device_obj->source_energy_ind =
          this->argument_pack.required_real_props.find_index(
              this->get_properties_map().at(props.source_energy));
    }
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
};

// Extern template declarations, so consumers do not re-instantiate what the
// library already provides.

extern template class LinearScatteringKernels<2, true>;

extern template class LinearScatteringKernels<2, false>;

extern template class LinearScatteringKernels<3, true>;

extern template class LinearScatteringKernels<3, false>;

}; // namespace VANTAGE::Reactions
#endif
