#ifndef REACTIONS_PAIR_SCATTERING_KERNELS_H
#define REACTIONS_PAIR_SCATTERING_KERNELS_H
#include "../pair_reaction_kernels.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <array>

namespace VANTAGE::Reactions {

/**
 * @brief Device type for general pair scattering kernels
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 */
template <int ndim_velocity>
struct PairScatteringKernelsOnDevice
    : public PairReactionKernelsBaseOnDevice<2> {

  PairScatteringKernelsOnDevice() = default;

  /**
   * @brief General pair scattering parent kernel. Sets the first product's
   * parent to the A particle and the second to the B particle.
   *
   * @param accessors Bundled pair-loop indices, input/descendant properties,
   * and prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles.
   */
  void parent_kernel(PairReactionKernelAccessors &accessors,
                     const std::array<int, 2> &out_states) const {
    accessors.descendant_products_a.set_parent(accessors.index_a, 0);
    accessors.descendant_products_b.set_parent(accessors.index_b, 0);
  }

  /**
   * @brief General pair scattering kernel - assumes that pre_req_data stores
   * the 2*ndim_velocity components for the two scattering particles.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param accessors Bundled pair-loop indices, input/descendant properties,
   * and prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles.
   * @param dt The current time step size.
   */
  void scattering_kernel(REAL &modified_weight,
                         PairReactionKernelAccessors &accessors,
                         const std::array<int, 2> &out_states,
                         double dt) const {
    const INT pair_row = accessors.pair_index.get_loop_linear_index();
    for (int dimx = 0; dimx < ndim_velocity; dimx++) {
      accessors.descendant_products_a.at_real(
          accessors.index_a, 0, this->descendant_velocity_ind_a, dimx) =
          accessors.pre_req_data.at(pair_row, dimx);
      accessors.descendant_products_b.at_real(
          accessors.index_b, 0, this->descendant_velocity_ind_b, dimx) =
          accessors.pre_req_data.at(pair_row, ndim_velocity + dimx);
    }
  }

  /**
   * @brief General pair scattering weight kernel - simply sets the product's
   * weight to the weight change due to the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param accessors Bundled pair-loop indices, input/descendant properties,
   * and prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles.
   * @param dt The current time step size.
   */
  void weight_kernel(REAL &modified_weight,
                     PairReactionKernelAccessors &accessors,
                     const std::array<int, 2> &out_states, double dt) const {
    accessors.descendant_products_a.at_real(accessors.index_a, 0,
                                            this->descendant_weight_ind_a, 0) =
        modified_weight;
    accessors.descendant_products_b.at_real(accessors.index_b, 0,
                                            this->descendant_weight_ind_b, 0) =
        modified_weight;
  }

  /**
   * @brief General pair scattering transformation kernel.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param accessors Bundled pair-loop indices, input/descendant properties,
   * and prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles.
   * @param dt The current time step size.
   */
  void transformation_kernel(REAL &modified_weight,
                             PairReactionKernelAccessors &accessors,
                             const std::array<int, 2> &out_states,
                             double dt) const {
    accessors.descendant_products_a.at_int(
        accessors.index_a, 0, this->descendant_internal_state_ind_a, 0) =
        out_states[0];
    accessors.descendant_products_b.at_int(
        accessors.index_b, 0, this->descendant_internal_state_ind_b, 0) =
        out_states[1];
  }

  /**
   * @brief General pair scattering feedback kernel for calculating and applying
   * background field modifications from the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param accessors Bundled pair-loop indices, input/descendant properties,
   * and prerequisite data accessors.
   * @param out_states Array defining the IDs of descendant particles.
   * @param dt The current time step size.
   */
  void feedback_kernel(REAL &modified_weight,
                       PairReactionKernelAccessors &accessors,
                       const std::array<int, 2> &out_states, double dt) const {
    accessors.req_real_props_a.at(this->weight_ind_a, 0) -= modified_weight;
    accessors.req_real_props_b.at(this->weight_ind_b, 0) -= modified_weight;
  }

public:
  INT weight_ind_a, weight_ind_b;
  INT descendant_internal_state_ind_a, descendant_velocity_ind_a,
      descendant_weight_ind_a;
  INT descendant_internal_state_ind_b, descendant_velocity_ind_b,
      descendant_weight_ind_b;
};

/**
 * @brief Host type for general pair scattering kernels - general kernels with
 * post-collision velocities defined by data calculator outputs.
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2).
 */
template <int ndim_velocity = 2>
struct PairScatteringKernels
    : public PairReactionKernelsBase<
          PairScatteringKernelsOnDevice<ndim_velocity>> {
  using Base =
      PairReactionKernelsBase<PairScatteringKernelsOnDevice<ndim_velocity>>;

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};
  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};
  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};

  /**
   * @brief Constructor for PairScatteringKernels.
   *
   * @param properties_map (Optional) A PropertiesMap object to be
   * used when remapping property names.
   */
  PairScatteringKernels(const PropertiesMap properties_map = PropertiesMap())
      : Base(Properties<REAL>(required_simple_real_props),
             Properties<INT>(required_descendant_simple_int_props),
             Properties<REAL>(required_descendant_simple_real_props),
             2 * ndim_velocity, ndim_velocity, 1, 1, properties_map) {
    this->on_device_obj = PairScatteringKernelsOnDevice<ndim_velocity>();
    this->index_on_device_object();
  }

  /** @brief Index A/B inputs and descendants from the argument pack. */
  void index_on_device_object() override {
    const auto &weight_name = this->get_properties_map().at(props.weight);
    const auto &internal_state_name =
        this->get_properties_map().at(props.internal_state);
    const auto &velocity_name = this->get_properties_map().at(props.velocity);

    this->on_device_obj->weight_ind_a =
        this->argument_pack.required_real_props_a.find_index(weight_name);
    this->on_device_obj->weight_ind_b =
        this->argument_pack.required_real_props_b.find_index(weight_name);

    this->on_device_obj->descendant_internal_state_ind_a =
        this->argument_pack.required_descendant_int_props_a.find_index(
            internal_state_name);
    this->on_device_obj->descendant_velocity_ind_a =
        this->argument_pack.required_descendant_real_props_a.find_index(
            velocity_name);
    this->on_device_obj->descendant_weight_ind_a =
        this->argument_pack.required_descendant_real_props_a.find_index(
            weight_name);

    this->on_device_obj->descendant_internal_state_ind_b =
        this->argument_pack.required_descendant_int_props_b.find_index(
            internal_state_name);
    this->on_device_obj->descendant_velocity_ind_b =
        this->argument_pack.required_descendant_real_props_b.find_index(
            velocity_name);
    this->on_device_obj->descendant_weight_ind_b =
        this->argument_pack.required_descendant_real_props_b.find_index(
            weight_name);
  }
};

}; // namespace VANTAGE::Reactions
#endif
