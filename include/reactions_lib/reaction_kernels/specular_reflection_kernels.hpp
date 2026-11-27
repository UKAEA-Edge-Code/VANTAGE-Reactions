#ifndef REACTIONS_SPECULAR_REFLECTION_KERNELS_H
#define REACTIONS_SPECULAR_REFLECTION_KERNELS_H
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include "../utils.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <array>

namespace VANTAGE::Reactions {

/**
 * @brief Device type for specular reflection surface process.
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 */
template <int ndim_velocity>
struct SpecularReflectionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<0> {
  SpecularReflectionKernelsOnDevice() = default;

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
    std::array<REAL, ndim_velocity> surface_n;
    REAL proj_factor = 0.0;

    // Calculate 2 * v_in dot n
    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = accessors.req_real_props.at(this->velocity_ind,
                                              accessors.index, vdim);
      surface_n[vdim] =
          accessors.req_real_props.at(this->normal_ind, accessors.index, vdim);
    }

    std::array<REAL, ndim_velocity> reflected =
        utils::reflect_vector(k_V, surface_n);
    // reflect across surface normal
    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      accessors.req_real_props.at(this->velocity_ind, accessors.index, vdim) =
          reflected[vdim];
    }
    accessors.req_real_props.at(this->weight_ind, accessors.index, 0) =
        modified_weight;
  }

public:
  INT velocity_ind, normal_ind, weight_ind;
};

/**
 * @brief Host type for simple specular reflection kernels, without any surface
 * feedback
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 */
template <int ndim_velocity = 2>
struct SpecularReflectionKernels
    : public ReactionKernelsBase<
          SpecularReflectionKernelsOnDevice<ndim_velocity>> {
  using Base =
      ReactionKernelsBase<SpecularReflectionKernelsOnDevice<ndim_velocity>>;

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 3> required_simple_real_props = {
      props.weight, props.velocity, props.boundary_intersection_normal};

  /**
   * @brief Specular reflection host type constructor
   *
   * @param properties_map A PropertiesMap object to be to be
   * used by the argument pack and device indexing to remap property names.
   */
  SpecularReflectionKernels(
      const PropertiesMap properties_map = PropertiesMap())
      : Base(Properties<REAL>(required_simple_real_props), 0, properties_map) {
    this->on_device_obj = SpecularReflectionKernelsOnDevice<ndim_velocity>();
    this->index_on_device_object();
  }

  /** @brief Index inputs from the argument pack. */
  void index_on_device_object() override {
    this->on_device_obj->velocity_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    this->on_device_obj->weight_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.weight));
    this->on_device_obj->normal_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.boundary_intersection_normal));
  }
};

// Extern template declarations, so consumers do not re-instantiate what the
// library already provides.

extern template class SpecularReflectionKernels<2>;

extern template class SpecularReflectionKernels<3>;

}; // namespace VANTAGE::Reactions
#endif
