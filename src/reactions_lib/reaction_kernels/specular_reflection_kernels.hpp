#pragma once
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include "../utils.hpp"
#include <array>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * struct SpecularReflectionKernelsOnDevice - SYCL device-compatible kernel for
 * specular reflection surface process.
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
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which descendant_product_loop is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param descendant_products Write accessor to descendant products
   * that may need to operated on
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for operations inside the kernel.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for operations inside the kernel.
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued NDLocalArray containing pre-calculated data
   * @param dt The current time step size.
   */
  void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::DescendantProducts::Write &descendant_products,
                       Access::SymVector::Write<INT> &req_int_props,
                       Access::SymVector::Write<REAL> &req_real_props,
                       const std::array<int, 0> &out_states,
                       Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                       double dt) const {
    std::array<REAL, ndim_velocity> k_V;
    std::array<REAL, ndim_velocity> surface_n;
    REAL proj_factor = 0.0;

    // Calculate 2 * v_in dot n
    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = req_real_props.at(velocity_ind, index, vdim);
      surface_n[vdim] = req_real_props.at(normal_ind, index, vdim);
    }

    std::array<REAL, ndim_velocity> reflected =
        utils::reflect_vector(k_V, surface_n);
    // reflect across surface normal
    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      req_real_props.at(velocity_ind, index, vdim) = reflected[vdim];
    }
    req_real_props.at(this->weight_ind, index, 0) = modified_weight;
  }

public:
  INT velocity_ind, normal_ind, weight_ind;
};

/**
 * @brief Simple specular reflection kernels, without any surface feedback
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 */
template <int ndim_velocity = 2>
struct SpecularReflectionKernels : public ReactionKernelsBase {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 3> required_simple_real_props = {
      props.weight, props.velocity, props.boundary_intersection_normal};

  /**
   * @brief Specular reflection host type constructor
   *
   * @param properties_map A std::map<int, std::string> object to be to be
   * passed to ReactionKernelsBase, used in remapping property names.
   */
  SpecularReflectionKernels(
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(Properties<INT>(),
                            Properties<REAL>(required_simple_real_props),
                            Properties<INT>(), Properties<REAL>()) {

    this->specular_reflection_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity,
                                                    this->properties_map);

    this->specular_reflection_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);

    this->specular_reflection_kernels_on_device.normal_ind =
        this->required_real_props.simple_prop_index(
            props.boundary_intersection_normal);
  };

private:
  SpecularReflectionKernelsOnDevice<ndim_velocity>
      specular_reflection_kernels_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */

  auto get_on_device_obj() {
    return this->specular_reflection_kernels_on_device;
  }
};
}; // namespace VANTAGE::Reactions
