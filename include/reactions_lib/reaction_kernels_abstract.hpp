#ifndef REACTIONS_REACTION_KERNELS_ABSTRACT_H
#define REACTIONS_REACTION_KERNELS_ABSTRACT_H

#include "../reactions/neso_test_assert.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <array>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace VANTAGE::Reactions {

/**
 * @brief Build a fresh product schema in canonical name order.
 *
 * The real property whose resolved name matches velocity uses ndim_velocity;
 * every other property has one component. Empty descriptions produce an
 * initialized empty schema.
 *
 * @param required_descendant_int_props Explicitly written integer descendant
 * property names.
 * @param required_descendant_real_props Explicitly written real descendant
 * property names.
 * @param ndim_velocity Velocity component count; must be positive if velocity
 * is present.
 * @param properties_map Map used to resolve the velocity name, matching the map
 * used to construct the descendant sets.
 * @param num_products Number of products for this channel; nonpositive values
 * return an empty schema without warning.
 * @return A new schema whose per-type field indices match the name sets.
 */
std::shared_ptr<NP::ProductMatrixSpec> build_kernel_product_schema(
    const ArgumentNameSet<INT> &required_descendant_int_props,
    const ArgumentNameSet<REAL> &required_descendant_real_props,
    int ndim_velocity = 2,
    const PropertiesMap &properties_map = PropertiesMap(),
    INT num_products = 1);

/**
 * @brief Optional no-op operations for bundled-accessor device kernels.
 * @tparam ACCESSOR_PACK_T Device accessor pack type.
 * @tparam num_products Fixed output-state extent.
 */
template <typename ACCESSOR_PACK_T, int num_products>
struct AbstractReactionKernelsOnDevice {
  static_assert(num_products >= 0, "Kernel product count must be nonnegative");
  static constexpr int NUM_PRODUCTS = num_products;

  AbstractReactionKernelsOnDevice() = default;

  /**
   * @brief Apply scattering; inherited implementation does nothing.
   * @param modified_weight Shared mutable reacted weight.
   * @param accessors Bundled loop accessors.
   * @param out_states Explicit descendant species IDs.
   * @param dt Current effective timestep.
   */
  void scattering_kernel(REAL &modified_weight, ACCESSOR_PACK_T &accessors,
                         const std::array<int, num_products> &out_states,
                         double dt) const {}
  /** @copydoc scattering_kernel */
  void weight_kernel(REAL &modified_weight, ACCESSOR_PACK_T &accessors,
                     const std::array<int, num_products> &out_states,
                     double dt) const {}
  /** @copydoc scattering_kernel */
  void transformation_kernel(REAL &modified_weight, ACCESSOR_PACK_T &accessors,
                             const std::array<int, num_products> &out_states,
                             double dt) const {}
  /** @copydoc scattering_kernel */
  void feedback_kernel(REAL &modified_weight, ACCESSOR_PACK_T &accessors,
                       const std::array<int, num_products> &out_states,
                       double dt) const {}
};

namespace detail {
template <typename DEVICE, typename ACCESSORS, int N>
using kernel_operation_signature = void (DEVICE::*)(REAL &, ACCESSORS &,
                                                    const std::array<int, N> &,
                                                    double) const;

template <typename DEVICE, typename ACCESSORS, typename = void>
struct has_kernel_operations_impl : std::false_type {};

template <typename DEVICE, typename ACCESSORS>
struct has_kernel_operations_impl<
    DEVICE, ACCESSORS,
    std::void_t<decltype(static_cast<kernel_operation_signature<
                             DEVICE, ACCESSORS, DEVICE::NUM_PRODUCTS>>(
                    &DEVICE::scattering_kernel)),
                decltype(static_cast<kernel_operation_signature<
                             DEVICE, ACCESSORS, DEVICE::NUM_PRODUCTS>>(
                    &DEVICE::weight_kernel)),
                decltype(static_cast<kernel_operation_signature<
                             DEVICE, ACCESSORS, DEVICE::NUM_PRODUCTS>>(
                    &DEVICE::transformation_kernel)),
                decltype(static_cast<kernel_operation_signature<
                             DEVICE, ACCESSORS, DEVICE::NUM_PRODUCTS>>(
                    &DEVICE::feedback_kernel))>> : std::true_type {};

template <typename DEVICE, typename ACCESSORS, typename = void>
struct has_kernel_operations : std::false_type {};

template <typename DEVICE, typename ACCESSORS>
struct has_kernel_operations<DEVICE, ACCESSORS,
                             std::enable_if_t<(DEVICE::NUM_PRODUCTS >= 0)>>
    : has_kernel_operations_impl<DEVICE, ACCESSORS> {};

void validate_kernel_pre_req_ndims(INT pre_req_ndims);
} // namespace detail

/**
 * @brief Base host class for reaction kernels.
 * @tparam ON_DEVICE_T Device type with bundled operations and NUM_PRODUCTS.
 * @tparam ARGUMENT_PACK_T Host argument-pack type.
 *
 * Derived constructors initialize on_device_obj, including non-index values,
 * then call their index_on_device_object() hook.
 */
template <typename ON_DEVICE_T, typename ARGUMENT_PACK_T>
struct AbstractReactionKernels {
  using ON_DEVICE_OBJ_TYPE = ON_DEVICE_T;
  using ARGUMENT_PACK_TYPE = ARGUMENT_PACK_T;
  static constexpr int NUM_PRODUCTS = ON_DEVICE_T::NUM_PRODUCTS;
  static_assert(NUM_PRODUCTS >= 0, "Kernel product count must be nonnegative");
  virtual ~AbstractReactionKernels() = default;

  /** @brief Get the argument pack. @return Read-only argument pack reference.
   */
  const ARGUMENT_PACK_T &get_arg_pack() const { return this->argument_pack; }

  /** @brief Get configured names. @return Read-only property map. */
  const PropertiesMap &get_properties_map() const {
    return this->properties_map;
  }

  /** @brief Get prerequisite size. @return Required component count. */
  INT get_pre_ndims() const { return this->pre_req_ndims; }

  /**
   * @brief Return the initialized device object.
   * @return Read-only device object owned by this kernel.
   */
  const ON_DEVICE_T &get_on_device_obj() const {
    NESOASSERT(this->on_device_obj.has_value(),
               "on_device_obj in AbstractReactionKernels not initialised");
    return this->on_device_obj.value();
  }

  /**
   * @brief Index required properties on the on-device object.
   */
  virtual void index_on_device_object() {}

protected:
  /**
   * @brief Store complete configuration without invoking the indexing hook.
   * @param argument_pack Authoritative kernel names.
   * @param pre_req_ndims Prerequisite-data component count, nonnegative.
   * @param properties_map Property-name mapping used to construct the pack.
   */
  AbstractReactionKernels(ARGUMENT_PACK_T argument_pack, INT pre_req_ndims = 0,
                          const PropertiesMap &properties_map = PropertiesMap())
      : argument_pack(std::move(argument_pack)), properties_map(properties_map),
        pre_req_ndims(pre_req_ndims) {
    detail::validate_kernel_pre_req_ndims(pre_req_ndims);
  }

  std::optional<ON_DEVICE_T> on_device_obj;
  ARGUMENT_PACK_T argument_pack;
  PropertiesMap properties_map;
  INT pre_req_ndims;
};

} // namespace VANTAGE::Reactions

#endif
