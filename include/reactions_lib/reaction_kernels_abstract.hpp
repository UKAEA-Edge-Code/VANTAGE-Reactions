#ifndef REACTIONS_REACTION_KERNELS_ABSTRACT_H
#define REACTIONS_REACTION_KERNELS_ABSTRACT_H

#include "accessor_pack.hpp"
#include "reaction_kernel_argument_pack.hpp"
#include <array>
#include <optional>
#include <type_traits>
#include <utility>

namespace VANTAGE::Reactions {

/** @brief Non-owning single-kernel accessors, preserving existing access modes.
 */
struct SingleReactionKernelAccessors {
  NP::Access::LoopIndex::Read index;
  NP::Access::SymVector::Write<INT> req_int_props;
  NP::Access::SymVector::Write<REAL> req_real_props;
  NP::Access::DescendantProducts::Write descendant_products;
  NP::Access::NDLocalArray::Read<REAL, 2> pre_req_data;
};

/** @brief Non-owning pair-kernel accessors with independent A/B addressing. */
struct PairReactionKernelAccessors {
  NP::Access::PairLoopIndex::Read index_a, index_b, pair_index;
  NP::Access::SymVector::Write<INT> req_int_props_a, req_int_props_b;
  NP::Access::SymVector::Write<REAL> req_real_props_a, req_real_props_b;
  NP::Access::DescendantProducts::Write descendant_products_a,
      descendant_products_b;
  NP::Access::NDLocalArray::Read<REAL, 2> pre_req_data;
};

/// \cond
template <> struct accessor_pack_for<SingleReactionKernelArgumentPack> {
  using type = SingleReactionKernelAccessors;
};
template <> struct accessor_pack_for<PairReactionKernelArgumentPack> {
  using type = PairReactionKernelAccessors;
};
/// \endcond

/**
 * @brief Optional no-op operations for bundled-accessor device kernels.
 * @tparam ACCESSOR_PACK_T Single or pair device accessor pack.
 * @tparam num_products Fixed output-state extent, not a parenting convention.
 */
template <typename ACCESSOR_PACK_T, int num_products>
struct AbstractReactionKernelsOnDevice {
  static_assert(num_products >= 0, "Kernel product count must be nonnegative");
  using ACCESSOR_PACK_TYPE = ACCESSOR_PACK_T;
  static constexpr int NUM_PRODUCTS = num_products;

  /**
   * @brief Apply scattering; inherited implementation does nothing.
   * @param modified_weight Shared mutable reacted weight.
   * @param access Non-owning loop accessors.
   * @param out_states Explicit descendant species IDs.
   * @param dt Current effective timestep.
   */
  void scattering_kernel(REAL &modified_weight, ACCESSOR_PACK_T &access,
                         const std::array<int, num_products> &out_states,
                         double dt) const {}
  /** @copydoc scattering_kernel */
  void weight_kernel(REAL &modified_weight, ACCESSOR_PACK_T &access,
                     const std::array<int, num_products> &out_states,
                     double dt) const {}
  /** @copydoc scattering_kernel */
  void transformation_kernel(REAL &modified_weight, ACCESSOR_PACK_T &access,
                             const std::array<int, num_products> &out_states,
                             double dt) const {}
  /** @copydoc scattering_kernel */
  void feedback_kernel(REAL &modified_weight, ACCESSOR_PACK_T &access,
                       const std::array<int, num_products> &out_states,
                       double dt) const {}
};

/**
 * @brief Single-kernel no-op operations; parent association stays
 * reaction-owned.
 * @tparam num_products Fixed output-state extent.
 */
template <int num_products>
using SingleReactionKernelOnDevice =
    AbstractReactionKernelsOnDevice<SingleReactionKernelAccessors,
                                    num_products>;

/**
 * @brief Pair-kernel no-op operations, including explicit parent association.
 * @tparam num_products Total fixed output-state extent across A and B.
 */
template <int num_products>
struct PairReactionKernelOnDevice
    : AbstractReactionKernelsOnDevice<PairReactionKernelAccessors,
                                      num_products> {
  /**
   * @brief Associate descendants with parents; default does nothing.
   * @param access Pair-loop accessors.
   * @param out_states Explicit descendant species IDs.
   */
  void parent_kernel(PairReactionKernelAccessors &access,
                     const std::array<int, num_products> &out_states) const {}
};

namespace detail {
template <typename DEVICE, typename ACCESSORS>
using kernel_operation_signature = void (DEVICE::*)(
    REAL &, ACCESSORS &, const std::array<int, DEVICE::NUM_PRODUCTS> &,
    double) const;

template <typename DEVICE, typename ACCESSORS, typename = void>
struct has_kernel_operations : std::false_type {};
template <typename DEVICE, typename ACCESSORS>
struct has_kernel_operations<
    DEVICE, ACCESSORS,
    std::void_t<
        decltype(static_cast<kernel_operation_signature<DEVICE, ACCESSORS>>(
            &DEVICE::scattering_kernel)),
        decltype(static_cast<kernel_operation_signature<DEVICE, ACCESSORS>>(
            &DEVICE::weight_kernel)),
        decltype(static_cast<kernel_operation_signature<DEVICE, ACCESSORS>>(
            &DEVICE::transformation_kernel)),
        decltype(static_cast<kernel_operation_signature<DEVICE, ACCESSORS>>(
            &DEVICE::feedback_kernel))>> : std::true_type {};

template <typename DEVICE>
using kernel_parent_signature =
    void (DEVICE::*)(PairReactionKernelAccessors &,
                     const std::array<int, DEVICE::NUM_PRODUCTS> &) const;
template <typename DEVICE, typename = void>
struct has_kernel_parent : std::false_type {};
template <typename DEVICE>
struct has_kernel_parent<
    DEVICE, std::void_t<decltype(static_cast<kernel_parent_signature<DEVICE>>(
                &DEVICE::parent_kernel))>> : std::true_type {};

void validate_kernel_product_counts(INT extent, INT count_a, INT count_b);
void validate_kernel_pre_req_ndims(INT pre_req_ndims);
} // namespace detail

/**
 * @brief Whether all operations have the exact const bundled-hook signatures.
 * @tparam DEVICE Device kernel type, including its fixed NUM_PRODUCTS extent.
 * @tparam PACK Single or pair host argument-pack type.
 */
template <typename DEVICE, typename PACK>
inline constexpr bool is_reaction_kernel_device_v =
    detail::has_kernel_operations<DEVICE, accessor_pack_for_t<PACK>>::value &&
    (std::is_same_v<PACK, SingleReactionKernelArgumentPack> ||
     detail::has_kernel_parent<DEVICE>::value);

/**
 * @brief Construction-only ownership shared by single and pair kernel hosts.
 * @tparam ON_DEVICE_T Device type with bundled operations and NUM_PRODUCTS.
 * @tparam ARGUMENT_PACK_T Single or pair host argument-pack type.
 *
 * Derived constructors initialize on_device_obj, including non-index values,
 * then call their protected indexing hook. The base never dispatches it during
 * construction. There is no pack setter, public reindexing, or recovery state.
 */
template <typename ON_DEVICE_T, typename ARGUMENT_PACK_T>
class AbstractReactionKernels {
public:
  using ON_DEVICE_OBJ_TYPE = ON_DEVICE_T;
  using ARGUMENT_PACK_TYPE = ARGUMENT_PACK_T;
  static constexpr int NUM_PRODUCTS = ON_DEVICE_T::NUM_PRODUCTS;
  virtual ~AbstractReactionKernels() = default;

  /** @brief Check device signatures at compile time. @return True if valid. */
  static constexpr bool validate_on_device_type() {
    static_assert(is_reaction_kernel_device_v<ON_DEVICE_T, ARGUMENT_PACK_T>,
                  "Device kernel operation signature mismatch");
    static_assert(NUM_PRODUCTS >= 0,
                  "Kernel product count must be nonnegative");
    return true;
  }
  /** @brief Copy the configuration. @return Independent argument pack. */
  ARGUMENT_PACK_T get_arg_pack() const { return this->argument_pack; }
  /** @brief Get configured names. @return Read-only property map. */
  const PropertiesMap &get_properties_map() const {
    return this->properties_map;
  }
  /** @brief Get prerequisite size. @return Required component count. */
  INT get_pre_ndims() const { return this->pre_req_ndims; }
  /**
   * @brief Return the initialized device object without permitting mutation.
   * @return Read-only device object owned by this kernel.
   * @throws std::bad_optional_access If the derived constructor omitted it.
   */
  const ON_DEVICE_T &get_on_device_obj() const {
    return this->on_device_obj.value();
  }

protected:
  /**
   * @brief Store complete configuration without invoking the indexing hook.
   * @param argument_pack Authoritative kernel names.
   * @param pre_req_ndims Prerequisite-data component count, nonnegative.
   * @param properties_map Property-name mapping used to construct the pack.
   */
  AbstractReactionKernels(ARGUMENT_PACK_T argument_pack, INT pre_req_ndims,
                          PropertiesMap properties_map)
      : argument_pack(std::move(argument_pack)),
        properties_map(std::move(properties_map)),
        pre_req_ndims(pre_req_ndims) {
    static_assert(validate_on_device_type());
    detail::validate_kernel_pre_req_ndims(pre_req_ndims);
  }
  /** @brief Index the initialized device from the current pack and mapping. */
  virtual void index_on_device_object() = 0;
  std::optional<ON_DEVICE_T> on_device_obj;

private:
  ARGUMENT_PACK_T argument_pack;
  PropertiesMap properties_map;
  INT pre_req_ndims;
};

/**
 * @brief Single-kernel host with a fixed count and derived descendant schema.
 * @tparam ON_DEVICE_T Device kernel type; NUM_PRODUCTS defines the count.
 */
template <typename ON_DEVICE_T>
class SingleReactionKernelBase
    : public AbstractReactionKernels<ON_DEVICE_T,
                                     SingleReactionKernelArgumentPack> {
  using Base =
      AbstractReactionKernels<ON_DEVICE_T, SingleReactionKernelArgumentPack>;

public:
  /** @brief Get the fixed count. @return Device output-state extent. */
  INT get_num_products() const { return this->NUM_PRODUCTS; }
  /** @brief Copy the schema. @return Independent mutable NESO schema. */
  std::shared_ptr<NP::ProductMatrixSpec> get_descendant_matrix_spec() const {
    return std::make_shared<NP::ProductMatrixSpec>(
        *this->descendant_matrix_spec);
  }

protected:
  /**
   * @brief Configure single-kernel ownership and prepare its schema.
   * @param pack Complete input and explicitly written descendant names.
   * @param pre_req_ndims Prerequisite component count.
   * @param ndim_velocity Legacy velocity component count.
   * @param properties_map Mapping used to construct pack.
   */
  SingleReactionKernelBase(
      SingleReactionKernelArgumentPack pack, INT pre_req_ndims,
      int ndim_velocity,
      const PropertiesMap &properties_map = PropertiesMap())
      : Base(std::move(pack), pre_req_ndims, std::move(properties_map)),
        descendant_matrix_spec(build_kernel_product_schema(
            this->get_arg_pack().descendants, ndim_velocity,
            this->get_properties_map())) {}

private:
  std::shared_ptr<const NP::ProductMatrixSpec> descendant_matrix_spec;
};

/**
 * @brief Pair-kernel host with independent A/B counts and derived schemas.
 * @tparam ON_DEVICE_T Device kernel type; NUM_PRODUCTS is the total extent.
 */
template <typename ON_DEVICE_T>
class PairReactionKernelBase
    : public AbstractReactionKernels<ON_DEVICE_T,
                                     PairReactionKernelArgumentPack> {
  using Base =
      AbstractReactionKernels<ON_DEVICE_T, PairReactionKernelArgumentPack>;

public:
  /** @brief Get the fixed A count. @return Number of A products. */
  INT get_num_products_a() const { return this->num_products_a; }
  /** @brief Get the fixed B count. @return Number of B products. */
  INT get_num_products_b() const { return this->num_products_b; }
  /** @brief Copy the A schema. @return Independent mutable NESO schema. */
  std::shared_ptr<NP::ProductMatrixSpec> get_descendant_matrix_spec_a() const {
    return std::make_shared<NP::ProductMatrixSpec>(
        *this->descendant_matrix_spec_a);
  }
  /** @brief Copy the B schema. @return Independent mutable NESO schema. */
  std::shared_ptr<NP::ProductMatrixSpec> get_descendant_matrix_spec_b() const {
    return std::make_shared<NP::ProductMatrixSpec>(
        *this->descendant_matrix_spec_b);
  }

protected:
  /**
   * @brief Validate counts before preparing either descendant schema.
   * @param pack Complete A/B input and descendant names.
   * @param pre_req_ndims Prerequisite component count.
   * @param ndim_velocity Legacy velocity component count.
   * @param num_products_a Nonnegative A count.
   * @param num_products_b Nonnegative B count; A+B must equal device extent.
   * @param properties_map Mapping used to construct pack.
   * @throws std::invalid_argument If counts are negative or inconsistent.
   */
  PairReactionKernelBase(
      PairReactionKernelArgumentPack pack, INT pre_req_ndims, int ndim_velocity,
      INT num_products_a, INT num_products_b,
      const PropertiesMap &properties_map = PropertiesMap())
      : Base(std::move(pack), pre_req_ndims, std::move(properties_map)),
        num_products_a(num_products_a), num_products_b(num_products_b) {
    detail::validate_kernel_product_counts(this->NUM_PRODUCTS, num_products_a,
                                           num_products_b);
    auto current_pack = this->get_arg_pack();
    this->descendant_matrix_spec_a = build_kernel_product_schema(
        current_pack.descendants_a, ndim_velocity, this->get_properties_map());
    this->descendant_matrix_spec_b = build_kernel_product_schema(
        current_pack.descendants_b, ndim_velocity, this->get_properties_map());
  }

private:
  INT num_products_a, num_products_b;
  std::shared_ptr<const NP::ProductMatrixSpec> descendant_matrix_spec_a,
      descendant_matrix_spec_b;
};

} // namespace VANTAGE::Reactions

#endif
