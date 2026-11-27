#ifndef REACTIONS_PAIR_REACTION_KERNELS_H
#define REACTIONS_PAIR_REACTION_KERNELS_H

#include "reaction_kernels_abstract.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Bundles the accessors passed into a ParticlePairLoop for pair reaction
 * kernels.
 */
struct PairReactionKernelAccessors {
  NP::Access::PairLoopIndex::Read index_a;
  NP::Access::PairLoopIndex::Read index_b;
  NP::Access::PairLoopIndex::Read pair_index;
  NP::Access::SymVector::Write<INT> req_int_props_a;
  NP::Access::SymVector::Write<REAL> req_real_props_a;
  NP::Access::SymVector::Write<INT> req_int_props_b;
  NP::Access::SymVector::Write<REAL> req_real_props_b;
  NP::Access::DescendantProducts::Write descendant_products_a;
  NP::Access::DescendantProducts::Write descendant_products_b;
  NP::Access::NDLocalArray::Read<REAL, 2> pre_req_data;
};

/**
 * @brief Bundles the host-side A/B input and descendant property requirements
 * for pair reaction kernels.
 */
struct PairReactionKernelArgumentPack {
  ArgumentNameSet<INT> required_int_props_a;
  ArgumentNameSet<REAL> required_real_props_a;
  ArgumentNameSet<INT> required_int_props_b;
  ArgumentNameSet<REAL> required_real_props_b;
  ArgumentNameSet<INT> required_descendant_int_props_a;
  ArgumentNameSet<REAL> required_descendant_real_props_a;
  ArgumentNameSet<INT> required_descendant_int_props_b;
  ArgumentNameSet<REAL> required_descendant_real_props_b;

  PairReactionKernelArgumentPack() = default;

  /**
   * @brief Build a pack from A/B input and descendant property requirements.
   *
   * @param required_int_props_a Integer input properties for particle A.
   * @param required_real_props_a Real input properties for particle A.
   * @param required_int_props_b Integer input properties for particle B.
   * @param required_real_props_b Real input properties for particle B.
   * @param required_descendant_int_props_a Explicitly written integer
   * descendant properties for channel A.
   * @param required_descendant_real_props_a Explicitly written real descendant
   * properties for channel A.
   * @param required_descendant_int_props_b Explicitly written integer
   * descendant properties for channel B.
   * @param required_descendant_real_props_b Explicitly written real descendant
   * properties for channel B.
   * @param properties_map Property-name mapping used for all requirements.
   */
  PairReactionKernelArgumentPack(
      Properties<INT> required_int_props_a,
      Properties<REAL> required_real_props_a,
      Properties<INT> required_int_props_b,
      Properties<REAL> required_real_props_b,
      Properties<INT> required_descendant_int_props_a,
      Properties<REAL> required_descendant_real_props_a,
      Properties<INT> required_descendant_int_props_b,
      Properties<REAL> required_descendant_real_props_b,
      const PropertiesMap &properties_map = PropertiesMap());

  /**
   * \overload
   * @brief Build an input-only pack from A/B property requirements.
   *
   * @param required_int_props_a Integer input properties for particle A.
   * @param required_real_props_a Real input properties for particle A.
   * @param required_int_props_b Integer input properties for particle B.
   * @param required_real_props_b Real input properties for particle B.
   * @param properties_map Property-name mapping used for all requirements.
   */
  PairReactionKernelArgumentPack(
      Properties<INT> required_int_props_a,
      Properties<REAL> required_real_props_a,
      Properties<INT> required_int_props_b,
      Properties<REAL> required_real_props_b,
      const PropertiesMap &properties_map = PropertiesMap());
};

/// \cond
namespace detail {
template <typename DEVICE, int N>
using kernel_parent_signature = void (DEVICE::*)(
    PairReactionKernelAccessors &, const std::array<int, N> &) const;

template <typename DEVICE, typename = void>
struct has_parent_kernel : std::false_type {};

template <typename DEVICE>
struct has_parent_kernel<
    DEVICE, std::void_t<decltype(static_cast<kernel_parent_signature<
                                     DEVICE, DEVICE::NUM_PRODUCTS>>(
                &DEVICE::parent_kernel))>> : std::true_type {};

void validate_kernel_product_counts(INT extent, INT count_a, INT count_b);
} // namespace detail

template <typename DEVICE>
inline constexpr bool is_pair_reaction_kernel_device_v = std::conjunction_v<
    detail::has_kernel_operations<DEVICE, PairReactionKernelAccessors>,
    detail::has_parent_kernel<DEVICE>>;
/// \endcond

/**
 * @brief Base pair reaction kernels object to be used on SYCL devices.
 *
 * @tparam num_products Fixed total output-state extent across A and B.
 */
template <int num_products>
struct PairReactionKernelsBaseOnDevice
    : AbstractReactionKernelsOnDevice<PairReactionKernelAccessors,
                                      num_products> {
  PairReactionKernelsBaseOnDevice() = default;

  /**
   * @brief Associate descendant products with parents; default does nothing.
   *
   * @param accessors Bundled pair-loop accessors.
   * @param out_states Explicit descendant species IDs.
   */
  void parent_kernel(PairReactionKernelAccessors &accessors,
                     const std::array<int, num_products> &out_states) const {}
};

/**
 * @brief Base host class for pair reaction kernels.
 *
 * Constructors without A/B product counts are only for zero-product kernels.
 * Inherited-only products require explicit counts and empty descendant sets.
 *
 * @tparam ON_DEVICE_T Type of the on-device kernel object.
 */
template <typename ON_DEVICE_T>
struct PairReactionKernelsBase
    : AbstractReactionKernels<ON_DEVICE_T, PairReactionKernelArgumentPack> {
  using Base =
      AbstractReactionKernels<ON_DEVICE_T, PairReactionKernelArgumentPack>;

  /** @brief Check device signatures at compile time. @return True if valid. */
  static constexpr bool validate_on_device_type() {
    static_assert(is_pair_reaction_kernel_device_v<ON_DEVICE_T>,
                  "Device kernel operation signature mismatch");
    return true;
  }

  /**
   * @brief Construct with distinct A/B input and descendant properties.
   */
  PairReactionKernelsBase(Properties<INT> required_int_props_a,
                          Properties<REAL> required_real_props_a,
                          Properties<INT> required_int_props_b,
                          Properties<REAL> required_real_props_b,
                          Properties<INT> required_descendant_int_props_a,
                          Properties<REAL> required_descendant_real_props_a,
                          Properties<INT> required_descendant_int_props_b,
                          Properties<REAL> required_descendant_real_props_b,
                          INT pre_req_ndims, int ndim_velocity,
                          INT num_products_a, INT num_products_b,
                          const PropertiesMap &properties_map = PropertiesMap())
      : PairReactionKernelsBase(PairReactionKernelArgumentPack(
                                    std::move(required_int_props_a),
                                    std::move(required_real_props_a),
                                    std::move(required_int_props_b),
                                    std::move(required_real_props_b),
                                    std::move(required_descendant_int_props_a),
                                    std::move(required_descendant_real_props_a),
                                    std::move(required_descendant_int_props_b),
                                    std::move(required_descendant_real_props_b),
                                    properties_map),
                                pre_req_ndims, ndim_velocity, num_products_a,
                                num_products_b, properties_map) {}

  /**
   * \overload
   * @brief Construct with symmetric A/B input and descendant properties.
   */
  PairReactionKernelsBase(Properties<INT> required_int_props,
                          Properties<REAL> required_real_props,
                          Properties<INT> required_descendant_int_props,
                          Properties<REAL> required_descendant_real_props,
                          INT pre_req_ndims, int ndim_velocity,
                          INT num_products_a, INT num_products_b,
                          const PropertiesMap &properties_map = PropertiesMap())
      : PairReactionKernelsBase(
            required_int_props, required_real_props, required_int_props,
            required_real_props, required_descendant_int_props,
            required_descendant_real_props, required_descendant_int_props,
            required_descendant_real_props, pre_req_ndims, ndim_velocity,
            num_products_a, num_products_b, properties_map) {}

  /**
   * \overload
   * @brief Construct with symmetric A/B real input properties and descendant
   * properties.
   */
  PairReactionKernelsBase(Properties<REAL> required_real_props,
                          Properties<INT> required_descendant_int_props,
                          Properties<REAL> required_descendant_real_props,
                          INT pre_req_ndims, int ndim_velocity,
                          INT num_products_a, INT num_products_b,
                          const PropertiesMap &properties_map = PropertiesMap())
      : PairReactionKernelsBase(
            Properties<INT>(), std::move(required_real_props),
            std::move(required_descendant_int_props),
            std::move(required_descendant_real_props), pre_req_ndims,
            ndim_velocity, num_products_a, num_products_b, properties_map) {}

  /**
   * \overload
   * @brief Construct with distinct A/B input properties (no explicitly written
   * descendants).
   */
  PairReactionKernelsBase(Properties<INT> req_int_props_a,
                          Properties<REAL> req_real_props_a,
                          Properties<INT> req_int_props_b,
                          Properties<REAL> req_real_props_b,
                          INT pre_req_ndims = 0,
                          const PropertiesMap &properties_map = PropertiesMap())
      : PairReactionKernelsBase(
            PairReactionKernelArgumentPack(
                std::move(req_int_props_a), std::move(req_real_props_a),
                std::move(req_int_props_b), std::move(req_real_props_b),
                properties_map),
            pre_req_ndims, 2, 0, 0, properties_map) {
    static_assert(
        Base::NUM_PRODUCTS == 0,
        "Pair kernels with products require explicit A/B counts; use a "
        "descendant-properties constructor.");
  }

  /**
   * \overload
   * @brief Construct with no required properties.
   */
  PairReactionKernelsBase(const PropertiesMap &properties_map = PropertiesMap())
      : PairReactionKernelsBase(Properties<INT>(), Properties<REAL>(),
                                Properties<INT>(), Properties<REAL>(), 0,
                                properties_map) {}

  /**
   * \overload
   * @brief Construct with symmetric A/B integer input properties.
   */
  PairReactionKernelsBase(Properties<INT> required_int_props,
                          INT pre_req_ndims = 0,
                          const PropertiesMap &properties_map = PropertiesMap())
      : PairReactionKernelsBase(required_int_props, Properties<REAL>(),
                                required_int_props, Properties<REAL>(),
                                pre_req_ndims, properties_map) {}

  /**
   * \overload
   * @brief Construct with symmetric A/B real input properties.
   */
  PairReactionKernelsBase(Properties<REAL> required_real_props,
                          INT pre_req_ndims = 0,
                          const PropertiesMap &properties_map = PropertiesMap())
      : PairReactionKernelsBase(Properties<INT>(), required_real_props,
                                Properties<INT>(), required_real_props,
                                pre_req_ndims, properties_map) {}

  /**
   * \overload
   * @brief Construct with symmetric A/B integer and real input properties.
   */
  PairReactionKernelsBase(Properties<INT> required_int_props,
                          Properties<REAL> required_real_props,
                          INT pre_req_ndims = 0,
                          const PropertiesMap &properties_map = PropertiesMap())
      : PairReactionKernelsBase(required_int_props, required_real_props,
                                required_int_props, required_real_props,
                                pre_req_ndims, properties_map) {}

  /** @brief Get the fixed total product count. @return Device output-state
   * extent. */
  INT get_num_products() const { return this->NUM_PRODUCTS; }

  /** @brief Get the fixed channel A product count. @return Number of A
   * products. */
  INT get_num_products_a() const { return this->num_products_a; }

  /** @brief Get the fixed channel B product count. @return Number of B
   * products. */
  INT get_num_products_b() const { return this->num_products_b; }

  /** @brief Copy the channel A product schema. @return Independent mutable NESO
   * schema. */
  std::shared_ptr<NP::ProductMatrixSpec> get_descendant_matrix_spec_a() const {
    return std::make_shared<NP::ProductMatrixSpec>(
        *this->descendant_matrix_spec_a);
  }

  /** @brief Copy the channel B product schema. @return Independent mutable NESO
   * schema. */
  std::shared_ptr<NP::ProductMatrixSpec> get_descendant_matrix_spec_b() const {
    return std::make_shared<NP::ProductMatrixSpec>(
        *this->descendant_matrix_spec_b);
  }

private:
  PairReactionKernelsBase(PairReactionKernelArgumentPack argument_pack,
                          INT pre_req_ndims, int ndim_velocity,
                          INT num_products_a, INT num_products_b,
                          const PropertiesMap &properties_map)
      : Base(std::move(argument_pack), pre_req_ndims, properties_map),
        num_products_a(num_products_a), num_products_b(num_products_b) {
    static_assert(validate_on_device_type());
    detail::validate_kernel_product_counts(this->NUM_PRODUCTS, num_products_a,
                                           num_products_b);
    this->descendant_matrix_spec_a = build_kernel_product_schema(
        this->argument_pack.required_descendant_int_props_a,
        this->argument_pack.required_descendant_real_props_a, ndim_velocity,
        this->get_properties_map(), num_products_a);
    this->descendant_matrix_spec_b = build_kernel_product_schema(
        this->argument_pack.required_descendant_int_props_b,
        this->argument_pack.required_descendant_real_props_b, ndim_velocity,
        this->get_properties_map(), num_products_b);
  }

  INT num_products_a = 0;
  INT num_products_b = 0;
  std::shared_ptr<const NP::ProductMatrixSpec> descendant_matrix_spec_a;
  std::shared_ptr<const NP::ProductMatrixSpec> descendant_matrix_spec_b;
};

/**
 * @brief Whether a type is a valid pair reaction kernel host.
 * @tparam T Candidate host kernel type.
 */
template <typename T, typename = void>
struct is_pair_reaction_kernel : std::false_type {};

/// \cond
template <typename T>
struct is_pair_reaction_kernel<T, std::void_t<typename T::ON_DEVICE_OBJ_TYPE>>
    : std::is_base_of<PairReactionKernelsBase<typename T::ON_DEVICE_OBJ_TYPE>,
                      T> {};
/// \endcond

} // namespace VANTAGE::Reactions

#endif
