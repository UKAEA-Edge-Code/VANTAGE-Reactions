#ifndef REACTIONS_REACTION_KERNELS_H
#define REACTIONS_REACTION_KERNELS_H

#include "reaction_kernels_abstract.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Bundles the accessors passed into a single ParticleLoop for
 * single-particle reaction kernels.
 */
struct SingleReactionKernelAccessors {
  NP::Access::LoopIndex::Read index;
  NP::Access::SymVector::Write<INT> req_int_props;
  NP::Access::SymVector::Write<REAL> req_real_props;
  NP::Access::DescendantProducts::Write descendant_products;
  NP::Access::NDLocalArray::Read<REAL, 2> pre_req_data;
};

/**
 * @brief Bundles the host-side input and descendant property requirements for
 * single-particle reaction kernels.
 */
struct SingleReactionKernelArgumentPack {
  ArgumentNameSet<INT> required_int_props;
  ArgumentNameSet<REAL> required_real_props;
  ArgumentNameSet<INT> required_descendant_int_props;
  ArgumentNameSet<REAL> required_descendant_real_props;

  SingleReactionKernelArgumentPack() = default;

  SingleReactionKernelArgumentPack(
      ArgumentNameSet<INT> required_int_props,
      ArgumentNameSet<REAL> required_real_props,
      ArgumentNameSet<INT> required_descendant_int_props = {},
      ArgumentNameSet<REAL> required_descendant_real_props = {})
      : required_int_props(std::move(required_int_props)),
        required_real_props(std::move(required_real_props)),
        required_descendant_int_props(std::move(required_descendant_int_props)),
        required_descendant_real_props(
            std::move(required_descendant_real_props)) {}

  /**
   * @brief Build a pack from regular and ephemeral input property requirements
   * and descendant property requirements.
   *
   * @param required_int_props Regular integer input properties.
   * @param required_real_props Regular real input properties.
   * @param required_int_props_ephemeral Ephemeral integer input properties.
   * @param required_real_props_ephemeral Ephemeral real input properties.
   * @param required_descendant_int_props Explicitly written integer descendant
   * properties.
   * @param required_descendant_real_props Explicitly written real descendant
   * properties.
   * @param properties_map Property-name mapping used for all requirements.
   */
  SingleReactionKernelArgumentPack(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      Properties<INT> required_int_props_ephemeral,
      Properties<REAL> required_real_props_ephemeral,
      Properties<INT> required_descendant_int_props,
      Properties<REAL> required_descendant_real_props,
      const PropertiesMap &properties_map = PropertiesMap());

  /**
   * \overload
   * @brief Build an input-only pack from regular and ephemeral property
   * requirements.
   *
   * @param required_int_props Regular integer input properties.
   * @param required_real_props Regular real input properties.
   * @param required_int_props_ephemeral Ephemeral integer input properties.
   * @param required_real_props_ephemeral Ephemeral real input properties.
   * @param properties_map Property-name mapping used for all requirements.
   */
  SingleReactionKernelArgumentPack(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      Properties<INT> required_int_props_ephemeral,
      Properties<REAL> required_real_props_ephemeral,
      const PropertiesMap &properties_map = PropertiesMap());

  std::vector<NP::Sym<INT>> int_sym_vector() const {
    return required_int_props.to_sym_vector();
  }
  std::vector<NP::Sym<REAL>> real_sym_vector() const {
    return required_real_props.to_sym_vector();
  }
};

/// \cond
template <> struct accessor_pack_for<SingleReactionKernelArgumentPack> {
  using type = SingleReactionKernelAccessors;
};
/// \endcond

/**
 * @brief Base single-particle reaction kernels object to be used on SYCL
 * devices.
 *
 * @tparam num_products_per_parent Fixed number of products produced per parent.
 */
template <int num_products_per_parent>
struct ReactionKernelsBaseOnDevice
    : AbstractReactionKernelsOnDevice<SingleReactionKernelAccessors,
                                      num_products_per_parent> {};

/**
 * @brief Base host class for single-particle reaction kernels.
 *
 * @tparam ON_DEVICE_T Type of the on-device kernel object.
 */
template <typename ON_DEVICE_T>
struct ReactionKernelsBase
    : AbstractReactionKernels<ON_DEVICE_T, SingleReactionKernelArgumentPack> {
  using Base =
      AbstractReactionKernels<ON_DEVICE_T, SingleReactionKernelArgumentPack>;

  /**
   * @brief Construct from a complete SingleReactionKernelArgumentPack.
   *
   * @param argument_pack Complete input and explicitly written descendant
   * names.
   * @param pre_req_ndims Prerequisite component count.
   * @param ndim_velocity Velocity component count for descendant velocity.
   * @param properties_map Mapping used to construct argument_pack.
   */
  ReactionKernelsBase(
      SingleReactionKernelArgumentPack argument_pack, INT pre_req_ndims = 0,
      int ndim_velocity = 2,
      const PropertiesMap &properties_map = PropertiesMap())
      : Base(std::move(argument_pack), pre_req_ndims, properties_map) {
    if constexpr (this->NUM_PRODUCTS > 0) {
      NESOWARN(
          !this->argument_pack.required_descendant_int_props.to_string_vector()
                  .empty() ||
              !this->argument_pack.required_descendant_real_props
                   .to_string_vector()
                   .empty(),
          "The number of products per parent is >= 1 but no required "
          "descendant properties are set. This will result in an empty "
          "descendant_matrix_spec.");
      this->descendant_matrix_spec = build_kernel_product_schema(
          this->argument_pack.required_descendant_int_props,
          this->argument_pack.required_descendant_real_props, ndim_velocity,
          this->get_properties_map());
    } else {
      this->descendant_matrix_spec =
          NP::product_matrix_spec(NP::ParticleSpec());
    }
  }

  /**
   * \overload
   * @brief Construct with regular/ephemeral inputs and descendant properties.
   */
  ReactionKernelsBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      Properties<INT> required_int_props_ephemeral,
      Properties<REAL> required_real_props_ephemeral,
      Properties<INT> required_descendant_int_props,
      Properties<REAL> required_descendant_real_props, INT pre_req_ndims,
      int ndim_velocity, const PropertiesMap &properties_map = PropertiesMap())
      : ReactionKernelsBase(
            SingleReactionKernelArgumentPack(
                std::move(required_int_props), std::move(required_real_props),
                std::move(required_int_props_ephemeral),
                std::move(required_real_props_ephemeral),
                std::move(required_descendant_int_props),
                std::move(required_descendant_real_props), properties_map),
            pre_req_ndims, ndim_velocity, properties_map) {}

  /**
   * \overload
   * @brief Construct with real input properties and descendant properties.
   */
  ReactionKernelsBase(Properties<REAL> required_real_props,
                      Properties<INT> required_descendant_int_props,
                      Properties<REAL> required_descendant_real_props,
                      INT pre_req_ndims, int ndim_velocity,
                      const PropertiesMap &properties_map = PropertiesMap())
      : ReactionKernelsBase(Properties<INT>(), std::move(required_real_props),
                            Properties<INT>(), Properties<REAL>(),
                            std::move(required_descendant_int_props),
                            std::move(required_descendant_real_props),
                            pre_req_ndims, ndim_velocity, properties_map) {}

  /**
   * \overload
   * @brief Construct with regular and ephemeral input properties (no
   * explicitly written descendants).
   */
  ReactionKernelsBase(Properties<INT> required_int_props,
                      Properties<REAL> required_real_props,
                      Properties<INT> required_int_props_ephemeral,
                      Properties<REAL> required_real_props_ephemeral,
                      INT pre_req_ndims = 0,
                      const PropertiesMap &properties_map = PropertiesMap())
      : ReactionKernelsBase(
            SingleReactionKernelArgumentPack(
                std::move(required_int_props), std::move(required_real_props),
                std::move(required_int_props_ephemeral),
                std::move(required_real_props_ephemeral), properties_map),
            pre_req_ndims, 2, properties_map) {}

  /**
   * \overload
   * @brief Construct with no required properties.
   */
  ReactionKernelsBase(const PropertiesMap &properties_map = PropertiesMap())
      : ReactionKernelsBase(Properties<INT>(), Properties<REAL>(),
                            Properties<INT>(), Properties<REAL>(), 0,
                            properties_map) {}

  /**
   * \overload
   * @brief Construct with only required integer input properties.
   */
  ReactionKernelsBase(Properties<INT> required_int_props, INT pre_req_ndims = 0,
                      const PropertiesMap &properties_map = PropertiesMap())
      : ReactionKernelsBase(std::move(required_int_props), Properties<REAL>(),
                            Properties<INT>(), Properties<REAL>(),
                            pre_req_ndims, properties_map) {}

  /**
   * \overload
   * @brief Construct with only required real input properties.
   */
  ReactionKernelsBase(Properties<REAL> required_real_props,
                      INT pre_req_ndims = 0,
                      const PropertiesMap &properties_map = PropertiesMap())
      : ReactionKernelsBase(Properties<INT>(), std::move(required_real_props),
                            Properties<INT>(), Properties<REAL>(),
                            pre_req_ndims, properties_map) {}

  /**
   * \overload
   * @brief Construct with required integer and real input properties.
   */
  ReactionKernelsBase(Properties<INT> required_int_props,
                      Properties<REAL> required_real_props,
                      INT pre_req_ndims = 0,
                      const PropertiesMap &properties_map = PropertiesMap())
      : ReactionKernelsBase(std::move(required_int_props),
                            std::move(required_real_props), Properties<INT>(),
                            Properties<REAL>(), pre_req_ndims, properties_map) {
  }

  /** @brief Get the fixed product count. @return Device output-state extent. */
  INT get_num_products() const { return this->NUM_PRODUCTS; }

  /** @brief Copy the product schema. @return Independent mutable NESO schema.
   */
  std::shared_ptr<NP::ProductMatrixSpec> get_descendant_matrix_spec() const {
    return std::make_shared<NP::ProductMatrixSpec>(
        *this->descendant_matrix_spec);
  }

private:
  std::shared_ptr<const NP::ProductMatrixSpec> descendant_matrix_spec;
};

/**
 * @brief Whether a type is a valid single-particle reaction kernel host.
 * @tparam T Candidate host kernel type.
 */
template <typename T, typename = void>
struct is_single_reaction_kernel : std::false_type {};

/// \cond
template <typename T>
struct is_single_reaction_kernel<
    T, std::void_t<typename T::ON_DEVICE_OBJ_TYPE,
                   typename T::ARGUMENT_PACK_TYPE, decltype(T::NUM_PRODUCTS)>>
    : std::bool_constant<
          std::is_base_of_v<ReactionKernelsBase<typename T::ON_DEVICE_OBJ_TYPE>,
                            T> &&
          std::is_same_v<typename T::ARGUMENT_PACK_TYPE,
                         SingleReactionKernelArgumentPack>> {};

namespace detail {
template <typename T, int num_products,
          bool IsKernel = is_single_reaction_kernel<T>::value>
struct has_matching_kernel_product_count : std::true_type {};

template <typename T, int num_products>
struct has_matching_kernel_product_count<T, num_products, true>
    : std::bool_constant<T::NUM_PRODUCTS == num_products> {};
} // namespace detail
/// \endcond

} // namespace VANTAGE::Reactions

#endif
