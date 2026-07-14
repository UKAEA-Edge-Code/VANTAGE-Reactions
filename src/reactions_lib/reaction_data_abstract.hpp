#ifndef REACTIONS_REACTION_DATA_ABSTRACT_H
#define REACTIONS_REACTION_DATA_ABSTRACT_H
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <type_traits>
#include <utility>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

using DEFAULT_RNG_KERNEL = NullKernelRNG<REAL>;

/**
 * @brief Compile-time helpers for checking that a derived on-device
 * reaction-data type defines calc_data with the correct parameter signature and
 * return type.
 *
 * These traits use std::void_t SFINAE (Substitution Failure Is Not An Error) to
 * detect whether T::calc_data(Args...) is a well-formed expression, and if
 * so, what type it returns.  They are used automatically inside
 * AbstractReactionData::validate_on_device_type(), and may also be used
 * manually in derived-class constructors if desired.
 */
namespace calc_data_traits {

// Primary template: calc_data(Args...) is NOT a valid expression.
template <typename, typename = void, typename...> struct calc_data_traits {
  static constexpr bool is_callable = false;
  using return_type = void;
};

/** Partial specialization: selected only when
 *  std::declval<const T>().calc_data(std::declval<Args>()...) is
 *  well-formed. std::void_t produces void for a valid expression
 *  and triggers SFINAE (Substitution Failure Is Not An Error) for an
 *  invalid one, causing the compiler to fall back to the primary
 *  template above.
 */
template <typename T, typename... Args>
struct calc_data_traits<T,
                        std::void_t<decltype(std::declval<const T>().calc_data(
                            std::declval<Args>()...))>,
                        Args...> {
  static constexpr bool is_callable = true;
  using return_type =
      decltype(std::declval<const T>().calc_data(std::declval<Args>()...));
};

} // namespace calc_data_traits

/**
 * @brief true if T::calc_data(Args...) is a valid call expression.
 */
template <typename T, typename... Args>
inline constexpr bool is_abstract_calc_data_callable_v =
    calc_data_traits::calc_data_traits<T, void, Args...>::is_callable;

/**
 * @brief The return type of T::calc_data(Args...) (or void if not
 * callable).
 */
template <typename T, typename... Args>
using abstract_calc_data_return_t =
    typename calc_data_traits::calc_data_traits<T, void, Args...>::return_type;

/**
 * @brief Check whether T::calc_data(Args...) returns exactly Expected.
 *
 * This helper short-circuits: if the parameter signature is wrong,
 * is_abstract_calc_data_callable_v is false and the function returns true
 * so that a separate static_assert on parameter mismatch can be the
 * only error emitted.  When the parameter signature is correct but
 * the return type differs, it returns false.
 */
template <typename T, typename Expected, typename... Args>
constexpr bool check_abstract_calc_data_return_type() {
  if constexpr (is_abstract_calc_data_callable_v<T, Args...>) {
    return std::is_same_v<abstract_calc_data_return_t<T, Args...>, Expected>;
  } else {
    return true;
  }
}

/**
 * @brief CRTP base class for argument packs, enforcing a uniform merge_with
 * interface. Each concrete argument pack (e.g. SingleReactionDataArgumentPack,
 * PairReactionDataArgumentPack) derives from this and implements
 * merge_with_impl.
 *
 * @tparam Derived The concrete argument pack type.
 */
template <typename Derived> struct AbstractArgumentPack {

  /**
   * @brief Merge this pack with another of the same type, returning a new pack
   * containing the union of all required properties.
   *
   * @param other The pack to merge with.
   * @return A new pack of type Derived with merged required properties.
   */
  Derived merge_with(const Derived &other) const {
    return static_cast<const Derived *>(this)->merge_with_impl(other);
  }
};

template <typename ON_DEVICE_T, typename ARGUMENT_PACK_T, size_t dim = 1,
          typename RNG_KERNEL_T = DEFAULT_RNG_KERNEL, size_t input_dim = 0>
struct AbstractReactionData {

  static_assert(
      std::is_base_of_v<AbstractArgumentPack<ARGUMENT_PACK_T>, ARGUMENT_PACK_T>,
      "ARGUMENT_PACK_T must derive from "
      "AbstractArgumentPack<ARGUMENT_PACK_T>");

  using RNG_KERNEL_TYPE = RNG_KERNEL_T;
  using ARGUMENT_PACK_TYPE = ARGUMENT_PACK_T;
  using ON_DEVICE_OBJ_TYPE = ON_DEVICE_T;
  static const size_t DIM = dim;
  static const size_t INPUT_DIM = input_dim;

  /**
   * @brief Validate that the on-device type defines calc_data with the
   * parameter signature and return type expected by this host base class.
   */
  static constexpr bool validate_on_device_type() {
    if constexpr (ON_DEVICE_T::INPUT_DIM > 0) {
      using input_t = const std::array<typename ON_DEVICE_T::INPUT_TYPE,
                                       ON_DEVICE_T::INPUT_DIM> &;

      static_assert(is_abstract_calc_data_callable_v<
                        ON_DEVICE_T, input_t,
                        const typename ON_DEVICE_T::ACCESSOR_PACK_TYPE &,
                        typename RNG_KERNEL_T::KernelType &>,
                    "ON_DEVICE_T::calc_data parameter signature mismatch");

      static_assert(
          check_abstract_calc_data_return_type<
              ON_DEVICE_T,
              std::array<typename ON_DEVICE_T::VALUE_TYPE, ON_DEVICE_T::DIM>,
              input_t, const typename ON_DEVICE_T::ACCESSOR_PACK_TYPE &,
              typename RNG_KERNEL_T::KernelType &>(),
          "ON_DEVICE_T::calc_data return type mismatch");
    }
    return true;
  }

  AbstractReactionData(
      ARGUMENT_PACK_T argument_pack,
      std::map<int, std::string> properties_map = get_default_map())
      : argument_pack(argument_pack), properties_map(properties_map) {

    static_assert(validate_on_device_type());
    this->rng_kernel = std::make_shared<RNG_KERNEL_T>();
  }

  ARGUMENT_PACK_T get_arg_pack() { return this->argument_pack; }
  void set_arg_pack(const ARGUMENT_PACK_T &argument_pack) {
    this->argument_pack = argument_pack;
    this->index_on_device_object();
  }

  void set_rng_kernel(std::shared_ptr<RNG_KERNEL_T> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }

  std::shared_ptr<RNG_KERNEL_T> get_rng_kernel() { return this->rng_kernel; }

  static constexpr size_t get_dim() { return dim; }

  virtual ~AbstractReactionData() = default;

  /**
   * @brief To be implemented by each derived class in order to handle required
   * property indexing on the on-device object
   */
  virtual void index_on_device_object() {};

  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */
  const ON_DEVICE_T &get_on_device_obj() {

    NESOASSERT(this->on_device_obj.has_value(),
               "on_device_obj in AbstractReactionData not initialised");
    return this->on_device_obj.value();
  }

protected:
  std::optional<ON_DEVICE_T> on_device_obj;
  ARGUMENT_PACK_T argument_pack;
  std::shared_ptr<RNG_KERNEL_T> rng_kernel;
  std::map<int, std::string> properties_map;
};

template <typename ACCESSOR_PACK_T, size_t dim = 1,
          typename RNG_KERNEL_T = DEFAULT_RNG_KERNEL, size_t input_dim = 0,
          typename VALUE_T = REAL, typename INPUT_T = REAL>
struct AbstractReactionDataOnDevice {
  using RNG_KERNEL_TYPE = RNG_KERNEL_T;
  using ACCESSOR_PACK_TYPE = ACCESSOR_PACK_T;
  using VALUE_TYPE = VALUE_T;
  using INPUT_TYPE = INPUT_T;
  static const size_t DIM = dim;
  static const size_t INPUT_DIM = input_dim;

  AbstractReactionDataOnDevice() = default;

  template <std::size_t D = INPUT_DIM,
            std::enable_if_t<(D == 0) && D == INPUT_DIM, int> = 0>
  std::array<VALUE_T, dim>
  calc_data(const ACCESSOR_PACK_T &accessor_pack,
            typename RNG_KERNEL_T::KernelType &rng_kernel) const {
    return std::array<REAL, dim>{0.0};
  }

  template <std::size_t D = INPUT_DIM,
            std::enable_if_t<(D > 0) && D == INPUT_DIM, int> = 0>
  std::array<VALUE_T, dim>
  calc_data(const std::array<INPUT_T, INPUT_DIM> &input,
            const ACCESSOR_PACK_T &accessor_pack,
            typename RNG_KERNEL_T::KernelType &rng_kernel) const {
    return std::array<REAL, dim>{0.0};
  }
  static constexpr size_t get_dim() { return dim; }
};

/**
 * @brief Compile-time mapping from a host-side argument-pack type to its
 * matching device-side accessor-pack type.
 *
 * The primary template is deliberately left undefined so that only the known
 * (host, device) pack pairings are admissible. The specialisations cover the
 * single and pair reaction-data pack pairs. Concrete reaction-data leaf types
 * that are pack-agnostic (e.g. UnaryArrayTransformData) take an
 * ARGUMENT_PACK_T template argument and derive the accessor pack via
 * accessor_pack_for_t<ARGUMENT_PACK_T>, so they can sit inside either a single
 * or a pair pipeline.
 *
 * @tparam ARGUMENT_PACK_T A concrete argument-pack type deriving from
 * AbstractArgumentPack.
 */
template <typename ARGUMENT_PACK_T> struct accessor_pack_for;

/**
 * @brief Convenience alias for accessor_pack_for<ARGUMENT_PACK_T>::type.
 */
template <typename ARGUMENT_PACK_T>
using accessor_pack_for_t = typename accessor_pack_for<ARGUMENT_PACK_T>::type;

}; // namespace VANTAGE::Reactions
#endif
