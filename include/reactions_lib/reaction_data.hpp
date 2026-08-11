#ifndef REACTIONS_REACTION_DATA_H
#define REACTIONS_REACTION_DATA_H
#include "../reactions/neso_particles_namespace_alias.hpp"
#include "../reactions/neso_test_assert.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>

#include <type_traits>
#include <utility>

namespace VANTAGE::Reactions {

/**
 * @brief An abstract base class for cross-section objects.
 * All classes derived from this class should be device copyable in order to be
 * used within ReactionData classes.
 */
struct AbstractCrossSection {

  /**
   * @brief Get the value of the cross section for a given relative velocity
   * value of projectile and target
   *
   * @param relative_vel Magnitude of relative velocity of target and projectile
   * @return NP::REAL-valued cross-section at requested relative vel magnitude
   */
  NP::REAL get_value_at(const NP::REAL &relative_vel) const {
    // This function should never actually be called. If it is called and we do
    // not have a return value then the calling function will receive an
    // undefined value. By setting a value we at least know what the returned
    // value is and can pick one that is detectable.
    return std::numeric_limits<NP::REAL>::lowest();
  };

  /**
   * @brief Get the maximum value of sigma*v_r where sigma is this cross-section
   * evaluated at v_r and v_r is the relative speed of the projectile and target
   *
   * @return NP::REAL-valued maximum rate
   */
  NP::REAL get_max_rate_val() const {
    // This function should never actually be called. If it is called and we do
    // not have a return value then the calling function will receive an
    // undefined value. By setting a value we at least know what the returned
    // value is and can pick one that is detectable.
    return std::numeric_limits<NP::REAL>::lowest();
  };

  /**
   * @brief Accept-reject function for when this cross-section is used in
   * rejection methods. Accepts if the uniform random number on (0,1) is less
   * than the ratio of sigma*v evaluated at a given relative speed to the
   * maximum value of sigma*v.
   *
   * @param relative_vel Magnitude of relative velocity of the projectile and
   * target
   * @param uniform_rand Uniformly sampled random number on (0,1)
   * @param value_at Value of cross section for a given relative velocity value
   * of projectile and target (NOTE this is currently a workaround due to the
   * limitation on calling get_value_at(...) inside this function.)
   * @param max_rate_val Maximum value of sigma*v_r (NOTE this is currently a
   * workaround due to the limitation on calling get_max_rate_val(...) inside
   * this function.)
   * @return true if relative_vel value is accepted, false otherwise
   */
  bool accept_reject(NP::REAL relative_vel, NP::REAL uniform_rand,
                     NP::REAL value_at, NP::REAL max_rate_val) const {
    return uniform_rand < (value_at * relative_vel / max_rate_val);
  }
};

using DEFAULT_RNG_KERNEL = NP::NullKernelRNG<NP::REAL>;

/**
 * @brief Compile-time helpers for checking that a derived on-device
 * reaction-data type defines calc_data with the correct parameter signature and
 * return type.
 *
 * These traits use std::void_t SFINAE (Substitution Failure Is Not An Error) to
 * detect whether T::calc_data(Args...) is a well-formed expression, and if
 * so, what type it returns.  They are used automatically inside
 * ReactionDataBase::validate_on_device_type(), and may also be used manually
 * in derived-class constructors if desired.
 */
namespace calc_data_traits_defs {

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

} // namespace calc_data_traits_defs

/**
 * @brief true if T::calc_data(Args...) is a valid call expression.
 */
template <typename T, typename... Args>
inline constexpr bool is_calc_data_callable_v =
    calc_data_traits_defs::calc_data_traits<T, void, Args...>::is_callable;

/**
 * @brief The return type of T::calc_data(Args...) (or void if not
 * callable).
 */
template <typename T, typename... Args>
using calc_data_return_t =
    typename calc_data_traits_defs::calc_data_traits<T, void,
                                                     Args...>::return_type;

/**
 * @brief Check whether T::calc_data(Args...) returns exactly Expected.
 *
 * This helper short-circuits: if the parameter signature is wrong,
 * is_calc_data_callable_v is false and the function returns true
 * so that a separate static_assert on parameter mismatch can be the
 * only error emitted.  When the parameter signature is correct but
 * the return type differs, it returns false.
 */
template <typename T, typename Expected, typename... Args>
constexpr bool check_calc_data_return_type() {
  if constexpr (is_calc_data_callable_v<T, Args...>) {
    return std::is_same_v<calc_data_return_t<T, Args...>, Expected>;
  } else {
    return true;
  }
}

/**
 * @brief Non-template implementation base for reaction data objects.
 *
 * Holds the required properties, property maps and constructors that do not
 * depend on the on-device type, dimension or RNG type.
 */
struct ReactionDataBaseImpl {

  /**
   * @brief Constructor for ReactionDataBaseImpl.
   *
   * @param required_int_props Properties<NP::INT> object containing information
   * regarding the required NP::INT-based properties for the reaction data.
   * @param required_real_props Properties<NP::REAL> object containing
   * information regarding the required NP::REAL-based properties for the
   * reaction data.
   * @param required_int_props_ephemeral Properties<NP::INT> object containing
   * information regarding the required NP::INT-based ephemeral properties for
   * the reaction data.
   * @param required_real_props_ephemeral Properties<NP::REAL> object containing
   * information regarding the required NP::REAL-based ephemeral properties for
   * the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBaseImpl(
      Properties<NP::INT> required_int_props,
      Properties<NP::REAL> required_real_props,
      Properties<NP::INT> required_int_props_ephemeral,
      Properties<NP::REAL> required_real_props_ephemeral,
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for ReactionDataBaseImpl that sets no required
   * properties.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBaseImpl(
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for ReactionDataBaseImpl that sets only required int
   * properties.
   *
   * @param required_int_props Properties<NP::INT> object containing information
   * regarding the required NP::INT-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBaseImpl(
      Properties<NP::INT> required_int_props,
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for ReactionDataBaseImpl that sets only required real
   * properties.
   *
   * @param required_real_props Properties<NP::REAL> object containing
   * information regarding the required NP::REAL-based properties for the
   * reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBaseImpl(
      Properties<NP::REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for ReactionDataBaseImpl that sets only required int
   * and real properties.
   *
   * @param required_int_props Properties<NP::INT> object containing information
   * regarding the required NP::INT-based properties for the reaction data.
   * @param required_real_props Properties<NP::REAL> object containing
   * information regarding the required NP::REAL-based properties for the
   * reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBaseImpl(
      Properties<NP::INT> required_int_props,
      Properties<NP::REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * @brief Return all required integer properties, including ephemeral
   *
   */
  ArgumentNameSet<NP::INT> get_required_int_props();

  /**
   * @brief Setter for required integer properties
   *
   * @param props ArgumentNameSet to use
   */
  virtual void set_required_int_props(const ArgumentNameSet<NP::INT> &props);

  /**
   * @brief Return all required integer properties as a vector of Syms
   *
   */
  std::vector<NP::Sym<NP::INT>> get_required_int_sym_vector();

  /**
   * @brief Return all required real properteis, including ephemeral
   * properties
   *
   */
  ArgumentNameSet<NP::REAL> get_required_real_props();

  /**
   * @brief Return all required real properties as a vector of Syms
   *
   */
  std::vector<NP::Sym<NP::REAL>> get_required_real_sym_vector();

  /**
   * @brief Setter for required real properties
   *
   * @param props ArgumentNameSet to use
   */
  virtual void set_required_real_props(const ArgumentNameSet<NP::REAL> &props);

  virtual ~ReactionDataBaseImpl();

  /**
   * @brief To be implemented by each derived class in order to handle required
   * property indexing on the on-device object
   */
  virtual void index_on_device_object();

protected:
  ArgumentNameSet<NP::INT> required_int_props;
  ArgumentNameSet<NP::REAL> required_real_props;
  std::map<int, std::string> properties_map;
};

/**
 * @brief Base reaction data object.
 *
 * @tparam ON_DEVICE_TYPE Type of the on-device object
 * @tparam dim Used to set the size of the array that calc_data returns
 * (Optional).
 * @tparam RNG_TYPE Sets the type of RNG that is used for sampling (Optional).
 * @tparam input_dim The dimension of the input array (Optional, defaults to 0,
 * not defining the corresponding calc_data)
 */
template <typename ON_DEVICE_TYPE, size_t dim = 1,
          typename RNG_TYPE = DEFAULT_RNG_KERNEL, size_t input_dim = 0>
struct ReactionDataBase : public ReactionDataBaseImpl {

  using RNG_KERNEL_TYPE = RNG_TYPE;
  using ON_DEVICE_OBJ_TYPE = ON_DEVICE_TYPE;
  static const size_t DIM = dim;
  static const size_t INPUT_DIM = input_dim;

  /**
   * @brief Validate that the on-device type defines calc_data with the
   * parameter signature and return type expected by this host base class.
   */
  static constexpr bool validate_on_device_type() {
    if constexpr (ON_DEVICE_TYPE::INPUT_DIM > 0) {
      using input_t = const std::array<typename ON_DEVICE_TYPE::INPUT_TYPE,
                                       ON_DEVICE_TYPE::INPUT_DIM> &;

      static_assert(
          is_calc_data_callable_v<ON_DEVICE_TYPE, input_t,
                                  const NP::Access::LoopIndex::Read &,
                                  const NP::Access::SymVector::Write<NP::INT> &,
                                  const NP::Access::SymVector::Read<NP::REAL> &,
                                  typename RNG_TYPE::KernelType &>,
          "ON_DEVICE_TYPE::calc_data parameter signature mismatch");

      static_assert(check_calc_data_return_type<
                        ON_DEVICE_TYPE,
                        std::array<typename ON_DEVICE_TYPE::VALUE_TYPE,
                                   ON_DEVICE_TYPE::DIM>,
                        input_t, const NP::Access::LoopIndex::Read &,
                        const NP::Access::SymVector::Write<NP::INT> &,
                        const NP::Access::SymVector::Read<NP::REAL> &,
                        typename RNG_TYPE::KernelType &>(),
                    "ON_DEVICE_TYPE::calc_data return type mismatch");
    }
    return true;
  }

  /**
   * @brief Constructor for ReactionDataBase.
   *
   * @param required_int_props Properties<NP::INT> object containing information
   * regarding the required NP::INT-based properties for the reaction data.
   * @param required_real_props Properties<NP::REAL> object containing
   * information regarding the required NP::REAL-based properties for the
   * reaction data.
   * @param required_int_props_ephemeral Properties<NP::INT> object containing
   * information regarding the required NP::INT-based ephemeral properties for
   * the reaction data.
   * @param required_real_props_ephemeral Properties<NP::REAL> object containing
   * information regarding the required NP::REAL-based ephemeral properties for
   * the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      Properties<NP::INT> required_int_props,
      Properties<NP::REAL> required_real_props,
      Properties<NP::INT> required_int_props_ephemeral,
      Properties<NP::REAL> required_real_props_ephemeral,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBaseImpl(required_int_props, required_real_props,
                             required_int_props_ephemeral,
                             required_real_props_ephemeral, properties_map) {
    static_assert(validate_on_device_type());
    this->rng_kernel = std::make_shared<RNG_TYPE>();
  }

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets no required properties.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(Properties<NP::INT>(), Properties<NP::REAL>(),
                         Properties<NP::INT>(), Properties<NP::REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required int
   * properties.
   *
   * @param required_int_props Properties<NP::INT> object containing information
   * regarding the required NP::INT-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      Properties<NP::INT> required_int_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(required_int_props, Properties<NP::REAL>(),
                         Properties<NP::INT>(), Properties<NP::REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required real
   * properties.
   *
   * @param required_real_props Properties<NP::REAL> object containing
   * information regarding the required NP::REAL-based properties for the
   * reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      Properties<NP::REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(Properties<NP::INT>(), required_real_props,
                         Properties<NP::INT>(), Properties<NP::REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required int and
   * real properties.
   *
   * @param required_int_props Properties<NP::INT> object containing information
   * regarding the required NP::INT-based properties for the reaction data.
   * @param required_real_props Properties<NP::REAL> object containing
   * information regarding the required NP::REAL-based properties for the
   * reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      Properties<NP::INT> required_int_props,
      Properties<NP::REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(required_int_props, required_real_props,
                         Properties<NP::INT>(), Properties<NP::REAL>(),
                         properties_map) {}

  void set_rng_kernel(std::shared_ptr<RNG_TYPE> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }

  std::shared_ptr<RNG_TYPE> get_rng_kernel() { return this->rng_kernel; }

  static constexpr size_t get_dim() { return dim; }

  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */
  const ON_DEVICE_TYPE &get_on_device_obj() {

    NESOASSERT(this->on_device_obj.has_value(),
               "on_device_obj in ReactionDataBase not initialised");
    return this->on_device_obj.value();
  }

protected:
  std::optional<ON_DEVICE_TYPE> on_device_obj;
  std::shared_ptr<RNG_TYPE> rng_kernel;
};

/**
 * @brief Base reaction data object to be used on SYCL devices.
 *
 * @tparam dim Used to set the size of the array that calc_data returns
 * (Optional).
 * @tparam RNG_TYPE Sets the type of RNG that is used for sampling (Optional).
 * @tparam input_dim The dimension of the optional input array (for use in
 * pipelines) (Optional, default 0)
 * @tparam VAL_TYPE Return type of this objects calc_data routine (Optional,
 * default NP::REAL)
 * @tparam IN_TYPE Input type of array required by this object (if input_dim >0)
 */
template <size_t dim = 1, typename RNG_TYPE = DEFAULT_RNG_KERNEL,
          size_t input_dim = 0, typename VAL_TYPE = NP::REAL,
          typename IN_TYPE = NP::REAL>
struct ReactionDataBaseOnDevice {
  using RNG_KERNEL_TYPE = RNG_TYPE;
  using VALUE_TYPE = VAL_TYPE;
  using INPUT_TYPE = IN_TYPE;
  static const size_t DIM = dim;
  static const size_t INPUT_DIM = input_dim;

  ReactionDataBaseOnDevice() = default;

  /**
   * @brief Function to calculate the reaction data.
   *
   * @param index Read-only accessor to a loop index for a NP::ParticleLoop
   * inside which calc_data is called. NP::Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction data calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction data calculation.
   * @param rng_kernel The random number generator kernel potentially used in
   * the calculation
   *
   * @return A NP::REAL-valued array of size dim containing the calculated
   * reaction rate.
   */
  template <std::size_t D = INPUT_DIM,
            std::enable_if_t<(D == 0) && D == INPUT_DIM, int> = 0>
  std::array<VAL_TYPE, dim>
  calc_data(const NP::Access::LoopIndex::Read &index,
            const NP::Access::SymVector::Write<NP::INT> &req_int_props,
            const NP::Access::SymVector::Read<NP::REAL> &req_real_props,
            typename RNG_TYPE::KernelType &rng_kernel) const {
    return std::array<NP::REAL, dim>{0.0};
  }

  template <std::size_t D = INPUT_DIM,
            std::enable_if_t<(D > 0) && D == INPUT_DIM, int> = 0>
  std::array<VAL_TYPE, dim>
  calc_data(const std::array<IN_TYPE, INPUT_DIM> &input,
            const NP::Access::LoopIndex::Read &index,
            const NP::Access::SymVector::Write<NP::INT> &req_int_props,
            const NP::Access::SymVector::Read<NP::REAL> &req_real_props,
            typename RNG_TYPE::KernelType &rng_kernel) const {
    return std::array<NP::REAL, dim>{0.0};
  }
  static constexpr size_t get_dim() { return dim; }
};

}; // namespace VANTAGE::Reactions
#endif
