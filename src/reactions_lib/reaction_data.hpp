#ifndef REACTIONS_REACTION_DATA_H
#define REACTIONS_REACTION_DATA_H
#include "reaction_data_abstract.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <type_traits>
#include <utility>

using namespace NESO::Particles;
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
   * @return REAL-valued cross-section at requested relative vel magnitude
   */
  REAL get_value_at(const REAL &relative_vel) const {
    // This function should never actually be called. If it is called and we do
    // not have a return value then the calling function will receive an
    // undefined value. By setting a value we at least know what the returned
    // value is and can pick one that is detectable.
    return std::numeric_limits<REAL>::lowest();
  };

  /**
   * @brief Get the maximum value of sigma*v_r where sigma is this cross-section
   * evaluated at v_r and v_r is the relative speed of the projectile and target
   *
   * @return REAL-valued maximum rate
   */
  REAL get_max_rate_val() const {
    // This function should never actually be called. If it is called and we do
    // not have a return value then the calling function will receive an
    // undefined value. By setting a value we at least know what the returned
    // value is and can pick one that is detectable.
    return std::numeric_limits<REAL>::lowest();
  };

  /**
   * @brief Get the greedy maximum value of sigma*v_r where sigma is this
   * cross-section evaluated at v_r and v_r is the relative speed of the
   * projectile and target. This should (in most cases) be implemented to return
   * values less than or equal to the one with get_max_rate_val()
   *
   * @param relative_vel Magnitude of relative velocity of the projectile and
   * target
   * @return REAL-valued maximum rate
   */
  REAL get_max_rate_val(REAL relative_vel) const {
    // TODO: check if this still causes issues (see workarounds below)
    return this->get_max_rate_val();
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
  bool accept_reject(REAL relative_vel, REAL uniform_rand, REAL value_at,
                     REAL max_rate_val) const {
    return uniform_rand < (value_at * relative_vel / max_rate_val);
  }
};

using DEFAULT_RNG_KERNEL = NullKernelRNG<REAL>;

/**
 * @brief Bundles the accessors passed into a single ParticleLoop for single
 * (non-pair) reaction data.
 */
struct SingleReactionDataAccessors {
  Access::LoopIndex::Read index;
  Access::SymVector::Write<INT> req_int_props;
  Access::SymVector::Read<REAL> req_real_props;
};

/**
 * @brief Bundles the host-side property requirements for single (non-pair)
 * reaction data.
 */
struct SingleReactionDataArgumentPack
    : AbstractArgumentPack<SingleReactionDataArgumentPack> {
  ArgumentNameSet<INT> required_int_props;
  ArgumentNameSet<REAL> required_real_props;

  SingleReactionDataArgumentPack() = default;

  SingleReactionDataArgumentPack(ArgumentNameSet<INT> required_int_props,
                                 ArgumentNameSet<REAL> required_real_props)
      : required_int_props(required_int_props),
        required_real_props(required_real_props) {}

  std::vector<Sym<INT>> int_sym_vector() {
    return required_int_props.to_sym_vector();
  }
  std::vector<Sym<REAL>> real_sym_vector() {
    return required_real_props.to_sym_vector();
  }

  /**
   * @brief Merge this pack with another SingleReactionDataArgumentPack,
   * returning a new pack with the union of required properties.
   */
  SingleReactionDataArgumentPack
  merge_with_impl(const SingleReactionDataArgumentPack &other) const {
    SingleReactionDataArgumentPack result;
    result.required_int_props =
        this->required_int_props.merge_with(other.required_int_props);
    result.required_real_props =
        this->required_real_props.merge_with(other.required_real_props);
    return result;
  }
};

template <> struct accessor_pack_for<SingleReactionDataArgumentPack> {
  using type = SingleReactionDataAccessors;
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
struct ReactionDataBase
    : AbstractReactionData<ON_DEVICE_TYPE, SingleReactionDataArgumentPack, dim,
                           RNG_TYPE, input_dim> {

  using Base =
      AbstractReactionData<ON_DEVICE_TYPE, SingleReactionDataArgumentPack, dim,
                           RNG_TYPE, input_dim>;

  /**
   * @brief Constructor for ReactionDataBase.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param required_int_props_ephemeral Properties<INT> object containing
   * information regarding the required INT-based ephemeral properties for the
   * reaction data.
   * @param required_real_props_ephemeral Properties<REAL> object containing
   * information regarding the required REAL-based ephemeral properties for the
   * reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  ReactionDataBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      Properties<INT> required_int_props_ephemeral,
      Properties<REAL> required_real_props_ephemeral,
      std::map<int, std::string> properties_map = get_default_map())
      : Base(SingleReactionDataArgumentPack(
                 ArgumentNameSet(required_int_props, properties_map)
                     .merge_with(ArgumentNameSet(required_int_props_ephemeral,
                                                 properties_map)),
                 ArgumentNameSet(required_real_props, properties_map)
                     .merge_with(ArgumentNameSet(required_real_props_ephemeral,
                                                 properties_map))),
             properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets no required properties.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  ReactionDataBase(
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(Properties<INT>(), Properties<REAL>(),
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required int
   * properties.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  ReactionDataBase(
      Properties<INT> required_int_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(required_int_props, Properties<REAL>(),
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required real
   * properties.
   *
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  ReactionDataBase(
      Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(Properties<INT>(), required_real_props,
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required int and
   * real properties.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  ReactionDataBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(required_int_props, required_real_props,
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}
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
 * default REAL)
 * @tparam IN_TYPE Input type of array required by this object (if input_dim >0)
 */
template <size_t dim = 1, typename RNG_TYPE = DEFAULT_RNG_KERNEL,
          size_t input_dim = 0, typename VAL_TYPE = REAL,
          typename IN_TYPE = REAL>
struct ReactionDataBaseOnDevice
    : AbstractReactionDataOnDevice<SingleReactionDataAccessors, dim, RNG_TYPE,
                                   input_dim, VAL_TYPE, IN_TYPE> {};

}; // namespace VANTAGE::Reactions
#endif
