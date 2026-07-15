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
