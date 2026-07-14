#ifndef REACTIONS_PAIR_REACTION_DATA_H
#define REACTIONS_PAIR_REACTION_DATA_H
#include "reaction_data_abstract.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <type_traits>
#include <utility>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

struct PairReactionDataAccessors {

  PairReactionDataAccessors(Access::PairLoopIndex::Read index,
                            Access::SymVector::Write<INT> req_int_props_a,
                            Access::SymVector::Read<REAL> req_real_props_a,
                            Access::SymVector::Write<INT> req_int_props_b,
                            Access::SymVector::Read<REAL> req_real_props_b)
      : index(index), req_int_props_a(req_int_props_a),
        req_real_props_a(req_real_props_a), req_int_props_b(req_int_props_b),
        req_real_props_b(req_real_props_b) {};
  Access::PairLoopIndex::Read index;
  Access::SymVector::Write<INT> req_int_props_a;
  Access::SymVector::Read<REAL> req_real_props_a;
  Access::SymVector::Write<INT> req_int_props_b;
  Access::SymVector::Read<REAL> req_real_props_b;
};

struct PairReactionDataArgumentPack
    : AbstractArgumentPack<PairReactionDataArgumentPack> {

  PairReactionDataArgumentPack() = default;

  PairReactionDataArgumentPack(ArgumentNameSet<INT> req_int_props_a,
                               ArgumentNameSet<REAL> req_real_props_a,
                               ArgumentNameSet<INT> req_int_props_b,
                               ArgumentNameSet<REAL> req_real_props_b)
      : required_int_props_a(req_int_props_a),
        required_real_props_a(req_real_props_a),
        required_int_props_b(req_int_props_b),
        required_real_props_b(req_real_props_b) {};

  /**
   * @brief Merge this pack with another PairReactionDataArgumentPack,
   * returning a new pack with the union of required properties (merging
   * the _a and _b property sets separately).
   */
  PairReactionDataArgumentPack
  merge_with_impl(const PairReactionDataArgumentPack &other) const {
    PairReactionDataArgumentPack result;
    result.required_int_props_a =
        this->required_int_props_a.merge_with(other.required_int_props_a);
    result.required_int_props_b =
        this->required_int_props_b.merge_with(other.required_int_props_b);
    result.required_real_props_a =
        this->required_real_props_a.merge_with(other.required_real_props_a);
    result.required_real_props_b =
        this->required_real_props_b.merge_with(other.required_real_props_b);
    return result;
  }

  ArgumentNameSet<INT> required_int_props_a;
  ArgumentNameSet<REAL> required_real_props_a;
  ArgumentNameSet<INT> required_int_props_b;
  ArgumentNameSet<REAL> required_real_props_b;
};

template <> struct accessor_pack_for<PairReactionDataArgumentPack> {
  using type = PairReactionDataAccessors;
};

using DEFAULT_RNG_KERNEL = NullKernelRNG<REAL>;

/**
 * @brief Base pair reaction data object.
 *
 * @tparam ON_DEVICE_T Type of the on-device object
 * @tparam dim Used to set the size of the array that calc_data returns
 * (Optional).
 * @tparam RNG_KERNEL_T Sets the type of RNG that is used for sampling
 * (Optional).
 * @tparam input_dim The dimension of the input array (Optional, defaults to 0,
 * not defining the corresponding calc_data)
 */
template <typename ON_DEVICE_T, size_t dim = 1,
          typename RNG_KERNEL_T = DEFAULT_RNG_KERNEL, size_t input_dim = 0>
struct PairReactionDataBase
    : AbstractReactionData<ON_DEVICE_T, PairReactionDataArgumentPack, dim,
                           RNG_KERNEL_T, input_dim> {

  /**
   * @brief Constructor for PairReactionDataBase.
   *
   * @param required_int_props_a Properties<INT> object containing information
   * regarding the required INT-based properties of the first particle for the
   * reaction data.
   * @param required_real_props_a Properties<REAL> object containing information
   * regarding the required REAL-based properties of the first particle for the
   * reaction data.
   * @param required_int_props_b Properties<INT> object containing information
   * regarding the required INT-based properties of the second particle for the
   * reaction data.
   * @param required_real_props_b Properties<REAL> object containing information
   * regarding the required REAL-based properties of the second particle for the
   * reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  PairReactionDataBase(
      Properties<INT> required_int_props_a,
      Properties<REAL> required_real_props_a,
      Properties<INT> required_int_props_b,
      Properties<REAL> required_real_props_b,
      std::map<int, std::string> properties_map = get_default_map())
      : AbstractReactionData<ON_DEVICE_T, PairReactionDataArgumentPack, dim,
                             RNG_KERNEL_T, input_dim>(
            PairReactionDataArgumentPack(
                ArgumentNameSet(required_int_props_a, properties_map),
                ArgumentNameSet(required_real_props_a, properties_map),
                ArgumentNameSet(required_int_props_b, properties_map),
                ArgumentNameSet(required_real_props_b, properties_map)),
            properties_map) {}

  /**
   * \overload
   * @brief Constructor for PairReactionDataBase that sets no required
   * properties.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  PairReactionDataBase(
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase(Properties<INT>(), Properties<REAL>(),
                             Properties<INT>(), Properties<REAL>(),
                             properties_map) {}

  /**
   * \overload
   * @brief Constructor for PairReactionDataBase that sets only required int
   * properties that are the same for both particles.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  PairReactionDataBase(
      Properties<INT> required_int_props,
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase(required_int_props, Properties<REAL>(),
                             required_int_props, Properties<REAL>(),
                             properties_map) {}

  /**
   * \overload
   * @brief Constructor for PairReactionDataBase that sets only required real
   * properties that are the same for both particles.
   *
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  PairReactionDataBase(
      Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase(Properties<INT>(), required_real_props,
                             Properties<INT>(), required_real_props,
                             properties_map) {}

  /**
   * \overload
   * @brief Constructor for PairReactionDataBase that sets required int and
   * real properties, same for both particles.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  PairReactionDataBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase(required_int_props, required_real_props,
                             required_int_props, required_real_props,
                             properties_map) {}
};

/**
 * @brief Base pair reaction data object to be used on SYCL devices.
 *
 * @tparam dim Used to set the size of the array that calc_data returns
 * (Optional).
 * @tparam RNG_KERNEL_T Sets the type of RNG that is used for sampling
 * (Optional).
 * @tparam input_dim The dimension of the optional input array (for use in
 * pipelines) (Optional, default 0)
 * @tparam VALUE_T Return type of this objects calc_data routine (Optional,
 * default REAL)
 * @tparam INPUT_T Input type of array required by this object (if input_dim >0)
 */
template <size_t dim = 1, typename RNG_KERNEL_T = DEFAULT_RNG_KERNEL,
          size_t input_dim = 0, typename VALUE_T = REAL,
          typename INPUT_T = REAL>
struct PairReactionDataBaseOnDevice
    : AbstractReactionDataOnDevice<PairReactionDataAccessors, dim, RNG_KERNEL_T,
                                   input_dim, VALUE_T, INPUT_T> {};

}; // namespace VANTAGE::Reactions
#endif
