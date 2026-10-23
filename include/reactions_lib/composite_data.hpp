#ifndef REACTIONS_COMPOSITE_DATA_H
#define REACTIONS_COMPOSITE_DATA_H
#include "reaction_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief On device composite data base class
 *
 * @tparam dim Used to set the size of the array that calc_data returns
 * @tparam input_dim The dimension of the optional input array (for use in
 * pipelines)
 * @tparam VALUE_T Return type of this objects calc_data routine
 * @tparam INPUT_T Input type of array required by this object (if input_dim >0)
 * @tparam DATA_T ReactionDataOnDevice variadic parameters whose calc_data is
 * called from this object
 */

template <size_t dim, size_t input_dim, typename VALUE_T, typename INPUT_T,
          typename... DATA_T>
struct CompositeDataOnDevice
    : public AbstractReactionDataOnDevice<
          typename std::tuple_element_t<
              0, std::tuple<DATA_T...>>::ACCESSOR_PACK_TYPE,
          dim,
          NP::TupleRNG<std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>,
          input_dim, VALUE_T, INPUT_T> {

  static_assert(
      (std::is_same_v<typename std::tuple_element_t<
                          0, std::tuple<DATA_T...>>::ACCESSOR_PACK_TYPE,
                      typename DATA_T::ACCESSOR_PACK_TYPE> &&
       ...),
      "All contained on-device data objects must use the same "
      "ACCESSOR_PACK_TYPE.");

  CompositeDataOnDevice() = default;

  /**
   * @brief Constructor for CompositeDataOnDevice.
   *
   * @param data Variadic argument with all of the contained
   * ReactionDataOnDevice objects
   */
  CompositeDataOnDevice(DATA_T... data) : data(NP::Tuple::to_tuple(data...)) {};

protected:
  NP::Tuple::Tuple<DATA_T...> data;
};

/**
 * @brief Getter that returns tuple of ReactionDataOnDevice objects associated
 * with each DATA_T object in data.
 */
template <typename... DATA_T>
inline std::tuple<typename DATA_T::ON_DEVICE_OBJ_TYPE...>
get_on_device_objs(std::tuple<DATA_T...> &data) {

  return std::apply(
      [](auto &&...args) { return std::tuple(args.get_on_device_obj()...); },
      data);
};

/**
 * @brief Composite ReactionData object containing multiple other ReactionData
 * objects.
 *
 * @tparam ON_DEVICE_T Type of the on-device object
 * @tparam dim Used to set the size of the array that calc_data returns
 * @tparam input_dim The dimension of the input array
 * @tparam DATA_T ReactionData derived types contained within this composite
 * object
 */
template <typename ON_DEVICE_T, size_t dim, size_t input_dim,
          typename... DATA_T>
struct CompositeData
    : public AbstractReactionData<
          ON_DEVICE_T,
          typename std::tuple_element_t<
              0, std::tuple<DATA_T...>>::ARGUMENT_PACK_TYPE,
          dim,
          NP::TupleRNG<std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>,
          input_dim> {

  using ARGUMENT_PACK_TYPE =
      typename std::tuple_element_t<0,
                                    std::tuple<DATA_T...>>::ARGUMENT_PACK_TYPE;

  static_assert(
      (std::is_same_v<typename std::tuple_element_t<
                          0, std::tuple<DATA_T...>>::ARGUMENT_PACK_TYPE,
                      typename DATA_T::ARGUMENT_PACK_TYPE> &&
       ...),
      "All contained ReactionData objects must use the same "
      "ARGUMENT_PACK_TYPE.");
  /**
   * @brief Constructor for CompositeData
   *
   * @param data Variadic argument with all of the contained ReactionData
   * objects
   */
  CompositeData(DATA_T... data)
      : AbstractReactionData<
            ON_DEVICE_T, ARGUMENT_PACK_TYPE, dim,
            NP::TupleRNG<std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>,
            input_dim>(ARGUMENT_PACK_TYPE(), get_default_map()),
        data(std::make_tuple(data...)) {};

  /**
   * @brief To be called by derived class constructors to set up the merged
   * argument pack and RNG kernel from the contained child data objects.
   */
  void post_init() {

    ARGUMENT_PACK_TYPE merged_pack = this->get_arg_pack();
    std::apply(
        [&](auto &&...args) {
          ((merged_pack = merged_pack.merge_with(args.get_arg_pack())), ...);
        },
        this->data);

    // Propagate merged pack to each child (also re-indexes children)
    std::apply([&](auto &&...args) { ((args.set_arg_pack(merged_pack)), ...); },
               this->data);

    this->argument_pack = merged_pack;

    this->index_on_device_object();

    this->set_rng_kernel(std::apply(
        NP::tuple_rng<std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>,
        this->get_rng_kernels_children()));
  }

  /**
   * @brief To be implemented by each derived class in order to handle required
   * property indexing on the on-device object
   */
  virtual void index_on_device_object() {};

  void set_arg_pack(const ARGUMENT_PACK_TYPE &argument_pack) {
    this->argument_pack = argument_pack;
    this->post_init();
  }

  std::tuple<std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>
  get_rng_kernels_children() {

    return std::apply(
        [](auto &&...args) { return std::tuple(args.get_rng_kernel()...); },
        this->data);
  }

protected:
  std::tuple<DATA_T...> data;
};

}; // namespace VANTAGE::Reactions
#endif
