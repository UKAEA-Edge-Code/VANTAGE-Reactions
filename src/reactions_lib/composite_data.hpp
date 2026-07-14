#ifndef REACTIONS_COMPOSITE_DATA_H
#define REACTIONS_COMPOSITE_DATA_H
#include "reaction_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device composite data base class
 *
 * @tparam dim Used to set the size of the array that calc_data returns
 * @tparam input_dim The dimension of the optional input array (for use in
 * pipelines)
 * @tparam VAL_TYPE Return type of this objects calc_data routine
 * @tparam IN_TYPE Input type of array required by this object (if input_dim >0)
 * @tparam DATATYPE ReactionDataOnDevice variadic parameters whose calc_data is
 * called from this object
 */

template <size_t dim, size_t input_dim, typename VAL_TYPE, typename IN_TYPE,
          typename... DATATYPE>
struct CompositeDataOnDevice
    : public AbstractReactionDataOnDevice<
          typename std::tuple_element_t<
              0, std::tuple<DATATYPE...>>::ACCESSOR_PACK_TYPE,
          dim, TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>,
          input_dim, VAL_TYPE, IN_TYPE> {

  static_assert(
      (std::is_same_v<typename std::tuple_element_t<
                          0, std::tuple<DATATYPE...>>::ACCESSOR_PACK_TYPE,
                      typename DATATYPE::ACCESSOR_PACK_TYPE> &&
       ...),
      "All contained on-device data objects must use the same "
      "ACCESSOR_PACK_TYPE.");

  CompositeDataOnDevice() = default;

  CompositeDataOnDevice(DATATYPE... data) : data(Tuple::to_tuple(data...)) {};

protected:
  Tuple::Tuple<DATATYPE...> data;
};

template <typename... DATATYPE>
inline std::tuple<typename DATATYPE::ON_DEVICE_OBJ_TYPE...>
get_on_device_objs(std::tuple<DATATYPE...> &data) {

  return std::apply(
      [](auto &&...args) { return std::tuple(args.get_on_device_obj()...); },
      data);
};

/**
 * @brief Composite ReactionData object containing multiple other ReactionData
 * objects.
 *
 * @tparam ON_DEVICE_TYPE Type of the on-device object
 * @tparam dim Used to set the size of the array that calc_data returns
 * @tparam input_dim The dimension of the input array
 * @tparam DATATYPE ReactionData derived types contained within this composite
 * object
 */
template <typename ON_DEVICE_TYPE, size_t dim, size_t input_dim,
          typename... DATATYPE>
struct CompositeData
    : public AbstractReactionData<
          ON_DEVICE_TYPE,
          typename std::tuple_element_t<
              0, std::tuple<DATATYPE...>>::ARGUMENT_PACK_TYPE,
          dim, TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>,
          input_dim> {

  using ARGUMENT_PACK_TYPE = typename std::tuple_element_t<
      0, std::tuple<DATATYPE...>>::ARGUMENT_PACK_TYPE;

  static_assert(
      (std::is_same_v<typename std::tuple_element_t<
                          0, std::tuple<DATATYPE...>>::ARGUMENT_PACK_TYPE,
                      typename DATATYPE::ARGUMENT_PACK_TYPE> &&
       ...),
      "All contained ReactionData objects must use the same "
      "ARGUMENT_PACK_TYPE.");
  /**
   * @brief Constructor for CompositeData
   *
   * @param data Variadic argument with all of the contained ReactionData
   * objects
   */
  CompositeData(DATATYPE... data)
      : AbstractReactionData<
            ON_DEVICE_TYPE, ARGUMENT_PACK_TYPE, dim,
            TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>,
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
        tuple_rng<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>,
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

  std::tuple<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>
  get_rng_kernels_children() {

    return std::apply(
        [](auto &&...args) { return std::tuple(args.get_rng_kernel()...); },
        this->data);
  }

protected:
  std::tuple<DATATYPE...> data;
};

}; // namespace VANTAGE::Reactions
#endif
