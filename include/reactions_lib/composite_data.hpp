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
 * @tparam VAL_TYPE Return type of this objects calc_data routine
 * @tparam IN_TYPE Input type of array required by this object (if input_dim >0)
 * @tparam DATATYPE ReactionDataOnDevice variadic parameters whose calc_data is
 * called from this object
 */

template <size_t dim, size_t input_dim, typename VAL_TYPE, typename IN_TYPE,
          typename... DATATYPE>
struct CompositeDataOnDevice
    : public ReactionDataBaseOnDevice<
          dim,
          NP::TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>,
          input_dim, VAL_TYPE, IN_TYPE> {

  CompositeDataOnDevice() = default;

  /**
   * @brief Constructor for CompositeDataOnDevice.
   *
   * @param data Variadic argument with all of the contained
   * ReactionDataOnDevice objects
   */
  CompositeDataOnDevice(DATATYPE... data)
      : data(NP::Tuple::to_tuple(data...)) {};

protected:
  NP::Tuple::Tuple<DATATYPE...> data;
};

/**
 * @brief Getter that returns tuple of ReactionDataOnDevice objects associated
 * with each DATATYPE object in data.
 */
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
    : public ReactionDataBase<
          ON_DEVICE_TYPE, dim,
          NP::TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>,
          input_dim> {

  /**
   * @brief Constructor for CompositeData
   *
   * @param data Variadic argument with all of the contained ReactionData
   * objects
   */
  CompositeData(DATATYPE... data) : data(std::make_tuple(data...)) {};

  /**
   * @brief To be called by derived class constructors to access virtual
   * index_on_device_object() table
   */
  void post_init() {

    this->set_required_int_props(this->get_required_int_props_children());
    this->set_required_real_props(this->get_required_real_props_children());
    this->set_rng_kernel(std::apply(
        NP::tuple_rng<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>,
        this->get_rng_kernels_children()));
  }

  /**
   * @brief To be implemented by each derived class in order to handle required
   * property indexing on the on-device object
   */
  virtual void index_on_device_object() {};

  std::tuple<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>
  get_rng_kernels_children() {

    return std::apply(
        [](auto &&...args) { return std::tuple(args.get_rng_kernel()...); },
        this->data);
  }

  ArgumentNameSet<REAL> get_required_real_props_children() {

    auto new_set = ArgumentNameSet<REAL>();

    std::apply(
        [&](auto &&...args) {
          ((new_set = new_set.merge_with(args.get_required_real_props())), ...);
        },
        this->data);

    return new_set;
  }

  /**
   * @brief Getter for the merged set of required integer properties of all of
   * the contained ReactionData objects.
   */
  ArgumentNameSet<INT> get_required_int_props_children() {

    auto new_set = ArgumentNameSet<INT>();

    std::apply(
        [&](auto &&...args) {
          ((new_set = new_set.merge_with(args.get_required_int_props())), ...);
        },
        this->data);

    return new_set;
  }

  /**
   * @brief Setter of the required integer properties for all of the contained
   * ReactionData objects.
   */
  void set_required_int_props(const ArgumentNameSet<INT> &props) {
    this->required_int_props = props;
    std::apply(
        [&](auto &&...args) { ((args.set_required_int_props(props)), ...); },
        this->data);
    this->index_on_device_object();
  }

  /**
   * @brief Setter of the required REAL properties for all of the contained
   * ReactionData objects.
   */
  void set_required_real_props(const ArgumentNameSet<REAL> &props) {
    this->required_real_props = props;
    std::apply(
        [&](auto &&...args) { ((args.set_required_real_props(props)), ...); },
        this->data);
    this->index_on_device_object();
  }

protected:
  std::tuple<DATATYPE...> data;
};

}; // namespace VANTAGE::Reactions
#endif
