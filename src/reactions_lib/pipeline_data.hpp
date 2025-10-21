#ifndef REACTIONS_PIPELINE_DATA_H
#define REACTIONS_PIPELINE_DATA_H
#include "reaction_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

template <typename T> constexpr size_t last_dim() { return T::DIM; };
template <typename T, typename U, typename... DATATYPE>
constexpr size_t last_dim() {
  return last_dim<U, DATATYPE...>();
};

template <typename T, typename... DATATYPE> constexpr size_t first_in_dim() {
  return T::INPUT_DIM;
};
template <typename T> constexpr bool check_consistency() { return true; };
template <typename T, typename U, typename... DATATYPE>
constexpr bool check_consistency() {
  return (U::INPUT_DIM == T::DIM) && check_consistency<U, DATATYPE...>();
};

/**
 * @brief On device recursive pipeline data - calc_data returns the
 * composition of all contained ReactionDataOnDevice objects, passing on the
 * output from left to right
 *
 * @tparam DATATYPE ReactionDataOnDevice variadic parameters whose calc_data is
 * called from this object
 */
template <typename... DATATYPE>
struct PipelineDataOnDevice
    : public ReactionDataBaseOnDevice<
          last_dim<DATATYPE...>(),
          TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>> {

  PipelineDataOnDevice(DATATYPE... data) : data(Tuple::to_tuple(data...)) {

    static_assert(first_in_dim<DATATYPE...>() == 0 &&
                      check_consistency<DATATYPE...>(),
                  "Inconsistent input/output dimensions in pipeline data");
  };

  static const size_t DIM = last_dim<DATATYPE...>();

  /**
   * @brief Function to calculate the composed data
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties
   * that need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernels used in the
   * calculation, a TupleRNG accessor
   *
   * @return Concatenated return arrays of all the contained device types
   */
  std::array<REAL, DIM> calc_data(
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Read<REAL> &req_real_props,
      typename TupleRNG<
          std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>::KernelType
          &rng_kernel) const {

    return calc_data_recurse<0, DATATYPE...>(std::array<REAL, 0>{}, index,
                                             req_int_props, req_real_props,
                                             rng_kernel);
  }

  template <size_t I, typename T, typename... ARGS>
  std::array<REAL, DIM> calc_data_recurse(
      const std::array<REAL, T::INPUT_DIM> input,
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Read<REAL> &req_real_props,
      typename TupleRNG<
          std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>::KernelType
          &rng_kernel) const {

    const auto arg = Tuple::get<I>(this->data);
    if constexpr (I < (sizeof...(DATATYPE)) - 1) {
      if constexpr (T::INPUT_DIM > 0) {
        return this->calc_data_recurse<I + 1, ARGS...>(
            arg.calc_data(input, index, req_int_props, req_real_props,
                          rng_kernel.template get<I>()),
            index, req_int_props, req_real_props, rng_kernel);
      } else {

        return this->calc_data_recurse<I + 1, ARGS...>(
            arg.calc_data(index, req_int_props, req_real_props,
                          rng_kernel.template get<I>()),
            index, req_int_props, req_real_props, rng_kernel);
      }
    } else {

      if constexpr (T::INPUT_DIM > 0) {
        return arg.calc_data(input, index, req_int_props, req_real_props,
                             rng_kernel.template get<I>());
      } else {

        return arg.calc_data(index, req_int_props, req_real_props,
                             rng_kernel.template get<I>());
      }
    }
  }

private:
  Tuple::Tuple<DATATYPE...> data;
};

/**
 * @brief Composite ReactionData object containing multiple other ReactionData
 * objects. On calculation of the data, passes the output of each data object to
 * the next in the template order, returning the final result.
 *
 * @tparam DATATYPE ReactionData derived types contained within this composite
 * object
 */
template <typename... DATATYPE>
struct PipelineData
    : public ReactionDataBase<
          PipelineDataOnDevice<typename DATATYPE::ON_DEVICE_OBJ_TYPE...>,
          last_dim<DATATYPE...>(),
          TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>> {

  /**
   * @brief Constructor for PipelineData
   *
   * @param data Variadic argument with all of the contained ReactionData
   * objects
   */
  PipelineData(DATATYPE... data) : data(std::make_tuple(data...)) {
    this->set_required_int_props(this->get_required_int_props_children());
    this->set_required_real_props(this->get_required_real_props_children());
    this->set_rng_kernel(std::apply(
        tuple_rng<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>,
        this->get_rng_kernels_children()));
  };

  /**
   * @brief Reconstruct the composite on-device object (assuming the individual
   * on-device objects have been modified/re-indexed)
   */
  void index_on_device_object() {

    this->on_device_obj = std::make_from_tuple<
        PipelineDataOnDevice<typename DATATYPE::ON_DEVICE_OBJ_TYPE...>>(
        get_on_device_objs(this->data));
  };

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

  ArgumentNameSet<INT> get_required_int_props_children() {

    auto new_set = ArgumentNameSet<INT>();

    std::apply(
        [&](auto &&...args) {
          ((new_set = new_set.merge_with(args.get_required_int_props())), ...);
        },
        this->data);

    return new_set;
  }

  void set_required_int_props(const ArgumentNameSet<INT> &props) {
    this->required_int_props = props;
    std::apply(
        [&](auto &&...args) { ((args.set_required_int_props(props)), ...); },
        this->data);
    this->index_on_device_object();
  }

  void set_required_real_props(const ArgumentNameSet<REAL> &props) {
    this->required_real_props = props;
    std::apply(
        [&](auto &&...args) { ((args.set_required_real_props(props)), ...); },
        this->data);
    this->index_on_device_object();
  }

private:
  std::tuple<DATATYPE...> data;
};

template <typename... DATATYPE> inline auto pipe(DATATYPE... data) {

  return PipelineData(data...);
}
}; // namespace VANTAGE::Reactions
#endif
