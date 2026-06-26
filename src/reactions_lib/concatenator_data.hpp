#ifndef REACTIONS_CONCATENATOR_DATA_H
#define REACTIONS_CONCATENATOR_DATA_H
#include "composite_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

template <typename T> constexpr size_t total_dim() { return T::DIM; };
template <typename T, typename U, typename... DATATYPE>
constexpr size_t total_dim() {
  return T::DIM + total_dim<U, DATATYPE...>();
};

/**
 * @brief On device recursive concatenator data - calc_data returns the
 * concatenated result of all contained ReactionDataOnDevice objects
 *
 * @tparam DATATYPE ReactionDataOnDevice variadic parameters whose calc_data is
 * called from this object
 */
template <typename... DATATYPE>
struct ConcatenatorDataOnDevice
    : public CompositeDataOnDevice<total_dim<DATATYPE...>(), 0, REAL, REAL,
                                   DATATYPE...> {

  ConcatenatorDataOnDevice() = default;
  ConcatenatorDataOnDevice(DATATYPE... data)
      : CompositeDataOnDevice<total_dim<DATATYPE...>(), 0, REAL, REAL,
                              DATATYPE...>(data...) {};

  static const size_t DIM = total_dim<DATATYPE...>();

  /**
   * @brief Function to calculate the concatenated data
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

    std::array<REAL, DIM> result;

    calc_data_recurse<0>(index, req_int_props, req_real_props, rng_kernel,
                         result, 0);
    return result;
  }

  template <std::size_t I>
  void calc_data_recurse(
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Read<REAL> &req_real_props,
      typename TupleRNG<
          std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>...>::KernelType
          &rng_kernel,
      std::array<REAL, DIM> &result, size_t dat_dim_idx) const {
    if constexpr (I < (sizeof...(DATATYPE))) {

      const auto arg = Tuple::get<I>(this->data);
      constexpr auto data_dim = decltype(arg)::DIM;
      std::array<REAL, data_dim> calculated_data = arg.calc_data(
          index, req_int_props, req_real_props, rng_kernel.template get<I>());
      for (auto i = 0; i < data_dim; i++) {
        result[dat_dim_idx + i] = calculated_data[i];
      };

      this->calc_data_recurse<I + 1>(index, req_int_props, req_real_props,
                                     rng_kernel, result,
                                     dat_dim_idx + data_dim);
    };
  }
};

/**
 * @brief Composite ReactionData object constaining multiple other ReactionData
 * objects. On calculation of the data, returns the concatenated (in the
 * template order) results of the contained data objects.
 *
 * @tparam DATATYPE ReactionData derived types contained within this composite
 * object
 */
template <typename... DATATYPE>
struct ConcatenatorData
    : public CompositeData<
          ConcatenatorDataOnDevice<typename DATATYPE::ON_DEVICE_OBJ_TYPE...>,
          total_dim<DATATYPE...>(), 0, DATATYPE...> {

  /**
   * @brief Constructor for ConcatenatorData
   *
   * @param data Variadic argument with all of the contained ReactionData
   * objects
   */
  ConcatenatorData(DATATYPE... data)
      : CompositeData<
            ConcatenatorDataOnDevice<typename DATATYPE::ON_DEVICE_OBJ_TYPE...>,
            total_dim<DATATYPE...>(), 0, DATATYPE...>(data...) {
    this->post_init();
  };

  /**
   * @brief Reconstruct the composite on-device object (assuming the individual
   * on-device objects have been modified/re-indexed)
   */
  void index_on_device_object() {

    this->on_device_obj = std::make_from_tuple<
        ConcatenatorDataOnDevice<typename DATATYPE::ON_DEVICE_OBJ_TYPE...>>(
        get_on_device_objs(this->data));
  };
};
}; // namespace VANTAGE::Reactions
#endif
