#ifndef REACTIONS_CONCATENATOR_DATA_H
#define REACTIONS_CONCATENATOR_DATA_H
#include "composite_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Recursive helper function to accumulate the DIM-values of
 * each ReactionData (or ReactionDataOnDevice) object in DATA_T (specifically
 * the size of the output of their on-device calc_data functions).
 *
 * Use within ConcatenatorData or ConcatenatorDataOnDevice as:
 * total_dim<DATA_T...>();
 */
template <typename T> constexpr size_t total_dim() { return T::DIM; };
template <typename T, typename U, typename... DATA_T>
constexpr size_t total_dim() {
  return T::DIM + total_dim<U, DATA_T...>();
};

/**
 * @brief On device recursive concatenator data - calc_data returns the
 * concatenated result of all contained ReactionDataOnDevice objects
 *
 * @tparam DATA_T ReactionDataOnDevice variadic parameters whose calc_data is
 * called from this object
 */
template <typename... DATA_T>
struct ConcatenatorDataOnDevice
    : public CompositeDataOnDevice<total_dim<DATA_T...>(), 0, REAL, REAL,
                                   DATA_T...> {

  ConcatenatorDataOnDevice() = default;

  /**
   * @brief Constructor for ConcatenatorDataOnDevice.
   *
   * @param data Variadic argument with all of the contained
   * ReactionDataOnDevice objects
   */
  ConcatenatorDataOnDevice(DATA_T... data)
      : CompositeDataOnDevice<total_dim<DATA_T...>(), 0, REAL, REAL, DATA_T...>(
            data...) {};

  static const size_t DIM = total_dim<DATA_T...>();

  /**
   * @brief Function to calculate the concatenated data
   *
   * @param accessors Bundled accessors for the ParticleLoop.
   * @param rng_kernel The random number generator kernels used in the
   * calculation
   *
   * @return Concatenated return arrays of all the contained device types
   */
  std::array<REAL, DIM> calc_data(
      const typename CompositeDataOnDevice<total_dim<DATA_T...>(), 0, REAL,
                                           REAL, DATA_T...>::ACCESSOR_PACK_TYPE
          &accessors,
      typename NP::TupleRNG<
          std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>::KernelType
          &rng_kernel) const {

    std::array<REAL, DIM> result;

    calc_data_recurse<0>(accessors, rng_kernel, result, 0);
    return result;
  }

  template <std::size_t index>
  void calc_data_recurse(
      const typename CompositeDataOnDevice<total_dim<DATA_T...>(), 0, REAL,
                                           REAL, DATA_T...>::ACCESSOR_PACK_TYPE
          &accessors,
      typename NP::TupleRNG<
          std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>::KernelType
          &rng_kernel,
      std::array<REAL, DIM> &result, size_t dat_dim_idx) const {
    if constexpr (index < (sizeof...(DATA_T))) {

      const auto arg = NP::Tuple::get<index>(this->data);
      constexpr auto data_dim = decltype(arg)::DIM;
      std::array<REAL, data_dim> calculated_data =
          arg.calc_data(accessors, rng_kernel.template get<index>());
      for (auto i = 0; i < data_dim; i++) {
        result[dat_dim_idx + i] = calculated_data[i];
      };

      this->calc_data_recurse<index + 1>(accessors, rng_kernel, result,
                                         dat_dim_idx + data_dim);
    };
  }
};

/**
 * @brief Composite ReactionData object constaining multiple other ReactionData
 * objects. On calculation of the data, returns the concatenated (in the
 * template order) results of the contained data objects.
 *
 * @tparam DATA_T ReactionData derived types contained within this composite
 * object
 */
template <typename... DATA_T>
struct ConcatenatorData
    : public CompositeData<
          ConcatenatorDataOnDevice<typename DATA_T::ON_DEVICE_OBJ_TYPE...>,
          total_dim<DATA_T...>(), 0, DATA_T...> {

  /**
   * @brief Constructor for ConcatenatorData
   *
   * @param data Variadic argument with all of the contained ReactionData
   * objects
   */
  ConcatenatorData(DATA_T... data)
      : CompositeData<
            ConcatenatorDataOnDevice<typename DATA_T::ON_DEVICE_OBJ_TYPE...>,
            total_dim<DATA_T...>(), 0, DATA_T...>(data...) {
    this->post_init();
  };

  /**
   * @brief Reconstruct the composite on-device object (assuming the individual
   * on-device objects have been modified/re-indexed)
   */
  void index_on_device_object() {

    this->on_device_obj = std::make_from_tuple<
        ConcatenatorDataOnDevice<typename DATA_T::ON_DEVICE_OBJ_TYPE...>>(
        get_on_device_objs(this->data));
  };
};
}; // namespace VANTAGE::Reactions
#endif
