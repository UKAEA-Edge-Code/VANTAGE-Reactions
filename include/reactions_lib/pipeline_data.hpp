#ifndef REACTIONS_PIPELINE_DATA_H
#define REACTIONS_PIPELINE_DATA_H
#include "composite_data.hpp"
#include "reaction_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Recursive helper function to retrieve the DIM-value of
 * the last ReactionData (or ReactionDataOnDevice) object in DATA_T
 * (specifically the size of the output of its on-device calc_data
 * function).
 *
 * Use within PipelineData or PipelineDataOnDevice as: last_dim<DATA_T...>();
 */
template <typename T> constexpr size_t last_dim() { return T::DIM; };
template <typename T, typename U, typename... DATA_T>
constexpr size_t last_dim() {
  return last_dim<U, DATA_T...>();
};

/**
 * @brief Helper function to retrieve the INPUT_DIM-value of the first
 * ReactionData (or ReactionDataOnDevice) object in DATA_T
 *
 * Use within PipelineData or PipelineDataOnDevice as:
 * first_in_dim<DATA_T...>();
 */
template <typename T, typename... DATA_T> constexpr size_t first_in_dim() {
  return T::INPUT_DIM;
};

/**
 * @brief Helper function to check that the INPUT_DIM-value of a given
 * ReactionData object in DATA_T equals the DIM-value of
 * the previous ReactionData (or ReactionDataOnDevice) object.
 *
 * Use within PipelineData or PipelineDataOnDevice as:
 * check_consistency<DATA_T...>();
 */
template <typename T> constexpr bool check_consistency() { return true; };
template <typename T, typename U, typename... DATA_T>
constexpr bool check_consistency() {
  return (U::INPUT_DIM == T::DIM &&
          std::is_same<typename U::INPUT_TYPE,
                       typename T::VALUE_TYPE>::value) &&
         check_consistency<U, DATA_T...>();
};

/**
 * @brief On device recursive pipeline data - calc_data returns the
 * composition of all contained ReactionDataOnDevice objects, passing on the
 * output from left to right
 *
 * @tparam DATA_T ReactionDataOnDevice variadic parameters whose calc_data is
 * called from this object
 */
template <typename... DATA_T>
struct PipelineDataOnDevice
    : public CompositeDataOnDevice<last_dim<DATA_T...>(), 0, REAL, REAL,
                                   DATA_T...> {

  PipelineDataOnDevice() = default;

  /**
   * @brief Constructor for PipelineDataOnDevice.
   *
   * @param data Variadic argument with all of the contained
   * ReactionDataOnDevice objects
   */
  PipelineDataOnDevice(DATA_T... data)
      : CompositeDataOnDevice<last_dim<DATA_T...>(), 0, REAL, REAL, DATA_T...>(
            data...) {

    static_assert(first_in_dim<DATA_T...>() == 0 &&
                      check_consistency<DATA_T...>(),
                  "Inconsistent input/output dimensions in pipeline data");
  };

  static const size_t DIM = last_dim<DATA_T...>();

  /**
   * @brief Function to calculate the composed data
   *
   * @param accessors Bundled accessors for the ParticleLoop.
   * @param rng_kernel The random number generator kernels used in the
   * calculation, a TupleRNG accessor
   *
   * @return A return array containing the result of the calc_data of the last
   * ReactionDataOnDevice object in DATA_T.
   */
  std::array<REAL, DIM> calc_data(
      const typename CompositeDataOnDevice<last_dim<DATA_T...>(), 0, REAL, REAL,
                                           DATA_T...>::ACCESSOR_PACK_TYPE
          &accessors,
      typename NP::TupleRNG<
          std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>::KernelType
          &rng_kernel) const {

    return calc_data_recurse<0, DATA_T...>(std::array<REAL, 0>{}, accessors,
                                           rng_kernel);
  }

  template <size_t index, typename T, typename... Args>
  std::array<REAL, DIM> calc_data_recurse(
      const std::array<typename T::INPUT_TYPE, T::INPUT_DIM> input,
      const typename CompositeDataOnDevice<last_dim<DATA_T...>(), 0, REAL, REAL,
                                           DATA_T...>::ACCESSOR_PACK_TYPE
          &accessors,
      typename NP::TupleRNG<
          std::shared_ptr<typename DATA_T::RNG_KERNEL_TYPE>...>::KernelType
          &rng_kernel) const {

    const auto arg = NP::Tuple::get<index>(this->data);
    if constexpr (index < (sizeof...(DATA_T)) - 1) {
      if constexpr (T::INPUT_DIM > 0) {
        return this->calc_data_recurse<index + 1, Args...>(
            arg.calc_data(input, accessors, rng_kernel.template get<index>()),
            accessors, rng_kernel);
      } else {

        return this->calc_data_recurse<index + 1, Args...>(
            arg.calc_data(accessors, rng_kernel.template get<index>()),
            accessors, rng_kernel);
      }
    } else {

      if constexpr (T::INPUT_DIM > 0) {
        return arg.calc_data(input, accessors,
                             rng_kernel.template get<index>());
      } else {

        return arg.calc_data(accessors, rng_kernel.template get<index>());
      }
    }
  }
};

/**
 * @brief Composite ReactionData object containing multiple other ReactionData
 * objects. On calculation of the data, passes the output of each data object to
 * the next in the template order, returning the final result.
 *
 * @tparam DATA_T ReactionData derived types contained within this composite
 * object
 */
template <typename... DATA_T>
struct PipelineData
    : public CompositeData<
          PipelineDataOnDevice<typename DATA_T::ON_DEVICE_OBJ_TYPE...>,
          last_dim<DATA_T...>(), 0, DATA_T...> {

  /**
   * @brief Constructor for PipelineData
   *
   * @param data Variadic argument with all of the contained ReactionData
   * objects
   */
  PipelineData(DATA_T... data)
      : CompositeData<
            PipelineDataOnDevice<typename DATA_T::ON_DEVICE_OBJ_TYPE...>,
            last_dim<DATA_T...>(), 0, DATA_T...>(data...) {
    this->post_init();
  };

  /**
   * @brief Reconstruct the composite on-device object (assuming the individual
   * on-device objects have been modified/re-indexed)
   */
  void index_on_device_object() override {

    this->on_device_obj = std::make_from_tuple<
        PipelineDataOnDevice<typename DATA_T::ON_DEVICE_OBJ_TYPE...>>(
        get_on_device_objs(this->data));
  };
};

/**
 * @brief Helper function to construct a PipelineData object.
 *
 * @param data Variadic argument with all of the contained ReactionData
 * objects
 *
 * @return PipelineData object.
 */
template <typename... DATA_T> inline auto pipe(DATA_T... data) {

  return PipelineData(data...);
}
}; // namespace VANTAGE::Reactions
#endif
