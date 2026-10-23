#ifndef REACTIONS_BINARY_ARRAY_TRANSFORM_DATA_H
#define REACTIONS_BINARY_ARRAY_TRANSFORM_DATA_H
#include "composite_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Abstract base class encapsulating a binary transformation of two
 * std::arrays
 *
 * @tparam input_dim_1 Size of the first (lhs) array
 * @tparam input_dim_2 Size of the second (lhs) array
 * @tparam output_dim Size of the output (rhs) array
 */
template <size_t input_dim_1, size_t input_dim_2, size_t output_dim>
struct AbstractBinaryArrayTransform {

  static const size_t IN_DIM_1 = input_dim_1;
  static const size_t IN_DIM_2 = input_dim_2;
  static const size_t OUT_DIM = output_dim;

  /**
   * @brief Function to apply a binary transform to two arrays.
   *
   * @input_1 First (lhs) REAL-valued array
   * @input_2 Second (lhs) REAL-valued array
   *
   * @return REAL-valued array of size OUT_DIM that's the result of the binary
   * transformation on the two input arrays.
   */
  std::array<REAL, OUT_DIM>
  apply(const std::array<REAL, IN_DIM_1> &input_1,
        const std::array<REAL, IN_DIM_2> &input_2) const {
    return {};
  };
};

/**
 * @brief Binary array transform data on device, applying a binary
 * transformation on the outputs of two reaction data objects
 *
 * @tparam TRANSFORM_T The binary transform type
 * @tparam DATA1_T The first (lhs) operand type
 * @tparam DATA2_T The second (rhs) operand type
 */
template <typename TRANSFORM_T, typename DATA1_T, typename DATA2_T>
struct BinaryArrayTransformDataOnDevice
    : public CompositeDataOnDevice<TRANSFORM_T::OUT_DIM, 0, REAL, REAL, DATA1_T,
                                   DATA2_T> {

  BinaryArrayTransformDataOnDevice() = default;

  /**
   * @brief BinaryArrayTransformDataOnDevice constructor
   *
   * @param transform Transformation object to be applied to the results of the
   * two contained data objects
   * @param data1 The first (lhs) contained data object
   * @param data2 The second (rhs) contained data object
   */
  BinaryArrayTransformDataOnDevice(TRANSFORM_T transform, DATA1_T data1,
                                   DATA2_T data2)
      : CompositeDataOnDevice<TRANSFORM_T::OUT_DIM, 0, REAL, REAL, DATA1_T,
                              DATA2_T>(data1, data2),
        transform(transform) {

    static_assert(
        TRANSFORM_T::IN_DIM_1 == DATA1_T::DIM &&
            TRANSFORM_T::IN_DIM_2 == DATA2_T::DIM,
        "BinaryArrayTransformDataOnDevice input dimensions do not conform "
        "between the supplied transform and the contained data objects");
  };

  /**
   * @brief Return the result of applying the binary transform on the results of
   * the two contained objects
   *
   * @param accessors Bundled accessors for the ParticleLoop.
   * @param kernel The random number generator kernels used in the
   * calculation, a NP::TupleRNG accessor
   *
   * @return The result of applying the transform on the results of the two
   * contained data objects. REAL-valued array of size TRANSFORM_T::OUT_DIM
   */
  std::array<REAL, TRANSFORM_T::OUT_DIM> calc_data(
      const typename CompositeDataOnDevice<TRANSFORM_T::OUT_DIM, 0, REAL, REAL,
                                           DATA1_T, DATA2_T>::ACCESSOR_PACK_TYPE
          &accessors,
      typename NP::TupleRNG<
          std::shared_ptr<typename DATA1_T::RNG_KERNEL_TYPE>,
          std::shared_ptr<typename DATA2_T::RNG_KERNEL_TYPE>>::KernelType
          &rng_kernel) const {

    return this->transform.apply(
        NP::Tuple::get<0>(this->data)
            .calc_data(accessors, rng_kernel.template get<0>()),
        NP::Tuple::get<1>(this->data)
            .calc_data(accessors, rng_kernel.template get<1>()));
  }

private:
  TRANSFORM_T transform;
};

/**
 * @brief Composite ReactionData object containing two other ReactionData
 * objects. On calculation of the data, passes the output of the objects to
 * a binary transformation object which is then applied to the two arrays
 *
 * @tparam TRANSFORM_T The binary transformation object to be applied to the
 * results of the contained data objects
 * @tparam DATA1_T The host type of the first (lhs) contained object
 * @tparam DATA2_T The host type of the second (rhs) contained object
 */
template <typename TRANSFORM_T, typename DATA1_T, typename DATA2_T>
struct BinaryArrayTransformData
    : public CompositeData<
          BinaryArrayTransformDataOnDevice<
              TRANSFORM_T, typename DATA1_T::ON_DEVICE_OBJ_TYPE,
              typename DATA2_T::ON_DEVICE_OBJ_TYPE>,
          TRANSFORM_T::OUT_DIM, 0, DATA1_T, DATA2_T> {

  /**
   * @brief Constructor for BinaryArrayTransformData
   *
   * @param transform The binary transformation object to be applied to the
   * results of the two data objects
   * @param data1 The first (lhs) data object
   * @param data2 The secon (rhs) data object
   */
  BinaryArrayTransformData(TRANSFORM_T transform, DATA1_T data1, DATA2_T data2)
      : CompositeData<BinaryArrayTransformDataOnDevice<
                          TRANSFORM_T, typename DATA1_T::ON_DEVICE_OBJ_TYPE,
                          typename DATA2_T::ON_DEVICE_OBJ_TYPE>,
                      TRANSFORM_T::OUT_DIM, 0, DATA1_T, DATA2_T>(data1, data2),
        transform(transform) {
    this->post_init();
  };

  /**
   * @brief Reconstruct the composite on-device object (assuming the individual
   * on-device objects have been modified/re-indexed)
   */
  void index_on_device_object() {

    this->on_device_obj = BinaryArrayTransformDataOnDevice(
        this->transform, std::get<0>(this->data).get_on_device_obj(),
        std::get<1>(this->data).get_on_device_obj());
  };

private:
  TRANSFORM_T transform;
};
}; // namespace VANTAGE::Reactions
#endif
