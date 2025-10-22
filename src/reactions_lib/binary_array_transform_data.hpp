#ifndef REACTIONS_BINARY_ARRAY_TRANSFORM_DATA_H
#define REACTIONS_BINARY_ARRAY_TRANSFORM_DATA_H
#include "composite_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Abstract base class encapsulating a binary transformation of two
 * std::arrays
 */
template <size_t INPUT_DIM_1, size_t INPUT_DIM_2, size_t OUTPUT_DIM>
struct AbstractBinaryArrayTransform {

  static const size_t IN_DIM_1 = INPUT_DIM_1;
  static const size_t IN_DIM_2 = INPUT_DIM_2;
  static const size_t OUT_DIM = OUTPUT_DIM;

  virtual std::array<REAL, OUT_DIM>
  apply(const std::array<REAL, IN_DIM_1> &input_1,
        const std::array<REAL, IN_DIM_2> &input_2) const {};
};

/**
 * @brief Binary array transform data on device, applying a binary
 * transformation on the outputs of two reaction data objects
 *
 * @tparam TRANSFORM The binary transform type
 * @tparam DATATYPE1 The first (lhs) operand type
 * @tparam DATATYPE2 The second (rhs) operand type
 */
template <typename TRANSFORM, typename DATATYPE1, typename DATATYPE2>
struct BinaryArrayTransformDataOnDevice
    : public CompositeDataOnDevice<TRANSFORM::OUT_DIM, 0, REAL, REAL, DATATYPE1,
                                   DATATYPE2> {

  BinaryArrayTransformDataOnDevice() = default;

  /**
   * @brief BinaryArrayTransformDataOnDevice constructor
   *
   * @param transform Transformation object to be applied to the results of the
   * two contained data objects
   * @param data1 The first (lhs) contained data object
   * @param data2 The second (rhs) contained data object
   */
  BinaryArrayTransformDataOnDevice(TRANSFORM transform, DATATYPE1 data1,
                                   DATATYPE2 data2)
      : CompositeDataOnDevice<TRANSFORM::OUT_DIM, 0, REAL, REAL, DATATYPE1,
                              DATATYPE2>(data1, data2),
        transform(transform) {

    static_assert(
        TRANSFORM::IN_DIM_1 == DATATYPE1::DIM &&
            TRANSFORM::IN_DIM_2 == DATATYPE2::DIM,
        "BinaryArrayTransformDataOnDevice input dimensions do not conform to "
        "between the supplied transform and the contained data objects");
  };

  /**
   * @brief Return the result of applying the binary transform on the results of
   * the two contained objects
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
   * @return The result of applying the transform on the results of the two
   * contained data objects
   */
  std::array<REAL, TRANSFORM::OUT_DIM> calc_data(
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Read<REAL> &req_real_props,
      typename TupleRNG<std::shared_ptr<typename DATATYPE1::RNG_KERNEL_TYPE>,
                        std::shared_ptr<typename DATATYPE2::RNG_KERNEL_TYPE>>::
          KernelType &rng_kernel) const {

    return this->transform.apply(
        Tuple::get<0>(this->data)
            .calc_data(index, req_int_props, req_real_props,
                       rng_kernel.template get<0>()),
        Tuple::get<1>(this->data)
            .calc_data(index, req_int_props, req_real_props,
                       rng_kernel.template get<1>()));
  }

private:
  TRANSFORM transform;
};

/**
 * @brief Composite ReactionData object containing two other ReactionData
 * objects. On calculation of the data, passes the output of the objects to
 * a binary transformation object which is then applied to the two arrays
 *
 * @tparam TRANSFORM The binary transformation object to be applied to the
 * results of the contained data objects
 * @tparam DATATYPE1 The host type of the first (lhs) contained object
 * @tparam DATATYPE2 The host type of the second (rhs) contained object
 */
template <typename TRANSFORM, typename DATATYPE1, typename DATATYPE2>
struct BinaryArrayTransformData
    : public CompositeData<
          BinaryArrayTransformDataOnDevice<
              TRANSFORM, typename DATATYPE1::ON_DEVICE_OBJ_TYPE,
              typename DATATYPE2::ON_DEVICE_OBJ_TYPE>,
          TRANSFORM::OUT_DIM, 0, DATATYPE1, DATATYPE2> {

  /**
   * @brief Constructor for BinaryArrayTransformData
   *
   * @param transform The binary transformation object to be applied to the
   * results of the two data objects
   * @param data1 The first (lhs) data object
   * @param data2 The secon (rhs) data object
   */
  BinaryArrayTransformData(TRANSFORM transform, DATATYPE1 data1,
                           DATATYPE2 data2)
      : CompositeData<BinaryArrayTransformDataOnDevice<
                          TRANSFORM, typename DATATYPE1::ON_DEVICE_OBJ_TYPE,
                          typename DATATYPE2::ON_DEVICE_OBJ_TYPE>,
                      TRANSFORM::OUT_DIM, 0, DATATYPE1, DATATYPE2>(data1,
                                                                   data2),
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
  TRANSFORM transform;
};
}; // namespace VANTAGE::Reactions
#endif
