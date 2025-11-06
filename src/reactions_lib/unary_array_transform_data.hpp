#ifndef REACTIONS_UNARY_ARRAY_TRANSFORM_DATA_H
#define REACTIONS_UNARY_ARRAY_TRANSFORM_DATA_H
#include "reaction_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Abstract base class encapsulating a unary transformation of a
 * std::array
 */
template <size_t INPUT_DIM, size_t OUTPUT_DIM>
struct AbstractUnaryArrayTransform {

  static const size_t IN_DIM = INPUT_DIM;
  static const size_t OUT_DIM = OUTPUT_DIM;

  virtual std::array<REAL, OUT_DIM>
  apply(const std::array<REAL, IN_DIM> &input) const {};
};

/**
 * @brief On-device reaction data applying a unary array transform to an input
 * array
 *
 * @tparam TRANSFORM Transform derived from AbstractUnaryArrayTransform
 */
template <typename TRANSFORM>
struct UnaryArrayTransformDataOnDevice
    : public ReactionDataBaseOnDevice<TRANSFORM::OUT_DIM, DEFAULT_RNG_KERNEL,
                                      TRANSFORM::IN_DIM> {

  UnaryArrayTransformDataOnDevice() = default;
  /**
   * @brief Constructor of UnaryArrayTransformDataOnDevice
   *
   * @param transform The transform object to be applied
   */
  UnaryArrayTransformDataOnDevice(const TRANSFORM &transform)
      : transform(transform) {};

  /**
   * @brief Return the result of applying the contained transform on the input
   *
   * @param input Input array
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return Result of applying the contained transform
   */
  std::array<REAL, TRANSFORM::OUT_DIM>
  calc_data(const std::array<REAL, TRANSFORM::IN_DIM> &input,
            const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename DEFAULT_RNG_KERNEL::KernelType &kernel) const {

    return this->transform.apply(input);
  }

private:
  TRANSFORM transform;
};

/**
 * @brief Host type for data applying a unary transform on an input array
 *
 * @tparam TRANSFORM The transformation type being applied
 */
template <typename TRANSFORM>
struct UnaryArrayTransformData
    : public ReactionDataBase<UnaryArrayTransformDataOnDevice<TRANSFORM>,
                              TRANSFORM::OUT_DIM, DEFAULT_RNG_KERNEL,
                              TRANSFORM::IN_DIM> {

  /**
   * @brief Constructor for UnaryArrayTransformData
   *
   * @param transform Unary transform object (derived from
   * AbstractUnaryTransform) to be applied on input data
   */
  UnaryArrayTransformData(const TRANSFORM &transform) {
    this->on_device_obj = UnaryArrayTransformDataOnDevice(transform);
  };

  /**
   * @brief No-op since there are no required properties to index
   */
  void index_on_device_object() {};
};
}; // namespace VANTAGE::Reactions
#endif
