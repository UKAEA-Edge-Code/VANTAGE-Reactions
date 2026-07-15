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

  std::array<REAL, OUT_DIM> apply(const std::array<REAL, IN_DIM> &input) const {
    return {};
  };
};

/**
 * @brief On-device reaction data applying a unary array transform to an input
 * array
 *
 * The unary transform is a pure transform stage: it applies the transform to
 * its input array and does not itself read any particle properties, so the
 * argument/accessor pack is only a type-level compatibility tag required by the
 * enclosing pipeline. The accessor pack is therefore a template parameter
 * (defaulting to SingleReactionDataAccessors) rather than being derived from
 * child data objects, allowing the same stage to be used inside either a single
 * or a pair pipeline.
 *
 * @tparam TRANSFORM Transform derived from AbstractUnaryArrayTransform
 * @tparam ACCESSOR_PACK_T Bundled device accessors for the ParticleLoop inside
 * which calc_data is called (Optional, defaults to
 * SingleReactionDataAccessors).
 */
template <typename TRANSFORM,
          typename ACCESSOR_PACK_T = SingleReactionDataAccessors>
struct UnaryArrayTransformDataOnDevice
    : public AbstractReactionDataOnDevice<ACCESSOR_PACK_T, TRANSFORM::OUT_DIM,
                                          DEFAULT_RNG_KERNEL, TRANSFORM::IN_DIM,
                                          REAL, REAL> {

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
   * @param accessors Bundled accessors for the ParticleLoop. Unused by the
   * transform itself but carried so the signature matches the accessor pack
   * expected by the enclosing pipeline.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return Result of applying the contained transform
   */
  std::array<REAL, TRANSFORM::OUT_DIM>
  calc_data(const std::array<REAL, TRANSFORM::IN_DIM> &input,
            const ACCESSOR_PACK_T &accessors,
            typename DEFAULT_RNG_KERNEL::KernelType &kernel) const {

    return this->transform.apply(input);
  }

private:
  TRANSFORM transform;
};

/**
 * @brief Host type for data applying a unary transform on an input array
 *
 * As with the on-device type, the argument pack is a template parameter
 * (defaulting to SingleReactionDataArgumentPack) so that the stage can be
 * placed inside either a single or a pair pipeline. The matching device
 * accessor pack is derived via accessor_pack_for_t.
 *
 * @tparam TRANSFORM The transformation type being applied
 * @tparam ARGUMENT_PACK_T Bundled host-side property requirements
 * (Optional, defaults to SingleReactionDataArgumentPack).
 */
template <typename TRANSFORM,
          typename ARGUMENT_PACK_T = SingleReactionDataArgumentPack>
struct UnaryArrayTransformData
    : public AbstractReactionData<
          UnaryArrayTransformDataOnDevice<TRANSFORM,
                                          accessor_pack_for_t<ARGUMENT_PACK_T>>,
          ARGUMENT_PACK_T, TRANSFORM::OUT_DIM, DEFAULT_RNG_KERNEL,
          TRANSFORM::IN_DIM> {

  /**
   * @brief Constructor for UnaryArrayTransformData
   *
   * @param transform Unary transform object (derived from
   * AbstractUnaryTransform) to be applied on input data
   */
  UnaryArrayTransformData(const TRANSFORM &transform)
      : AbstractReactionData<
            UnaryArrayTransformDataOnDevice<
                TRANSFORM, accessor_pack_for_t<ARGUMENT_PACK_T>>,
            ARGUMENT_PACK_T, TRANSFORM::OUT_DIM, DEFAULT_RNG_KERNEL,
            TRANSFORM::IN_DIM>(ARGUMENT_PACK_T(), get_default_map()) {
    this->on_device_obj = UnaryArrayTransformDataOnDevice<
        TRANSFORM, accessor_pack_for_t<ARGUMENT_PACK_T>>(transform);
  };

  /**
   * @brief No-op since there are no required properties to index
   */
  void index_on_device_object() {};
};
}; // namespace VANTAGE::Reactions
#endif
