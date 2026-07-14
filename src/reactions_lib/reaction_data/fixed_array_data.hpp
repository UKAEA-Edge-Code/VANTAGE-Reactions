#ifndef REACTIONS_FIXED_ARRAY_DATA_H
#define REACTIONS_FIXED_ARRAY_DATA_H
#include "../reaction_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction data calculation returning a fixed array.
 *
 * @tparam ndim The dimension of the returned array
 */
template <size_t ndim>
struct FixedArrayDataOnDevice : public ReactionDataBaseOnDevice<ndim> {

  FixedArrayDataOnDevice() = default;
  /**
   * @brief Constructor for FixedArrayDataOnDevice.
   *
   * @param data REAL-valued array this object will returns.
   */
  FixedArrayDataOnDevice(const std::array<REAL, ndim> &data) : data(data) {};

  /**
   * @brief Returns fixed array
   *
   * @param accessors Bundled accessors for the ParticleLoop.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return Fixed ndim-sized array
   */
  std::array<REAL, ndim>
  calc_data(const SingleReactionDataAccessors &accessors,
            typename DEFAULT_RNG_KERNEL::KernelType &kernel) const {

    return this->data;
  }

private:
  std::array<REAL, ndim> data;
};

/**
 * @brief Reaction data returning a fixed array.
 *
 * @tparam ndim The size of the returned array
 */
template <size_t ndim>
struct FixedArrayData
    : public ReactionDataBase<FixedArrayDataOnDevice<ndim>, ndim> {

  /**
   * @brief Constructor for FixedArrayData.
   *
   * @param data REAL-valued array to always return.
   */
  FixedArrayData(const std::array<REAL, ndim> &data) {
    this->on_device_obj = FixedArrayDataOnDevice(data);
  };

  /**
   * @brief No-op since there are no required properties to index
   */
  void index_on_device_object() {};
};
}; // namespace VANTAGE::Reactions
#endif
