#ifndef REACTIONS_FIXED_RATE_DATA_H
#define REACTIONS_FIXED_RATE_DATA_H
#include "../reaction_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction rate data calculation for a fixed rate reaction.
 */
struct FixedRateDataOnDevice : public ReactionDataBaseOnDevice<> {

  FixedRateDataOnDevice() = default;
  /**
   * @brief Constructor for FixedRateDataOnDevice.
   *
   * @param rate REAL-valued rate to be used in reaction rate calculation.
   */
  FixedRateDataOnDevice(const REAL &rate) : rate(rate) {};

  /**
   * @brief Function to calculate the reaction rate for a fixed rate reaction
   *
   * @param accessors Bundled accessors for the ParticleLoop.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A REAL-valued array of size 1 containing the calculated reaction
   * rate.
   */
  std::array<REAL, 1>
  calc_data(const SingleReactionDataAccessors &accessors,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<REAL, 1>{this->rate};
  }

private:
  REAL rate;
};

/**
 * @brief Reaction rate data calculation for a fixed rate reaction.
 */
struct FixedRateData : public ReactionDataBase<FixedRateDataOnDevice> {

  /**
   * @brief Constructor for FixedRateData.
   *
   * @param rate REAL-valued rate to be used in reaction rate calculation.
   */
  FixedRateData(const REAL &rate) {
    this->on_device_obj = FixedRateDataOnDevice(rate);
  };

  /**
   * @brief No-op since there are no required properties to index
   */
  void index_on_device_object() {};
};
}; // namespace VANTAGE::Reactions
#endif
