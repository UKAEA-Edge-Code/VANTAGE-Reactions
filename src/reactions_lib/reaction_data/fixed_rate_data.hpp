#ifndef REACTIONS_FIXED_RATE_DATA_H
#define REACTIONS_FIXED_RATE_DATA_H
#include <neso_particles.hpp>
#include "../reaction_data.hpp"

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief SYCL device-compatible ReactionData class returning a fixed rate
 */
struct FixedRateDataOnDevice : public ReactionDataBaseOnDevice<> {

  /**
   * @brief Constructor for FixedRateDataOnDevice.
   *
   * @param rate REAL-valued rate to be used in reaction rate calculation.
   */
  FixedRateDataOnDevice(const REAL &rate) : rate(rate){};

  /**
   * @brief Function to calculate the reaction rate for a fixed rate reaction
   *
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
   * @return A REAL-valued array of size 1 containing the calculated reaction rate.
   */
  std::array<REAL, 1>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<REAL, 1>{this->rate};
  }

private:
  REAL rate;
};

/**
 * @brief A struct defining the data needed for a fixed rate reaction.
 */
struct FixedRateData : public ReactionDataBase<> {

  /**
   * @brief Constructor for FixedRateData.
   *
   * @param rate REAL-valued rate to be used in reaction rate calculation.
   */
  FixedRateData(const REAL &rate)
      : fixed_rate_data_on_device(FixedRateDataOnDevice(rate)) {}

private:
  FixedRateDataOnDevice fixed_rate_data_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */
  FixedRateDataOnDevice get_on_device_obj() {
    return this->fixed_rate_data_on_device;
  }
};
}; // namespace VANTAGE::Reactions
#endif