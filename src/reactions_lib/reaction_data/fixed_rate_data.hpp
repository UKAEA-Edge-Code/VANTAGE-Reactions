#pragma once
#include <neso_particles.hpp>
#include <neso_particles/containers/sym_vector.hpp>
#include "../reaction_data.hpp"

using namespace NESO::Particles;
namespace Reactions {

/**
 * @brief SYCL device-compatible ReactionData class returning a fixed rate
 *
 * @param rate_ REAL-valued rate to be used in reaction rate calculation.
 */
struct FixedRateDataOnDevice : public ReactionDataBaseOnDevice<> {
  FixedRateDataOnDevice(const REAL &rate_) : rate(rate_){};

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
   */
  std::array<REAL, 1>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Read<INT> &req_int_props,
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
 *
 * @param rate_ REAL-valued rate to be used in reaction rate calculation.
 */
struct FixedRateData : public ReactionDataBase<> {

  FixedRateData(const REAL &rate_)
      : fixed_rate_data_on_device(FixedRateDataOnDevice(rate_)) {}

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
}; // namespace Reactions
