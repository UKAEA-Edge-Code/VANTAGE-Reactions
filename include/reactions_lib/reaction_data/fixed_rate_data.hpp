#ifndef REACTIONS_FIXED_RATE_DATA_H
#define REACTIONS_FIXED_RATE_DATA_H
#include "../reaction_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction rate data calculation for a fixed rate reaction.
 */
struct FixedRateDataOnDevice : public ReactionDataBaseOnDevice<> {

  FixedRateDataOnDevice() = default;
  /**
   * @brief Constructor for FixedRateDataOnDevice.
   *
   * @param rate NP::REAL-valued rate to be used in reaction rate calculation.
   */
  FixedRateDataOnDevice(const NP::REAL &rate) : rate(rate) {};

  /**
   * @brief Function to calculate the reaction rate for a fixed rate reaction
   *
   * @param index Read-only accessor to a loop index for a NP::ParticleLoop
   * inside which calc_data is called. NP::Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A NP::REAL-valued array of size 1 containing the calculated
   * reaction rate.
   */
  std::array<NP::REAL, 1>
  calc_data(const NP::Access::LoopIndex::Read &index,
            const NP::Access::SymVector::Write<NP::INT> &req_int_props,
            const NP::Access::SymVector::Read<NP::REAL> &req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<NP::REAL, 1>{this->rate};
  }

private:
  NP::REAL rate;
};

/**
 * @brief Reaction rate data calculation for a fixed rate reaction.
 */
struct FixedRateData : public ReactionDataBase<FixedRateDataOnDevice> {

  /**
   * @brief Constructor for FixedRateData.
   *
   * @param rate NP::REAL-valued rate to be used in reaction rate calculation.
   */
  FixedRateData(const NP::REAL &rate);

  /**
   * @brief No-op since there are no required properties to index
   */
  void index_on_device_object();
};
}; // namespace VANTAGE::Reactions
#endif
