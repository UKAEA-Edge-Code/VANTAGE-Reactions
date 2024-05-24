#pragma once
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;

/**
 * @brief A struct that contains data and calc_rate functions that are to be
 * stored on and used on a SYCL device.
 *
 * @param rate_ REAL-valued rate to be used in reaction rate calculation.
 */
struct FixedRateIonisationDataOnDevice : public ReactionDataBaseOnDevice {
  FixedRateIonisationDataOnDevice(REAL rate_) : rate(rate_){};

  /**
   * @brief Function to calculate the reaction rate for a 1D AMJUEL-based
   * ionisation reaction.
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_simple_prop_ints Vector of symbols for simple integer-valued
   * properties that need to be used for the reaction rate calculation.
   * @param req_simple_prop_reals Vector of symbols for simple real-valued
   * properties that need to be used for the reaction rate calculation.
   * @param req_species_prop_ints Vector of symbols for species-dependent
   * integer-valued properties that need to be used for the reaction rate
   * calculation.
   * @param req_species_prop_reals Vector of symbols for species-dependent
   * real-valued properties that need to be used for the reaction rate
   * calculation.
   */
  REAL calc_rate(Access::LoopIndex::Read &index,
                 Access::SymVector::Read<INT> &req_simple_prop_ints,
                 Access::SymVector::Read<REAL> &req_simple_prop_reals,
                 Access::SymVector::Read<INT> &req_species_prop_ints,
                 Access::SymVector::Read<REAL> &req_species_prop_reals) const {

    return this->rate;
  }

private:
  REAL rate;
};

/**
 * @brief A struct defining the data needed for a fixed rate ionisation
 * reaction.
 *
 * @param rate_ REAL-valued rate to be used in reaction rate calculation.
 */
struct FixedRateIonisationData : public ReactionDataBase {
  FixedRateIonisationData() = default;

  FixedRateIonisationData(REAL rate_)
      : rate(rate_), fixed_rate_ionisation_data_on_device(
                         FixedRateIonisationDataOnDevice(rate_)) {}

private:
  FixedRateIonisationDataOnDevice fixed_rate_ionisation_data_on_device;

  REAL rate;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */
  FixedRateIonisationDataOnDevice get_on_device_obj() {
    return this->fixed_rate_ionisation_data_on_device;
  }
};