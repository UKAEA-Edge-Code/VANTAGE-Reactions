#pragma once
#include <neso_particles.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;

struct FixedRateIonisationDataOnDevice : public ReactionDataBaseOnDevice {
  FixedRateIonisationDataOnDevice(REAL rate_) : rate(rate_) {};

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

struct FixedRateIonisationData : public ReactionDataBase {
  FixedRateIonisationData() = default;

  FixedRateIonisationData(REAL rate_) : rate(rate_), fixed_rate_ionisation_data_on_device(FixedRateIonisationDataOnDevice(rate_)) {}

private:
  FixedRateIonisationDataOnDevice fixed_rate_ionisation_data_on_device;

  REAL rate;
public:
  FixedRateIonisationDataOnDevice get_on_device_obj() {
    return this->fixed_rate_ionisation_data_on_device;
  }
};