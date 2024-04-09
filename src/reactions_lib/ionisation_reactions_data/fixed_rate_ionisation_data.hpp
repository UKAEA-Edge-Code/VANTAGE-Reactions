#pragma once
#include "containers/sym_vector.hpp"
#include "typedefs.hpp"
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

struct FixedRateIonisationData
    : public ReactionDataBase<FixedRateIonisationData> {
  FixedRateIonisationData() = default;

  FixedRateIonisationData(REAL rate) : rate(rate) {}

  REAL calc_rate(Access::LoopIndex::Read &index,
                 Access::SymVector::Read<REAL> &vars) const {

    return this->rate;
  }

private:
  REAL rate;
};