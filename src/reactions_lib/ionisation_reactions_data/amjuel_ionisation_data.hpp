#pragma once
#include "containers/sym_vector.hpp"
#include "typedefs.hpp"
#include <array>
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

template <int num_coeffs>
struct IoniseReactionAMJUELData
    : public ReactionDataBase
    {
  IoniseReactionAMJUELData() = default;

  IoniseReactionAMJUELData(REAL density_normalisation,
                           std::array<REAL, num_coeffs> coeffs)
      : density_normalisation(density_normalisation), coeffs(coeffs) {}

  REAL calc_rate(Access::LoopIndex::Read &index,
                 Access::SymVector::Read<REAL> &vars) const {
    auto fluid_density = vars.at(8, index, 0);
    auto fluid_temperature = vars.at(7, index, 0);

    REAL log_cross_section_vel = 0.0;
    for (int i = 0; i < num_coeffs; i++) {
      log_cross_section_vel +=
          this->coeffs[i] * std::pow(std::log(fluid_temperature), i);
    }

    REAL cross_section_vel = std::exp(log_cross_section_vel);

    REAL rate = cross_section_vel * fluid_density * this->density_normalisation;

    return rate;
  }

private:
  REAL density_normalisation;
  std::array<REAL, num_coeffs> coeffs;
};