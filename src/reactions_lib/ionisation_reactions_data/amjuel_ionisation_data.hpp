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

#define FIELD_REAL_PROPS X(fluid_density), X(fluid_temperature)

template <int num_coeffs>
struct IoniseReactionAMJUELData : public ReactionDataBase {
  IoniseReactionAMJUELData() = default;

  IoniseReactionAMJUELData(REAL density_normalisation,
                           std::array<REAL, num_coeffs> coeffs)
      : density_normalisation(density_normalisation), coeffs(coeffs) {}

  REAL calc_rate(Access::LoopIndex::Read &index,
                 Access::SymVector::Read<INT> &req_part_ints,
                 Access::SymVector::Read<REAL> &req_part_reals,
                 Access::SymVector::Read<INT> &req_field_ints,
                 Access::SymVector::Read<REAL> &req_field_reals) const {
    auto fluid_density_dat = req_field_reals.at(fluid_density, index, 0);
    auto fluid_temperature_dat = req_field_reals.at(fluid_temperature, index, 0);

    REAL log_cross_section_vel = 0.0;
    for (int i = 0; i < num_coeffs; i++) {
      log_cross_section_vel +=
          this->coeffs[i] * std::pow(std::log(fluid_temperature_dat), i);
    }

    REAL cross_section_vel = std::exp(log_cross_section_vel);

    REAL rate = cross_section_vel * fluid_density_dat * this->density_normalisation;

    return rate;
  }
#define X(M) M
  enum { FIELD_REAL_PROPS, NUM_FIELD_REAL_PROPS };
#undef X
  const int get_num_field_real_props() { return NUM_FIELD_REAL_PROPS; }

#define X(M) #M
  const char *required_field_real_prop_names[NUM_FIELD_REAL_PROPS] = { FIELD_REAL_PROPS };
#undef X
  const char **get_required_field_real_props() {
    return required_field_real_prop_names;
  }

private:
  REAL density_normalisation;
  std::array<REAL, num_coeffs> coeffs;
};
#undef FIELD_REAL_PROPS