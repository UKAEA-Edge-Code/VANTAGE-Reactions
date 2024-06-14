#pragma once
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/containers/sym_vector.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;
using namespace ParticlePropertiesIndices;

//AMJUEL 1D Fit

namespace AMJUEL_IONISATION_DATA {

const auto props = ParticlePropertiesIndices::default_properties;

const std::vector<int> required_simple_real_props = {props.fluid_density,
                                                     props.fluid_temperature};
} // namespace AMJUEL_IONISATION_DATA

/**
 * @brief A struct that contains data and calc_rate functions that are to be
 * stored on and used on a SYCL device.
 *
 * @tparam num_coeffs The number of coefficients needed for 1D AMJUEL reaction
 * rate calculation.
 * @param density_normalisation A normalisation constant to be used in a 1D
 * AMJUEL reaction rate calculation.
 * @param coeffs A real-valued array of coefficients to be used in a 1D AMJUEL
 * reaction rate calculation.
 */
template <int num_coeffs>
struct IoniseReactionAMJUELDataOnDevice : public ReactionDataBaseOnDevice {
  IoniseReactionAMJUELDataOnDevice(const REAL& density_normalisation_,
                                   const std::array<REAL, num_coeffs>& coeffs_)
      : density_normalisation(density_normalisation_), coeffs(coeffs_){};

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
  REAL calc_rate(const Access::LoopIndex::Read &index,
                //  const Access::SymVector::Read<INT> &req_simple_prop_ints,
                //  const Access::SymVector::Read<REAL> &req_simple_prop_reals,
                //  const Access::SymVector::Read<INT> &req_species_prop_ints,
                //  const Access::SymVector::Read<REAL> &req_species_prop_reals
                const Access::SymVector::Read<INT> &req_int_props,
                const Access::SymVector::Read<REAL> &req_real_props
                 ) const {
    auto fluid_density_dat =
        req_real_props.at(this->fluid_density_ind, index, 0);
    auto fluid_temperature_dat =
        req_real_props.at(this->fluid_temperature_ind, index, 0);

    REAL log_rate = 0.0; // rate not cross_section_vel
    for (int i = 0; i < num_coeffs; i++) {
      log_rate +=
          this->coeffs[i] * std::pow(std::log(fluid_temperature_dat), i);
    }

    REAL rate = std::exp(log_rate);

    rate *= fluid_density_dat * this->density_normalisation;

    return rate;
  }

public:
  int fluid_density_ind, fluid_temperature_ind;
  REAL density_normalisation;
  std::array<REAL, num_coeffs> coeffs;
};

/**
 * @brief A struct defining the data needed for a 1D AMJUEL-base ionisation
 * reaction.
 *
 * @tparam num_coeffs The number of coefficients needed for 1D AMJUEL reaction
 * rate calculation.
 * @param density_normalisation A normalisation constant to be used in a 1D
 * AMJUEL reaction rate calculation.
 * @param coeffs A real-valued array of coefficients to be used in a 1D AMJUEL
 * reaction rate calculation.
 */
template <int num_coeffs>
struct IoniseReactionAMJUELData : public ReactionDataBase {
  IoniseReactionAMJUELData() = default;

  IoniseReactionAMJUELData(const REAL& density_normalisation_,
                           const std::array<REAL, num_coeffs>& coeffs_)
      : ionise_reaction_amjuel_data_on_device(
            IoniseReactionAMJUELDataOnDevice<num_coeffs>(density_normalisation_,
                                                         coeffs_)),
        required_real_props(Properties<REAL>(
            AMJUEL_IONISATION_DATA::required_simple_real_props,
            std::vector<Species>{}, std::vector<int>{})) {

    auto props = AMJUEL_IONISATION_DATA::props;

    this->ionise_reaction_amjuel_data_on_device.fluid_density_ind =
        this->required_real_props.required_simple_prop_index(props.fluid_density);
    this->ionise_reaction_amjuel_data_on_device.fluid_temperature_ind =
        this->required_real_props.required_simple_prop_index(props.fluid_temperature);
  }

private:
  IoniseReactionAMJUELDataOnDevice<num_coeffs>
      ionise_reaction_amjuel_data_on_device;

  Properties<REAL> required_real_props;

public:
  /**
   * @brief Getters for number of simple_real_props, required_simple_prop_names
   * and the SYCL device-specific struct.
   */
  std::vector<std::string> get_required_real_props() {
    return this->required_real_props.required_simple_prop_names();
  }

  IoniseReactionAMJUELDataOnDevice<num_coeffs> get_on_device_obj() {
    return this->ionise_reaction_amjuel_data_on_device;
  }
};