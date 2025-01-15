#pragma once
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <array>
#include <cmath>
#include <neso_particles.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

using namespace NESO::Particles;
namespace Reactions {

// AMJUEL 2D Fit

namespace AMJUEL_2D_DATA {

const auto props = default_properties;

const std::vector<int> required_simple_real_props = {
    props.fluid_density, props.fluid_temperature, props.weight};
} // namespace AMJUEL_2D_DATA

/**
 * @brief A struct that contains data and calc_data functions that are to be
 * stored on and used on a SYCL device.
 *
 * @tparam num_coeffs_T The number of fit parameters in the T direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @tparam num_coeffs_n The number of fit parameters in the n direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @param evolved_quantity_normalisation Normalisation constant for the evolved
 * quantity (for default rates should be 1)
 * @param density_normalisation Density normalisation constant in m^{-3}
 * @param temperature_normalisation Temperature normalisation in eV
 * @param time_normalisation Time normalisation in seconds
 * @param coeffs A real-valued 2D array of coefficients to be used in a 2D
 * AMJUEL reaction rate calculation.
 */
template <int num_coeffs_T, int num_coeffs_n>
struct AMJUEL2DDataOnDevice : public ReactionDataBaseOnDevice<> {
  AMJUEL2DDataOnDevice(
      const REAL &evolved_quantity_normalisation_,
      const REAL &density_normalisation_,
      const REAL &temperature_normalisation_, const REAL &time_normalisation_,
      const std::array<std::array<REAL, num_coeffs_n>, num_coeffs_T> &coeffs_)
      : mult_const(density_normalisation_ * time_normalisation_ /
                   evolved_quantity_normalisation_),
        density_normalisation(density_normalisation_),
        temperature_normalisation(temperature_normalisation_),
        coeffs(coeffs_){};

  /**
   * @brief Function to calculate the reaction rate for a 2D AMJUEL-based
   * reaction.
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
    auto fluid_density_dat =
        req_real_props.at(this->fluid_density_ind, index, 0);
    auto fluid_temperature_dat =
        req_real_props.at(this->fluid_temperature_ind, index, 0);
    REAL log_temp =
        std::log(fluid_temperature_dat * this->temperature_normalisation);
    REAL log_rate = 0.0;
    // Ensuring the Coronal asymptote gets treated correctly
    // TODO: Add variable Coronal cut-off density
    auto log_n =
        (fluid_density_dat * this->density_normalisation >= 1e14)
            ? std::log(fluid_density_dat * this->density_normalisation / 1e14)
            : 0;
    // TODO: Ensure LTE asymptotic behaviour obeyed
    for (int j = 0; j < num_coeffs_n; j++) {
      auto log_n_m = (j == 0) ? 1.0 : std::pow(log_n, j);
      for (int i = 0; i < num_coeffs_T; i++) {
        log_rate += this->coeffs[i][j] * log_n_m * std::pow(log_temp, i);
      }
    }

    REAL rate = std::exp(log_rate) * 1.0e-6;

    rate *= req_real_props.at(this->weight_ind, index, 0) * fluid_density_dat *
            this->mult_const;

    return std::array<REAL, 1>{rate};
  }

public:
  int fluid_density_ind, fluid_temperature_ind, weight_ind;
  REAL density_normalisation;
  REAL temperature_normalisation;
  REAL mult_const;
  std::array<std::array<REAL, num_coeffs_n>, num_coeffs_T> coeffs;
};

/**
 * @brief A struct defining the data needed for a 2D AMJUEL rate calculation,
 * assuming density is the second parameter. Handles Coronal approximation
 * correctly.
 *
 * @tparam num_coeffs_T The number of fit parameters in the T direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @tparam num_coeffs_n The number of fit parameters in the n direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @param density_normalisation Density normalisation constant in m^{-3}
 * @param temperature_normalisation Temperature normalisation in eV
 * @param time_normalisation Time normalisation in seconds
 * @param coeffs A real-valued 2D array of coefficients to be used in a 2D
 * AMJUEL reaction rate calculation.
 * @param properties_map_ A std::map<int, std::string> object to be passed to
 * ReactionDataBase
 */
template <int num_coeffs_T, int num_coeffs_n>
struct AMJUEL2DData : public ReactionDataBase<> {

  AMJUEL2DData(
      const REAL &evolved_quantity_normalisation_,
      const REAL &density_normalisation_,
      const REAL &temperature_normalisation_, const REAL &time_normalisation_,
      const std::array<std::array<REAL, num_coeffs_n>, num_coeffs_T> &coeffs_,
      std::map<int, std::string> properties_map_ = default_map)
      : ReactionDataBase(
            Properties<REAL>(AMJUEL_2D_DATA::required_simple_real_props,
                             std::vector<Species>{}, std::vector<int>{}),
            properties_map_),
        amjuel_2d_data_on_device(
            AMJUEL2DDataOnDevice<num_coeffs_T, num_coeffs_n>(
                evolved_quantity_normalisation_, density_normalisation_,
                temperature_normalisation_, time_normalisation_, coeffs_)) {

    auto props = AMJUEL_2D_DATA::props;

    this->amjuel_2d_data_on_device.fluid_density_ind =
        this->required_real_props.simple_prop_index(props.fluid_density,
                                                    this->properties_map);
    this->amjuel_2d_data_on_device.fluid_temperature_ind =
        this->required_real_props.simple_prop_index(props.fluid_temperature,
                                                    this->properties_map);
    this->amjuel_2d_data_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);
  }

private:
  AMJUEL2DDataOnDevice<num_coeffs_T, num_coeffs_n> amjuel_2d_data_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */

  AMJUEL2DDataOnDevice<num_coeffs_T, num_coeffs_n> get_on_device_obj() {
    return this->amjuel_2d_data_on_device;
  }
};
}; // namespace Reactions
