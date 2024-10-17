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

namespace AMJUEL_2D_DATA_H3 {

const auto props = default_properties;

const std::vector<int> required_simple_real_props = {
    props.fluid_density, props.fluid_temperature, props.fluid_flow_speed,
    props.weight, props.velocity};
} // namespace AMJUEL_2D_DATA_H3

/**
 * @brief On device: Reaction rate data calculation based on AMJUEL H.3 fits
 * against neutral particle energy and ion/plasma temperature
 *
 * @tparam num_coeffs_T The number of fit parameters in the T direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @tparam num_coeffs_E The number of fit parameters in the n direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @param evolved_quantity_normalisation Normalisation constant for the evolved
 * quantity (for default rates should be 1)
 * @param density_normalisation Density normalisation constant in m^{-3}
 * @param temperature_normalisation Temperature normalisation in eV
 * @param time_normalisation Time normalisation in seconds
 * @param velocity_normalisation Velocity normalisation in m/s
 * @param mass_amu Mass of the neutral particle in amus
 * @param coeffs A real-valued 2D array of coefficients to be used in a 2D
 * AMJUEL reaction rate calculation.
 */
template <size_t num_coeffs_T, size_t num_coeffs_E, size_t dim>
struct AMJUEL2DDataH3OnDevice : public ReactionDataBaseOnDevice<> {
  AMJUEL2DDataH3OnDevice(
      const REAL &evolved_quantity_normalisation_,
      const REAL &density_normalisation_,
      const REAL &temperature_normalisation_, const REAL &time_normalisation_,
      const REAL &velocity_normalisation_, const REAL &mass_amu_,
      const std::array<std::array<REAL, num_coeffs_E>, num_coeffs_T> &coeffs_)
      : mult_const(time_normalisation_ * density_normalisation_ /
                   evolved_quantity_normalisation_),
        temperature_normalisation(temperature_normalisation_),
        en_mult_const(std::pow(velocity_normalisation_, 2) * mass_amu_ *
                      1.66053904e-27 / (2 * 1.60217663e-19)),
        coeffs(coeffs_){};

  /**
   * @brief Function to calculate the reaction rate for a 2D H.3 AMJUEL-based
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

    REAL E = 0;
    for (int i = 0; i < dim; i++) {
      E += std::pow(req_real_props.at(this->fluid_flow_speed_ind, index, i) -
                        req_real_props.at(this->velocity_ind, index, i),
                    2);
    }
    REAL log_E = std::log(en_mult_const * E);
    REAL log_rate = 0.0;
    for (int j = 0; j < num_coeffs_E; j++) {
      auto log_E_m = std::pow(log_E, j);
      for (int i = 0; i < num_coeffs_T; i++) {
        log_rate += this->coeffs[i][j] * log_E_m * std::pow(log_temp, i);
      }
    }

    REAL rate = std::exp(log_rate) * 1.0e-6;

    rate *= req_real_props.at(this->weight_ind, index, 0) * fluid_density_dat *
            this->mult_const;

    return std::array<REAL, 1>{rate};
  }

public:
  int fluid_density_ind, fluid_temperature_ind, fluid_flow_speed_ind,
      velocity_ind, weight_ind;
  REAL temperature_normalisation;
  REAL en_mult_const;
  REAL mult_const;
  std::array<std::array<REAL, num_coeffs_E>, num_coeffs_T> coeffs;
};

/**
 * @brief  Reaction rate data calculation based on AMJUEL H.3 fits against
 * neutral particle energy and ion/plasma temperature
 *
 * @tparam num_coeffs_T The number of fit parameters in the T direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @tparam num_coeffs_E The number of fit parameters in the n direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @param evolved_quantity_normalisation Normalisation constant for the evolved
 * quantity (for default rates should be 1)
 * @param density_normalisation Density normalisation constant in m^{-3}
 * @param temperature_normalisation Temperature normalisation in eV
 * @param time_normalisation Time normalisation in seconds
 * @param velocity_normalisation Velocity normalisation in m/s
 * @param mass_amu Mass of the neutral particle in amus
 * @param coeffs A real-valued 2D array of coefficients to be used in a 2D
 * AMJUEL reaction rate calculation.
 */
template <size_t num_coeffs_T, size_t num_coeffs_E, size_t dim = 2>
struct AMJUEL2DDataH3 : public ReactionDataBase<> {

  AMJUEL2DDataH3(
      const REAL &evolved_quantity_normalisation_,
      const REAL &density_normalisation_,
      const REAL &temperature_normalisation_, const REAL &time_normalisation_,
      const REAL &velocity_normalisation_, const REAL &mass_amu_,
      const std::array<std::array<REAL, num_coeffs_E>, num_coeffs_T> &coeffs_,
      std::map<int, std::string> properties_map_ = default_map)
      : ReactionDataBase(
            Properties<REAL>(AMJUEL_2D_DATA_H3::required_simple_real_props,
                             std::vector<Species>{}, std::vector<int>{}),
            properties_map_),
        amjuel_2d_data_on_device(
            AMJUEL2DDataH3OnDevice<num_coeffs_T, num_coeffs_E, dim>(
                evolved_quantity_normalisation_, density_normalisation_,
                temperature_normalisation_, time_normalisation_,
                velocity_normalisation_, mass_amu_, coeffs_)) {

    auto props = AMJUEL_2D_DATA_H3::props;

    this->amjuel_2d_data_on_device.fluid_density_ind =
        this->required_real_props.simple_prop_index(props.fluid_density,
                                                    this->properties_map);
    this->amjuel_2d_data_on_device.fluid_temperature_ind =
        this->required_real_props.simple_prop_index(props.fluid_temperature,
                                                    this->properties_map);
    this->amjuel_2d_data_on_device.fluid_flow_speed_ind =
        this->required_real_props.simple_prop_index(props.fluid_flow_speed,
                                                    this->properties_map);
    this->amjuel_2d_data_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);
    this->amjuel_2d_data_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity,
                                                    this->properties_map);
  }

private:
  AMJUEL2DDataH3OnDevice<num_coeffs_T, num_coeffs_E, dim>
      amjuel_2d_data_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */

  AMJUEL2DDataH3OnDevice<num_coeffs_T, num_coeffs_E, dim> get_on_device_obj() {
    return this->amjuel_2d_data_on_device;
  }
};
}; // namespace Reactions
