#pragma once
#include "../particle_properties_map.hpp"
#include <array>
#include <cmath>
#include <neso_particles.hpp>
#include "../reaction_data.hpp"

using namespace NESO::Particles;
namespace Reactions {

// AMJUEL 1D Fit

namespace AMJUEL_1D_DATA {

const auto props = default_properties;

const std::vector<int> required_simple_real_props = {
    props.fluid_density, props.fluid_temperature, props.weight};
} // namespace AMJUEL_1D_DATA

/**
 * @brief A struct that contains data and calc_data functions that are to be
 * stored on and used on a SYCL device.
 *
 * @tparam num_coeffs The number of coefficients needed for 1D AMJUEL reaction
 * rate calculation.
 * @param evolved_quantity_normalisation Normalisation constant for the evolved
 * quantity (for default rates should be 1)
 * @param density_normalisation Density normalisation constant in m^{-3}
 * @param temperature_normalisation Temperature normalisation in eV
 * @param time_normalisation Time normalisation in seconds
 * @param coeffs A real-valued array of coefficients to be used in a 1D AMJUEL
 * reaction rate calculation.
 */
template <int num_coeffs>
struct AMJUEL1DDataOnDevice : public ReactionDataBaseOnDevice<> {
  AMJUEL1DDataOnDevice(const REAL &evolved_quantity_normalisation_,
                       const REAL &density_normalisation_,
                       const REAL &temperature_normalisation_,
                       const REAL &time_normalisation_,
                       const std::array<REAL, num_coeffs> &coeffs_)
      : evolved_quantity_normalisation(1.0 / evolved_quantity_normalisation_),
        density_normalisation(density_normalisation_),
        temperature_normalisation(temperature_normalisation_),
        time_normalisation(time_normalisation_), coeffs(coeffs_){};

  /**
   * @brief Function to calculate the reaction rate for a 1D AMJUEL-based
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

    REAL log_rate = 0.0;
    for (int i = 0; i < num_coeffs; i++) {
      log_rate +=
          this->coeffs[i] * std::pow(std::log(fluid_temperature_dat *
                                              this->temperature_normalisation),
                                     i);
    }

    REAL rate = std::exp(log_rate) * 1.0e-6;

    rate *= req_real_props.at(this->weight_ind, index, 0) * fluid_density_dat *
            this->time_normalisation * this->density_normalisation *
            this->evolved_quantity_normalisation;

    return std::array<REAL, 1>{rate};
  }

public:
  int fluid_density_ind, fluid_temperature_ind, weight_ind;
  REAL evolved_quantity_normalisation;
  REAL density_normalisation;
  REAL temperature_normalisation;
  REAL time_normalisation;
  std::array<REAL, num_coeffs> coeffs;
};

/**
 * @brief A struct defining the data needed for a 1D AMJUEL rate calculation
 *
 * @tparam num_coeffs The number of coefficients needed for 1D AMJUEL reaction
 * rate calculation.
 * @param evolved_quantity_normalisation Normalisation of the evolved quantity (the one evolved with this rate)
 * @param density_normalisation Density normalisation constant in m^{-3}
 * @param temperature_normalisation Temperature normalisation in eV
 * @param time_normalisation Time normalisation in seconds
 * @param coeffs A real-valued array of coefficients to be used in a 1D AMJUEL
 * reaction rate calculation.
 * @param properties_map_ A std::map<int, std::string> object to be passed to
 * ReactionDataBase.
 */
template <int num_coeffs> struct AMJUEL1DData : public ReactionDataBase<> {

  AMJUEL1DData(const REAL &evolved_quantity_normalisation_,
               const REAL &density_normalisation_,
               const REAL &temperature_normalisation_,
               const REAL &time_normalisation_,
               const std::array<REAL, num_coeffs> &coeffs_,
               std::map<int, std::string> properties_map_ = default_map)
      : ReactionDataBase(
            Properties<REAL>(AMJUEL_1D_DATA::required_simple_real_props,
                             std::vector<Species>{}, std::vector<int>{}),
            properties_map_),
        amjuel_1d_data_on_device(AMJUEL1DDataOnDevice<num_coeffs>(
            evolved_quantity_normalisation_, density_normalisation_,
            temperature_normalisation_, time_normalisation_, coeffs_)) {

    auto props = AMJUEL_1D_DATA::props;

    this->amjuel_1d_data_on_device.fluid_density_ind =
        this->required_real_props.simple_prop_index(props.fluid_density,
                                                    this->properties_map);
    this->amjuel_1d_data_on_device.fluid_temperature_ind =
        this->required_real_props.simple_prop_index(props.fluid_temperature,
                                                    this->properties_map);
    this->amjuel_1d_data_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);
  }

private:
  AMJUEL1DDataOnDevice<num_coeffs> amjuel_1d_data_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */

  AMJUEL1DDataOnDevice<num_coeffs> get_on_device_obj() {
    return this->amjuel_1d_data_on_device;
  }
};
}; // namespace Reactions
