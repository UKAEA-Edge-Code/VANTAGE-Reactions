#ifndef AMJUEL_2D_DATA_H
#define AMJUEL_2D_DATA_H
#include "../particle_properties_map.hpp"
#include "../reaction_data.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include <array>
#include <cmath>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief A struct that contains data and calc_data functions that are to be
 * stored on and used on a SYCL device.
 *
 * @tparam num_coeffs_T The number of fit parameters in the T direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @tparam num_coeffs_n The number of fit parameters in the n direction needed
 * for 2D AMJUEL reaction rate calculation.
 */
template <int num_coeffs_T, int num_coeffs_n>
struct AMJUEL2DDataOnDevice : public ReactionDataBaseOnDevice<> {

  /**
   * @brief Constructor for AMJUEL2DDataOnDevice.
   *
   * @param evolved_quantity_normalisation Normalisation constant for the evolved
   * quantity (for default rates should be 1)
   * @param density_normalisation Density normalisation constant in m^{-3}
   * @param temperature_normalisation Temperature normalisation in eV
   * @param time_normalisation Time normalisation in seconds
   * @param coeffs A real-valued 2D array of coefficients to be used in a 2D
   * AMJUEL reaction rate calculation.
   */
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
   *
   * @return A REAL-valued array containing the calculated reaction rate.
   */
  std::array<REAL, 1>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {
    auto fluid_density_dat =
        req_real_props.at(this->fluid_density_ind, index, 0);
    auto fluid_temperature_dat =
        req_real_props.at(this->fluid_temperature_ind, index, 0);
    REAL log_temp =
        Kernel::log(fluid_temperature_dat * this->temperature_normalisation);

    std::array<REAL, num_coeffs_T> log_temp_arr;
    log_temp_arr[0] = 1.0;
    for (int i = 1; i < num_coeffs_T; i++) {
      log_temp_arr[i] = log_temp_arr[i - 1] * log_temp;
    }

    REAL log_rate = 0.0;
    // Ensuring the Coronal asymptote gets treated correctly
    // TODO: Add variable Coronal cut-off density
    auto log_n =
        (fluid_density_dat * this->density_normalisation >= 1e14)
            ? Kernel::log(fluid_density_dat * this->density_normalisation / 1e14)
            : 0;
    // TODO: Ensure LTE asymptotic behaviour obeyed

    std::array<REAL, num_coeffs_n> log_n_arr;
    log_n_arr[0] = 1.0;
    for (int i = 1; i < num_coeffs_n; i++) {
      log_n_arr[i] = log_n_arr[i - 1] * log_n;
    }

    for (int j = 0; j < num_coeffs_n; j++) {
      for (int i = 0; i < num_coeffs_T; i++) {
        log_rate += this->coeffs[i][j] * log_n_arr[j] * log_temp_arr[i];
      }
    }

    REAL rate = Kernel::exp(log_rate) * 1.0e-6;

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
 */
template <int num_coeffs_T, int num_coeffs_n>
struct AMJUEL2DData : public ReactionDataBase<> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 3> required_simple_real_props = {
      props.fluid_density, props.fluid_temperature, props.weight};

  /**
   * @brief Constructor for AMJUEL2DData.
   *
   * @param density_normalisation Density normalisation constant in m^{-3}
   * @param temperature_normalisation Temperature normalisation in eV
   * @param time_normalisation Time normalisation in seconds
   * @param coeffs A real-valued 2D array of coefficients to be used in a 2D
   * AMJUEL reaction rate calculation.
   * @param properties_map A std::map<int, std::string> object to be passed to
   * ReactionDataBase
   */
  AMJUEL2DData(
      const REAL &evolved_quantity_normalisation_,
      const REAL &density_normalisation_,
      const REAL &temperature_normalisation_, const REAL &time_normalisation_,
      const std::array<std::array<REAL, num_coeffs_n>, num_coeffs_T> &coeffs_,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(Properties<REAL>(required_simple_real_props),
                         properties_map),
        amjuel_2d_data_on_device(
            AMJUEL2DDataOnDevice<num_coeffs_T, num_coeffs_n>(
                evolved_quantity_normalisation_, density_normalisation_,
                temperature_normalisation_, time_normalisation_, coeffs_)) {

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
}; // namespace VANTAGE::Reactions
#endif