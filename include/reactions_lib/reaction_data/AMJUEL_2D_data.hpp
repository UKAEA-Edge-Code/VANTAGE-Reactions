#ifndef REACTIONS_AMJUEL_2D_DATA_H
#define REACTIONS_AMJUEL_2D_DATA_H
#include "../particle_properties_map.hpp"
#include "../reaction_data.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <array>
#include <cmath>

namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction rate data calculation based on 2D AMJUEL rate
 * calculation against ion/plasma density  and ion/plasma temperature. Handles
 * Coronal approximation correctly.
 *
 * @tparam num_coeffs_T The number of fit parameters in the T direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @tparam num_coeffs_n The number of fit parameters in the n direction needed
 * for 2D AMJUEL reaction rate calculation.
 */
template <int num_coeffs_T, int num_coeffs_n>
struct AMJUEL2DDataOnDevice : public ReactionDataBaseOnDevice<> {

  AMJUEL2DDataOnDevice() = default;
  /**
   * @brief Constructor for AMJUEL2DDataOnDevice.
   *
   * @param evolved_quantity_normalisation Normalisation constant for the
   * evolved quantity (for default rates should be 1)
   * @param density_normalisation Density normalisation constant in m^{-3}
   * @param temperature_normalisation Temperature normalisation in eV
   * @param time_normalisation Time normalisation in seconds
   * @param coeffs A real-valued 2D array of coefficients to be used in a 2D
   * AMJUEL reaction rate calculation.
   */
  AMJUEL2DDataOnDevice(const NP::REAL &evolved_quantity_normalisation,
                       const NP::REAL &density_normalisation,
                       const NP::REAL &temperature_normalisation,
                       const NP::REAL &time_normalisation,
                       const std::array<std::array<NP::REAL, num_coeffs_n>,
                                        num_coeffs_T> &coeffs)
      : mult_const(density_normalisation * time_normalisation /
                   evolved_quantity_normalisation),
        density_normalisation(density_normalisation),
        temperature_normalisation(temperature_normalisation), coeffs(coeffs) {};

  /**
   * @brief Function to calculate the reaction rate for a 2D AMJUEL-based
   * reaction.
   *
   * @param index Read-only accessor to a loop index for a NP::ParticleLoop
   * inside which calc_data is called. NP::Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A NP::REAL-valued array containing the calculated reaction rate.
   */
  std::array<NP::REAL, 1>
  calc_data(const NP::Access::LoopIndex::Read &index,
            const NP::Access::SymVector::Write<NP::INT> &req_int_props,
            const NP::Access::SymVector::Read<NP::REAL> &req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {
    auto fluid_density_dat =
        req_real_props.at(this->fluid_density_ind, index, 0);
    auto fluid_temperature_dat =
        req_real_props.at(this->fluid_temperature_ind, index, 0);
    NP::REAL log_temp = NP::Kernel::log(fluid_temperature_dat *
                                        this->temperature_normalisation);

    std::array<NP::REAL, num_coeffs_T> log_temp_arr;
    log_temp_arr[0] = 1.0;
    for (int i = 1; i < num_coeffs_T; i++) {
      log_temp_arr[i] = log_temp_arr[i - 1] * log_temp;
    }

    NP::REAL log_rate = 0.0;
    // Ensuring the Coronal asymptote gets treated correctly
    // TODO: Add variable Coronal cut-off density
    auto log_n = (fluid_density_dat * this->density_normalisation >= 1e14)
                     ? NP::Kernel::log(fluid_density_dat *
                                       this->density_normalisation / 1e14)
                     : 0;
    // TODO: Ensure LTE asymptotic behaviour obeyed

    std::array<NP::REAL, num_coeffs_n> log_n_arr;
    log_n_arr[0] = 1.0;
    for (int i = 1; i < num_coeffs_n; i++) {
      log_n_arr[i] = log_n_arr[i - 1] * log_n;
    }

    for (int j = 0; j < num_coeffs_n; j++) {
      for (int i = 0; i < num_coeffs_T; i++) {
        log_rate += this->coeffs[i][j] * log_n_arr[j] * log_temp_arr[i];
      }
    }

    NP::REAL rate = NP::Kernel::exp(log_rate) * 1.0e-6;

    rate *= req_real_props.at(this->weight_ind, index, 0) * fluid_density_dat *
            this->mult_const;

    return std::array<NP::REAL, 1>{rate};
  }

public:
  int fluid_density_ind, fluid_temperature_ind, weight_ind;
  NP::REAL density_normalisation;
  NP::REAL temperature_normalisation;
  NP::REAL mult_const;
  std::array<std::array<NP::REAL, num_coeffs_n>, num_coeffs_T> coeffs;
};

/**
 * @brief Reaction rate data calculation based on 2D AMJUEL rate calculation
 * against ion/plasma density  and ion/plasma temperature. Handles Coronal
 * approximation correctly.
 *
 * @tparam num_coeffs_T The number of fit parameters in the T direction needed
 * for 2D AMJUEL reaction rate calculation.
 * @tparam num_coeffs_n The number of fit parameters in the n direction needed
 * for 2D AMJUEL reaction rate calculation.
 */
template <int num_coeffs_T, int num_coeffs_n>
struct AMJUEL2DData : public ReactionDataBase<
                          AMJUEL2DDataOnDevice<num_coeffs_T, num_coeffs_n>> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 3> required_simple_real_props = {
      props.fluid_density, props.fluid_temperature, props.weight};

  /**
   * @brief Constructor for AMJUEL2DData.
   *
   * @param evolved_quantity_normalisation Normalisation constant for the
   * evolved quantity (for default rates should be 1)
   * @param density_normalisation Density normalisation constant in m^{-3}
   * @param temperature_normalisation Temperature normalisation in eV
   * @param time_normalisation Time normalisation in seconds
   * @param coeffs A real-valued 2D array of coefficients to be used in a 2D
   * AMJUEL reaction rate calculation.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  AMJUEL2DData(const NP::REAL &evolved_quantity_normalisation,
               const NP::REAL &density_normalisation,
               const NP::REAL &temperature_normalisation,
               const NP::REAL &time_normalisation,
               const std::array<std::array<NP::REAL, num_coeffs_n>,
                                num_coeffs_T> &coeffs,
               std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<AMJUEL2DDataOnDevice<num_coeffs_T, num_coeffs_n>>(
            Properties<NP::REAL>(required_simple_real_props), properties_map) {
    this->on_device_obj = AMJUEL2DDataOnDevice<num_coeffs_T, num_coeffs_n>(
        evolved_quantity_normalisation, density_normalisation,
        temperature_normalisation, time_normalisation, coeffs);

    this->index_on_device_object();
  }

  /**
   * @brief Index the fluid density, temperature, and particle weight on the
   * on-device object
   */
  void index_on_device_object() {

    this->on_device_obj->fluid_density_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.fluid_density));

    this->on_device_obj->fluid_temperature_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.fluid_temperature));

    this->on_device_obj->weight_ind = this->required_real_props.find_index(
        this->properties_map.at(props.weight));
  };
};
}; // namespace VANTAGE::Reactions
#endif
