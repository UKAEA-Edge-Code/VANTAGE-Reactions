#ifndef REACTIONS_AMJUEL_1D_DATA_H
#define REACTIONS_AMJUEL_1D_DATA_H
#include "../particle_properties_map.hpp"
#include "../reaction_data.hpp"
#include <array>
#include <cmath>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction rate data calculation based on AMJUEL fits against
 * ion/plasma temperature.
 *
 * @tparam num_coeffs The number of coefficients needed for 1D AMJUEL reaction
 * rate calculation.
 */
template <int num_coeffs>
struct AMJUEL1DDataOnDevice : public ReactionDataBaseOnDevice<> {

  AMJUEL1DDataOnDevice() = default;
  /**
   * @brief Constructor for AMJUEL1DDataOnDevice.
   *
   * @param evolved_quantity_normalisation Normalisation constant for the
   * evolved quantity (for default rates should be 1)
   * @param density_normalisation Density normalisation constant in m^{-3}
   * @param temperature_normalisation Temperature normalisation in eV
   * @param time_normalisation Time normalisation in seconds
   * @param coeffs A real-valued array of coefficients to be used in a 1D AMJUEL
   * reaction rate calculation.
   */
  AMJUEL1DDataOnDevice(const REAL &evolved_quantity_normalisation,
                       const REAL &density_normalisation,
                       const REAL &temperature_normalisation,
                       const REAL &time_normalisation,
                       const std::array<REAL, num_coeffs> &coeffs)
      : mult_const(density_normalisation * time_normalisation /
                   evolved_quantity_normalisation),
        temperature_normalisation(temperature_normalisation), coeffs(coeffs) {};

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
   *
   * @return A REAL-valued array of size 1 containing the calculated reaction
   * rate.
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

    auto log_temp =
        Kernel::log(fluid_temperature_dat * this->temperature_normalisation);

    std::array<REAL, num_coeffs> log_rate_arr;
    log_rate_arr[0] = 1.0;
    for (int i = 1; i < num_coeffs; i++) {
      log_rate_arr[i] = log_rate_arr[i - 1] * log_temp;
    }

    REAL log_rate = 0.0;
    for (int i = 0; i < num_coeffs; i++) {
      log_rate += this->coeffs[i] * log_rate_arr[i];
    }

    REAL rate = Kernel::exp(log_rate) * 1.0e-6;

    rate *= req_real_props.at(this->weight_ind, index, 0) * fluid_density_dat *
            this->mult_const;

    return std::array<REAL, 1>{rate};
  }

public:
  int fluid_density_ind, fluid_temperature_ind, weight_ind;
  REAL temperature_normalisation;
  REAL mult_const;
  std::array<REAL, num_coeffs> coeffs;
};

/**
 * @brief Reaction rate data calculation based on AMJUEL fits against ion/plasma
 * temperature.
 *
 * @tparam num_coeffs The number of coefficients needed for 1D AMJUEL reaction
 * rate calculation.
 */
template <int num_coeffs>
struct AMJUEL1DData
    : public ReactionDataBase<AMJUEL1DDataOnDevice<num_coeffs>> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 3> required_simple_real_props = {
      props.fluid_density, props.fluid_temperature, props.weight};

  /**
   * @brief Constructor for AMJUEL1DData
   *
   * @param evolved_quantity_normalisation Normalisation of the evolved quantity
   * (the one evolved with this rate)
   * @param density_normalisation Density normalisation constant in m^{-3}
   * @param temperature_normalisation Temperature normalisation in eV
   * @param time_normalisation Time normalisation in seconds
   * @param coeffs A real-valued array of coefficients to be used in a 1D AMJUEL
   * reaction rate calculation.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  AMJUEL1DData(const REAL &evolved_quantity_normalisation,
               const REAL &density_normalisation,
               const REAL &temperature_normalisation,
               const REAL &time_normalisation,
               const std::array<REAL, num_coeffs> &coeffs,
               std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<AMJUEL1DDataOnDevice<num_coeffs>>(
            Properties<REAL>(required_simple_real_props), properties_map) {

    this->on_device_obj = AMJUEL1DDataOnDevice<num_coeffs>(
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
