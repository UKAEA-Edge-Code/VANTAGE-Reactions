#ifndef REACTIONS_ADAS_DATA_H
#define REACTIONS_ADAS_DATA_H
#include "../particle_properties_map.hpp"
#include "../reaction_data.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include <array>
#include <cmath>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
template <std::size_t num_coeffs_T, std::size_t num_coeffs_n>
struct ADASDataOnDevice : public ReactionDataBaseOnDevice<> {
  ADASDataOnDevice() = default;

  ADASDataOnDevice(
      const std::array<std::array<REAL, num_coeffs_n>, num_coeffs_T> &coeffs,
      const std::array<REAL, num_coeffs_T> temperature_range,
      const std::array<REAL, num_coeffs_n> density_range)
      : coeffs(coeffs), density_range(density_range),
        temperature_range(temperature_range) {};

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
    auto weight_dat = req_real_props.at(this->weight_ind, index, 0);

    std::array<std::size_t, 2> closest_dens, closest_temp;

    REAL fluid_dens_min = this->density_range[0];
    REAL fluid_dens_max = this->density_range[num_coeffs_n-1];
    REAL fluid_temp_min = this->density_range[0];
    REAL fluid_temp_max = this->density_range[num_coeffs_T-1];

    if (fluid_dens_min > fluid_density_dat) { closest_dens = {0, 1}; }
    else if (fluid_dens_max < fluid_density_dat) { closest_dens = {num_coeffs_n-2, num_coeffs_n-1}; }
    else { closest_dens = utils::calc_closest_point_indices(fluid_density_dat, this->density_range); }

    if (fluid_temp_min > fluid_temperature_dat) { closest_temp = {0, 1}; }
    else if (fluid_temp_max < fluid_temperature_dat) { closest_temp = {num_coeffs_n-2, num_coeffs_n-1}; }
    else { closest_temp =utils::calc_closest_point_indices(fluid_temperature_dat, this->density_range); }

    // indices
    std::size_t t0 = closest_temp[0];
    std::size_t t1 = closest_temp[1];
    std::size_t n0 = closest_dens[0];
    std::size_t n1 = closest_dens[1];

    // values
    REAL temp0 = this->temperature_range[t0];
    REAL temp1 = this->temperature_range[t1];
    REAL dens0 = this->density_range[n0];
    REAL dens1 = this->density_range[n1];

    // function values
    REAL f_n0_t0 = coeffs[t0][n0];
    REAL f_n0_t1 = coeffs[t1][n0];
    REAL f_n1_t0 = coeffs[t0][n1];
    REAL f_n1_t1 = coeffs[t1][n1];

    REAL f_n0_t = utils::linear_interp(fluid_temperature_dat, temp0, temp1,
                                      f_n0_t0, f_n0_t1);
    REAL f_n1_t = utils::linear_interp(fluid_temperature_dat, temp0, temp1,
                                      f_n1_t0, f_n1_t1);

    REAL f_n_t =
        utils::linear_interp(fluid_density_dat, dens0, dens1, f_n0_t, f_n1_t);

    REAL rate = f_n_t;

    // rate *= weight_dat;

    return std::array<REAL, 1>{rate};
  }

public:
  int fluid_density_ind, fluid_temperature_ind, weight_ind;
  std::array<std::array<REAL, num_coeffs_n>, num_coeffs_T> coeffs;
  std::array<REAL, num_coeffs_n> density_range;
  std::array<REAL, num_coeffs_T> temperature_range;
};

template <int num_coeffs_T, int num_coeffs_n>
struct ADASData
    : public ReactionDataBase<ADASDataOnDevice<num_coeffs_T, num_coeffs_n>> {
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 3> required_simple_real_props = {
      props.fluid_density, props.fluid_temperature, props.weight};

  ADASData(
      const std::array<std::array<REAL, num_coeffs_n>, num_coeffs_T> &coeffs,
      const std::array<REAL, num_coeffs_T> &temperature_range,
      const std::array<REAL, num_coeffs_n> &density_range,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<ADASDataOnDevice<num_coeffs_T, num_coeffs_n>>(
            Properties<REAL>(required_simple_real_props), properties_map) {
    this->on_device_obj = ADASDataOnDevice<num_coeffs_T, num_coeffs_n>(
        coeffs, temperature_range, density_range);

    this->index_on_device_object();
  }

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
} // namespace VANTAGE::Reactions

#endif