#ifndef REACTIONS_ADAS_DATA_H
#define REACTIONS_ADAS_DATA_H
#include "../particle_properties_map.hpp"
#include "../reaction_data.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/compute_target.hpp>
#include <neso_particles/device_buffers.hpp>
#include <vector>

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
    REAL fluid_dens_max = this->density_range[num_coeffs_n - 1];
    REAL fluid_temp_min = this->density_range[0];
    REAL fluid_temp_max = this->density_range[num_coeffs_T - 1];

    if (fluid_dens_min > fluid_density_dat) {
      closest_dens = {0, 1};
    } else if (fluid_dens_max < fluid_density_dat) {
      closest_dens = {num_coeffs_n - 2, num_coeffs_n - 1};
    } else {
      closest_dens = utils::calc_closest_point_indices(fluid_density_dat,
                                                       this->density_range);
    }

    if (fluid_temp_min > fluid_temperature_dat) {
      closest_temp = {0, 1};
    } else if (fluid_temp_max < fluid_temperature_dat) {
      closest_temp = {num_coeffs_n - 2, num_coeffs_n - 1};
    } else {
      closest_temp = utils::calc_closest_point_indices(fluid_temperature_dat,
                                                       this->density_range);
    }

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

    // fetch pointer and calculate indices for flattened vector
    auto coeff_index = [=](std::size_t t, std::size_t n) {
      return (t * num_coeffs_n) + n;
    };
    auto nd_coeffs_vec = nd_coeffs->ptr;

    // function values
    REAL f_n0_t0 = nd_coeffs_vec[coeff_index(t0, n0)];
    REAL f_n0_t1 = nd_coeffs_vec[coeff_index(t1, n0)];
    REAL f_n1_t0 = nd_coeffs_vec[coeff_index(t0, n1)];
    REAL f_n1_t1 = nd_coeffs_vec[coeff_index(t1, n1)];

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
  BufferDevice<REAL> *nd_coeffs;
};

template <int num_coeffs_T, int num_coeffs_n>
struct ADASData
    : public ReactionDataBase<ADASDataOnDevice<num_coeffs_T, num_coeffs_n>> {
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 3> required_simple_real_props = {
      props.fluid_density, props.fluid_temperature, props.weight};

  ADASData(const std::vector<REAL> &nd_coeffs,
           const std::vector<std::vector<REAL>> &ranges,
           const SYCLTargetSharedPtr &sycl_target,
           std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<ADASDataOnDevice<num_coeffs_T, num_coeffs_n>>(
            Properties<REAL>(required_simple_real_props), properties_map) {
    std::array<REAL, num_coeffs_T> temperature_range;
    for (std::size_t it = 0; it < num_coeffs_T; it++) {
      temperature_range[it] = ranges[0][it];
    }

    std::array<REAL, num_coeffs_n> density_range;
    for (std::size_t in = 0; in < num_coeffs_n; in++) {
      density_range[in] = ranges[1][in];
    }

    auto filled_array = [=]() {
      std::array<std::array<REAL, num_coeffs_n>, num_coeffs_T> coeffs;
      for (int iT = 0; iT < num_coeffs_T; iT++) {
        for (int in = 0; in < num_coeffs_n; in++) {
          coeffs[iT][in] = nd_coeffs[(iT * num_coeffs_n) + in];
        }
      }
      return coeffs;
    };

    this->on_device_obj = ADASDataOnDevice<num_coeffs_T, num_coeffs_n>(
        filled_array(), temperature_range, density_range);

    this->nd_coeff_array =
        std::make_shared<BufferDevice<REAL>>(sycl_target, nd_coeffs);

    this->index_on_device_object();
  }

  std::shared_ptr<BufferDevice<REAL>> nd_coeff_array;

  void index_on_device_object() {
    this->on_device_obj->fluid_density_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.fluid_density));

    this->on_device_obj->fluid_temperature_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.fluid_temperature));

    this->on_device_obj->weight_ind = this->required_real_props.find_index(
        this->properties_map.at(props.weight));

    this->on_device_obj->nd_coeffs = this->nd_coeff_array.get();
  };
};
} // namespace VANTAGE::Reactions

#endif