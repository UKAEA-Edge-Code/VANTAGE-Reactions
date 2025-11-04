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
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
// template <std::size_t num_coeffs_T, std::size_t num_coeffs_n>
struct ADASDataOnDevice : public ReactionDataBaseOnDevice<> {
  ADASDataOnDevice() = default;

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

    std::size_t closest_dens, closest_temp;

    auto dens_range_vec_ = this->dens_range_buf;
    auto temp_range_vec_ = this->temp_range_buf;

    REAL fluid_dens_min = dens_range_vec_[0];
    REAL fluid_dens_max = dens_range_vec_[this->num_coeffs_n - 1];
    REAL fluid_temp_min = temp_range_vec_[0];
    REAL fluid_temp_max = temp_range_vec_[this->num_coeffs_T - 1];

    closest_dens = 0;
    // utils::ndim_interp::calc_closest_point_index(
    // fluid_density_dat, dens_range_vec_, this->num_coeffs_n);
    closest_temp = 0;
    // utils::ndim_interp::calc_closest_point_index(
    // fluid_temperature_dat, temp_range_vec_, this->num_coeffs_T);

    // indices
    std::size_t t0 = closest_temp;
    std::size_t t1 = closest_temp + 1;
    std::size_t n0 = closest_dens;
    std::size_t n1 = closest_dens + 1;

    // values
    REAL temp0 = temp_range_vec_[t0];
    REAL temp1 = temp_range_vec_[t1];
    REAL dens0 = dens_range_vec_[n0];
    REAL dens1 = dens_range_vec_[n1];

    // fetch pointer and calculate indices for flattened vector
    auto coeff_index = [=](std::size_t t, std::size_t n) {
      return (t * this->num_coeffs_n) + n;
    };
    auto nd_coeffs_vec = this->nd_coeffs_buf;

    // function values
    REAL f_n0_t0 = nd_coeffs_vec[coeff_index(t0, n0)];
    REAL f_n0_t1 = nd_coeffs_vec[coeff_index(t1, n0)];
    REAL f_n1_t0 = nd_coeffs_vec[coeff_index(t0, n1)];
    REAL f_n1_t1 = nd_coeffs_vec[coeff_index(t1, n1)];

    REAL f_n0_t = 1.0;
    // utils::ndim_interp::linear_interp(fluid_temperature_dat, temp0, temp1,
    //               f_n0_t0, f_n0_t1);
    REAL f_n1_t = 1.0;
    // utils::ndim_interp::linear_interp(fluid_temperature_dat, temp0, temp1,
    //                                f_n1_t0, f_n1_t1);

    REAL f_n_t = 1.0;
    // utils::ndim_interp::linear_interp(fluid_density_dat, dens0, dens1,
    // f_n0_t, f_n1_t);

    REAL rate = f_n_t;

    // rate *= weight_dat;

    return std::array<REAL, 1>{rate};
  }

public:
  int fluid_density_ind, fluid_temperature_ind, weight_ind;
  REAL *nd_coeffs_buf;
  REAL *temp_range_buf;
  REAL *dens_range_buf;
  std::size_t num_coeffs_n;
  std::size_t num_coeffs_T;
};

struct ADASData : public ReactionDataBase<ADASDataOnDevice> {
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 3> required_simple_real_props = {
      props.fluid_density, props.fluid_temperature, props.weight};

  ADASData(const std::vector<REAL> &nd_coeffs,
           const std::vector<std::vector<REAL>> &ranges,
           const SYCLTargetSharedPtr &sycl_target,
           std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<ADASDataOnDevice>(
            Properties<REAL>(required_simple_real_props), properties_map) {
    this->on_device_obj = ADASDataOnDevice();

    this->index_on_device_object();

    this->nd_coeff_array =
        std::make_shared<BufferDevice<REAL>>(sycl_target, nd_coeffs);

    this->temp_range_buf =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges[0]);

    this->dens_range_buf =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges[1]);

    this->on_device_obj->nd_coeffs_buf = this->nd_coeff_array->ptr;

    this->on_device_obj->temp_range_buf = this->temp_range_buf->ptr;

    this->on_device_obj->dens_range_buf = this->dens_range_buf->ptr;

    this->on_device_obj->num_coeffs_n = this->dens_range_buf->size;

    this->on_device_obj->num_coeffs_T = this->temp_range_buf->size;
  }

  std::shared_ptr<BufferDevice<REAL>> nd_coeff_array;
  std::shared_ptr<BufferDevice<REAL>> temp_range_buf;
  std::shared_ptr<BufferDevice<REAL>> dens_range_buf;

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
