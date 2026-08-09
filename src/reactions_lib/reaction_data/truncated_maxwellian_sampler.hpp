#ifndef REACTIONS_TRUNCATED_MAXWELLIAN_SAMPLER_H
#define REACTIONS_TRUNCATED_MAXWELLIAN_SAMPLER_H
#include "../reaction_data.hpp"
#include "../particle_properties_map.hpp"
#include "../utils.hpp"
#include <iostream>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
/**
 * @brief On device: Reaction data class for calculating velocity samples from a
 * truncated Maxwellian distribution given a fluid temperature and flow speed.
 * The truncated distribution is generated using a rejection sampling method outlined 
 * in https://doi.org/10.1088/0031-8949/90/1/015204.
 *
 */
struct TruncatedMaxwellianOnDevice
    : public ReactionDataBaseOnDevice<3, HostAtomicBlockKernelRNG<REAL>> {

  TruncatedMaxwellianOnDevice() = default;


  TruncatedMaxwellianOnDevice(const REAL &norm_ratio): norm_ratio(norm_ratio) {};

  /**
  * @brief solves the cubic equation used to find the maximum of 
  * the rejection sampling function for the truncated maxwellian distribution
  *
  * @param d Ratio of flow speed to sigma (thermal spread) of the distribution
  * @return maximum of the rejection sampling function
  */
  inline static REAL solve_cubic(REAL d) {
    const REAL ca = 2.0;
    const REAL cb = -4.0 - d;
    const REAL cc = d;
    const REAL cd = 1.0;

    const REAL p = (3.0 * ca * cc - cb * cb) / (3.0 * ca * ca);
    const REAL q = (2.0 * cb * cb * cb - 9.0 * ca * cb * cc +
                    27.0 * ca * ca * cd) /
                   (27.0 * ca * ca * ca);

    const REAL sqrt_term = Kernel::sqrt(-p / 3.0);
    const REAL root = 2.0 * sqrt_term *
                      Kernel::cos(sycl::acos((3.0 * q) / (2.0 * p * sqrt_term)) /
                                  3.0 - 2.0 * REAL(M_PI) / 3.0);
    return root - cb / (3.0 * ca);
  }

  /**
  * @brief Gives the unnormalized function used in the truncated maxwellian rejection sampling algorithm
  *
  * @param d Ratio of flow speed to sigma (thermal spread) of the distribution
  * @return maximum of the rejection sampling function
  */  
  inline static REAL rejection_function(REAL d, REAL t) {
    const REAL s = 1.0 - t;
    return t * Kernel::exp(-((t / s - d) * (t / s - d)) / 2.0) / (s * s * s);
  }

  REAL sample_drifting_maxwellian(
      REAL drift, REAL thermal_sigma, const Access::LoopIndex::Read &index,
      typename HostAtomicBlockKernelRNG<REAL>::KernelType &kernel,
      int &sample_counter, bool &is_kernel_valid) const {
    if (!is_kernel_valid) {
      return 0.0;
    }

    REAL rand1 = kernel.at(index, sample_counter++, &is_kernel_valid);
    REAL rand2 = kernel.at(index, sample_counter++, &is_kernel_valid);
    if (!is_kernel_valid) {
      return 0.0;
    }

    auto normal_samples = utils::box_muller_transform(rand1, rand2);
    return drift + thermal_sigma * normal_samples[0];
  }

  REAL sample_positive_maxwellian(
      REAL drift, REAL thermal_sigma, const Access::LoopIndex::Read &index,
      typename HostAtomicBlockKernelRNG<REAL>::KernelType &kernel,
      int &sample_counter, bool &is_kernel_valid) const {
    if (!is_kernel_valid || thermal_sigma <= REAL(0.0)) {
      return 0.0;
    }

    const REAL d = drift / thermal_sigma;
    const REAL maxval = rejection_function(d, solve_cubic(d));
    REAL candidate = 0.0;
    REAL compare = 0.0;

    do {
      candidate = kernel.at(index, sample_counter++, &is_kernel_valid);
      compare = maxval * kernel.at(index, sample_counter++, &is_kernel_valid);
      if (!is_kernel_valid) {
        return 0.0;
      }

      REAL t = candidate;
      
      if (compare <= rejection_function(d, t)) {
        return (t / (1.0 - t)) * thermal_sigma;
      }
    } while (true);
  }


  std::array<REAL, 3>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename HostAtomicBlockKernelRNG<REAL>::KernelType &kernel) const {
    const REAL fluid_temperature_dat =
        req_real_props.at(this->fluid_temperature_ind, index, 0);

    std::array<REAL, 3> fluid_flow_speed{};
    std::array<REAL, 3> basis_e1{};
    std::array<REAL, 3> basis_e2{};
    std::array<REAL, 3> basis_pi{};

    for (int i = 0; i < 3; i++) {
      fluid_flow_speed[i] = req_real_props.at(this->fluid_flow_speed_ind, index, i);
      basis_e1[i] = req_real_props.at(this->basis_e1_ind, index, i);
      basis_e2[i] = req_real_props.at(this->basis_e2_ind, index, i);
      basis_pi[i] = req_real_props.at(this->basis_pi_ind, index, i);
    }

    REAL drift_e1 = 0.0;
    REAL drift_e2 = 0.0;
    REAL drift_pi = 0.0;
    for (int i = 0; i < 3; i++) {
      drift_e1 += fluid_flow_speed[i] * basis_e1[i];
      drift_e2 += fluid_flow_speed[i] * basis_e2[i];
      drift_pi += fluid_flow_speed[i] * basis_pi[i];
    }

    const REAL thermal_sigma = Kernel::sqrt(fluid_temperature_dat* this->norm_ratio);

    bool is_kernel_valid = true;
    int sample_counter = 0;

    REAL rand1 = kernel.at(index, sample_counter++, &is_kernel_valid);
    REAL rand2 = kernel.at(index, sample_counter++, &is_kernel_valid);

    if (!is_kernel_valid) {
      req_int_props.at(this->panic_ind, index, 0) += 1;
      return std::array<REAL, 3>{};
    }

    auto normal_samples = utils::box_muller_transform(rand1, rand2);
    const REAL sample_e1 = drift_e1 + thermal_sigma * normal_samples[0];
    const REAL sample_e2 = drift_e2 + thermal_sigma * normal_samples[1];


    /*const REAL sample_e1 = sample_drifting_maxwellian(
        drift_e1, thermal_sigma, index, kernel, sample_counter, is_kernel_valid);
    if (!is_kernel_valid) {
      req_int_props.at(this->panic_ind, index, 0) += 1;
      return std::array<REAL, 3>{};
    }

    const REAL sample_e2 = sample_drifting_maxwellian(
        drift_e2, thermal_sigma, index, kernel, sample_counter, is_kernel_valid);
    if (!is_kernel_valid) {
      req_int_props.at(this->panic_ind, index, 0) += 1;
      return std::array<REAL, 3>{};
    }*/

    const REAL sample_pi = sample_positive_maxwellian(
        drift_pi, thermal_sigma, index, kernel, sample_counter, is_kernel_valid);
    if (!is_kernel_valid) {
      req_int_props.at(this->panic_ind, index, 0) += 1;
      return std::array<REAL, 3>{};
    }

    std::array<REAL, 3> sampled_vels{};
    for (int i = 0; i < 3; i++) {
      sampled_vels[i] = sample_e1 * basis_e1[i] + sample_e2 * basis_e2[i] +
                        sample_pi * basis_pi[i];
    }

    return sampled_vels;
  }


  public:
  int fluid_flow_speed_ind, fluid_temperature_ind,
      basis_e1_ind, basis_e2_ind, basis_pi_ind, panic_ind;
  REAL norm_ratio;
  };

/**
 * @brief Reaction data class for sampling a velocity vector from a drifting
 * Maxwellian in the tangential directions and a positive/truncated Maxwellian
 * along the surface-normal direction.
 *
 */
struct TruncatedMaxwellianSampler
    : public ReactionDataBase<TruncatedMaxwellianOnDevice, 3,
                              HostAtomicBlockKernelRNG<REAL>> {

  constexpr static auto props = default_properties;

  constexpr static auto required_simple_real_props = std::array<int, 5>{
      props.fluid_flow_speed, props.fluid_temperature,
      props.surface_basis_e1, props.surface_basis_e2, props.surface_basis_pi};

  constexpr static auto required_simple_int_props =
      std::array<int, 1>{props.panic};

  TruncatedMaxwellianSampler(
      const REAL &norm_ratio,
      std::shared_ptr<HostAtomicBlockKernelRNG<REAL>> rng_kernel,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<TruncatedMaxwellianOnDevice, 3,
                         HostAtomicBlockKernelRNG<REAL>>(
            Properties<INT>(required_simple_int_props),
            Properties<REAL>(required_simple_real_props), properties_map) {
    this->on_device_obj = TruncatedMaxwellianOnDevice(norm_ratio);
    this->set_rng_kernel(rng_kernel);
    this->index_on_device_object();
  }

  void index_on_device_object() {
    this->on_device_obj->fluid_flow_speed_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.fluid_flow_speed));

    this->on_device_obj->fluid_temperature_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.fluid_temperature));

    this->on_device_obj->basis_e1_ind = this->required_real_props.find_index(
        this->properties_map.at(props.surface_basis_e1));

    this->on_device_obj->basis_e2_ind = this->required_real_props.find_index(
        this->properties_map.at(props.surface_basis_e2));

    this->on_device_obj->basis_pi_ind = this->required_real_props.find_index(
        this->properties_map.at(props.surface_basis_pi));

    this->on_device_obj->panic_ind = this->required_int_props.find_index(
        this->properties_map.at(props.panic));
  };
};
}; // namespace VANTAGE::Reactions
#endif
