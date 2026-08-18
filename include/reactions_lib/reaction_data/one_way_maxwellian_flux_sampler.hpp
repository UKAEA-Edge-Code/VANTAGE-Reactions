#ifndef REACTIONS_ONE_WAY_MAXWELLIAN_FLUX_SAMPLER_H
#define REACTIONS_ONE_WAY_MAXWELLIAN_FLUX_SAMPLER_H
#include "../particle_properties_map.hpp"
#include "../reaction_data.hpp"
#include "../utils.hpp"
#include <iostream>
#include <neso_particles.hpp>
#include <vector>

namespace VANTAGE::Reactions {
/**
 * @brief On device: Reaction data class for calculating velocity samples from a
 * one way / truncated Maxwellian distribution given a fluid temperature and
 * flow speed. The one way / truncated distribution is generated using a
 * rejection sampling method outlined in
 * https://doi.org/10.1088/0031-8949/90/1/015204.
 *
 */
struct OneWayMaxwellianFluxOnDevice
    : public ReactionDataBaseOnDevice<3, NP::HostAtomicBlockKernelRNG<REAL>> {

  OneWayMaxwellianFluxOnDevice() = default;

  OneWayMaxwellianFluxOnDevice(const REAL &norm_ratio)
      : norm_ratio(norm_ratio) {};

  /**
   * @brief solves the cubic equation used to find the maximum of
   * the rejection sampling function for the one way / truncated maxwellian
   * distribution
   *
   * @param d Ratio of flow speed to sigma (thermal spread) of the distribution
   * @return maximum of the rejection sampling function
   */
  static REAL cardano_cubic_solver(REAL d) {
    const REAL coef_a = 2.0;
    const REAL coef_b = -4.0 - d;
    const REAL coef_c = d;
    const REAL coef_d = 1.0;
    const REAL coef_a_sq = coef_a * coef_a;

    const REAL p =
        (3.0 * coef_a * coef_c - coef_b * coef_b) / (3.0 * coef_a_sq);
    const REAL q =
        (2.0 * coef_b * coef_b * coef_b - 9.0 * coef_a * coef_b * coef_c +
         27.0 * coef_a_sq * coef_d) /
        (27.0 * coef_a_sq * coef_a);

    const REAL sqrt_term = NP::Kernel::sqrt(-p / 3.0);
    const REAL root =
        2.0 * sqrt_term *
        NP::Kernel::cos(sycl::acos((3.0 * q) / (2.0 * p * sqrt_term)) / 3.0 -
                        2.0 * REAL(M_PI) / 3.0);
    return root - coef_b / (3.0 * coef_a);
  }

  /**
   * @brief Gives the unnormalized function used in the one way / truncated
   * maxwellian rejection sampling algorithm
   *
   * @param d Ratio of flow speed to sigma (thermal spread) of the distribution
   * @return maximum of the rejection sampling function
   */
  static REAL rejection_function(REAL d, REAL t) {
    const REAL s = 1.0 - t;
    return t * NP::Kernel::exp(-((t / s - d) * (t / s - d)) / 2.0) /
           (s * s * s);
  }

  REAL sample_positive_maxwellian(
      REAL drift, REAL thermal_sigma, const NP::Access::LoopIndex::Read &index,
      typename NP::HostAtomicBlockKernelRNG<REAL>::KernelType &kernel,
      int &sample_counter, bool &is_kernel_valid) const {
    if (!is_kernel_valid || thermal_sigma <= REAL(0.0)) {
      return 0.0;
    }

    const REAL d = drift / thermal_sigma;
    const REAL maxval = rejection_function(d, cardano_cubic_solver(d));
    REAL candidate = 0.0;
    REAL compare = 0.0;

    do {
      candidate = kernel.at(index, sample_counter++, &is_kernel_valid);
      compare = maxval * kernel.at(index, sample_counter++, &is_kernel_valid);
      if (!is_kernel_valid) {
        return 0.0;
      }

      if (compare <= rejection_function(d, candidate)) {
        return (candidate / (1.0 - candidate)) * thermal_sigma;
      }
    } while (true);
  }

  std::array<REAL, 3> calc_data(
      const NP::Access::LoopIndex::Read &index,
      const NP::Access::SymVector::Write<INT> &req_int_props,
      const NP::Access::SymVector::Read<REAL> &req_real_props,
      typename NP::HostAtomicBlockKernelRNG<REAL>::KernelType &kernel) const {
    const REAL fluid_temperature_dat =
        req_real_props.at(this->fluid_temperature_ind, index, 0);

    std::array<REAL, 3> fluid_flow_speed{};
    std::array<REAL, 3> basis_e1{};
    std::array<REAL, 3> basis_e2{};
    std::array<REAL, 3> basis_pi{};

    for (int i = 0; i < 3; i++) {
      fluid_flow_speed[i] =
          req_real_props.at(this->fluid_flow_speed_ind, index, i);
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

    const REAL thermal_sigma =
        NP::Kernel::sqrt(fluid_temperature_dat * this->norm_ratio);

    bool is_kernel_valid = true;
    int sample_counter = 0;

    REAL rand1 = kernel.at(index, sample_counter++, &is_kernel_valid);
    REAL rand2 = kernel.at(index, sample_counter++, &is_kernel_valid);

    if (!is_kernel_valid) {
      req_int_props.at(this->panic_ind, index, 0) += 1;
      return std::array<REAL, 3>{0.0, 0.0, 0.0};
    }

    auto normal_samples = utils::box_muller_transform(rand1, rand2);
    const REAL sample_e1 = drift_e1 + thermal_sigma * normal_samples[0];
    const REAL sample_e2 = drift_e2 + thermal_sigma * normal_samples[1];

    const REAL sample_pi =
        sample_positive_maxwellian(drift_pi, thermal_sigma, index, kernel,
                                   sample_counter, is_kernel_valid);
    if (!is_kernel_valid) {
      req_int_props.at(this->panic_ind, index, 0) += 1;
      return std::array<REAL, 3>{0.0, 0.0, 0.0};
    }

    std::array<REAL, 3> sampled_vels{};
    for (int i = 0; i < 3; i++) {
      sampled_vels[i] = sample_e1 * basis_e1[i] + sample_e2 * basis_e2[i] +
                        sample_pi * basis_pi[i];
    }

    return sampled_vels;
  }

public:
  int fluid_flow_speed_ind, fluid_temperature_ind, basis_e1_ind, basis_e2_ind,
      basis_pi_ind, panic_ind;
  REAL norm_ratio;
};

/**
 * @brief Reaction data class for sampling a velocity vector from a drifting
 * Maxwellian in the tangential directions and a one way / truncated Maxwellian
 * along the surface-normal direction.
 * surface_basis_e1, surface_basis_e2, surface_basis_pi: The three basis
 * vectors that define the surface tangential and normal directions.
 * These are assumed to be orthonormal. See EIRENE docs section 1.5 Recycling
 * surface sources for more details.
 * @param norm_ratio The ratio of the temperature and kinetic energy
 * normalisations. Specifically kT/mv^2 where m is the mass of the ions, and T
 * and v are the temperature and velocity normalisation constants
 * @param rng_kernel A shared pointer of a
 * NP::HostAtomicBlockKernelRNG<REAL> to be set as the rng_kernel in
 * ReactionDataBase.
 * @param properties_map (Optional) A std::map<int, std::string> object to be
 * used when remapping property names.
 */
struct OneWayMaxwellianFluxSampler
    : public ReactionDataBase<OneWayMaxwellianFluxOnDevice, 3,
                              NP::HostAtomicBlockKernelRNG<REAL>> {

  constexpr static auto props = default_properties;

  constexpr static auto required_simple_real_props = std::array<int, 5>{
      props.fluid_flow_speed, props.fluid_temperature, props.surface_basis_e1,
      props.surface_basis_e2, props.surface_basis_pi};

  constexpr static auto required_simple_int_props =
      std::array<int, 1>{props.panic};

  OneWayMaxwellianFluxSampler(
      const REAL &norm_ratio,
      std::shared_ptr<NP::HostAtomicBlockKernelRNG<REAL>> rng_kernel,
      std::map<int, std::string> properties_map = get_default_map());

  void index_on_device_object();
};
}; // namespace VANTAGE::Reactions
#endif
