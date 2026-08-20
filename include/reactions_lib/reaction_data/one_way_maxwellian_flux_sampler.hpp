#ifndef REACTIONS_ONE_WAY_MAXWELLIAN_FLUX_SAMPLER_H
#define REACTIONS_ONE_WAY_MAXWELLIAN_FLUX_SAMPLER_H
#include "../particle_properties_map.hpp"
#include "../reaction_data.hpp"
#include "../utils.hpp"
#include <neso_particles.hpp>

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

  /**
   * Constructor for OneWayMaxwellianFluxOnDevice.
   *
   * @param norm_ratio The ratio of the temperature and kinetic energy
   * normalisations. Specifically kT/mv^2 where m is the mass of the ions, and T
   * and v are the temperature and velocity normalisation constants.
   */
  OneWayMaxwellianFluxOnDevice(const REAL &norm_ratio)
      : norm_ratio(norm_ratio) {};

  /**
   * @brief Solves the cubic equation used to find the maximum of
   * the rejection sampling function for the one way / truncated maxwellian
   * distribution
   *
   * @param d Ratio of flow speed to sigma (thermal spread) of the distribution
   * @return Solution of cubic equation of the form x^3 + p * x + q = 0 (where p
   * and q are calculated within this function).
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
   * @brief Gives the unnormalized value used in the one way / truncated
   * maxwellian rejection sampling algorithm
   *
   * @param d Ratio of flow speed to sigma (thermal spread) of the distribution
   * @param t Result of cardano_cubic_solver(d) or a sampled random number.
   * @return maximum of the rejection sampling function
   */
  static REAL rejection_function(REAL d, REAL t) {
    const REAL s = 1.0 - t;
    return t * NP::Kernel::exp(-((t / s - d) * (t / s - d)) / 2.0) /
           (s * s * s);
  }

  /**
   * @brief Samples a single value from a positive Maxwellian.
   *
   * @param drift Drift velocity due to fluid flow speed in the basis_pi
   * direction.
   * @param thermal_sigma Thermal velocity (derived from fluid temperature).
   * @param index Read-only accessor to a loop index (used by kernel.at()).
   * @param kernel The random number generator kernel.
   * @param sample_counter Marker used to select which component of the kernel
   * to access in kernel.at().
   * @param is_kernel_valid Boolean that stores the validity of the kernel as
   * returned by kernel.at(). If this is false then a value of 0.0 is returned.
   *
   * @return The sampled velocity value.
   */
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

  /**
   * @brief Function to calculate the sampled ion velocities from a one way /
   * truncated Maxwellian.
   *
   * @param index Read-only accessor to a loop index for a NP::ParticleLoop
   * inside which calc_data is called. NP::Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel
   *
   * @return A REAL-valued array of size 3 that contains the calculated sampled
   * ion velocities.
   */
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

  /**
   * @brief Constructor for OneWayMaxwellianFluxSampler.
   *
   * @param norm_ratio The ratio of the temperature and kinetic energy
   * normalisations. Specifically kT/mv^2 where m is the mass of the ions, and T
   * and v are the temperature and velocity normalisation constants
   * @param rng_kernel A shared pointer of a
   * NP::HostAtomicBlockKernelRNG<REAL> to be set as the rng_kernel in
   * ReactionDataBase.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  OneWayMaxwellianFluxSampler(
      const REAL &norm_ratio,
      std::shared_ptr<NP::HostAtomicBlockKernelRNG<REAL>> rng_kernel,
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * @brief Index the fluid flow speed, fluid temperature, surface basis
   * functions and the panic flag on the on-device object
   */
  void index_on_device_object();
};
}; // namespace VANTAGE::Reactions
#endif
