#ifndef REACTIONS_FILTERED_MAXWELLIAN_SAMPLER_H
#define REACTIONS_FILTERED_MAXWELLIAN_SAMPLER_H
#include "../cross_sections/constant_rate_cs.hpp"
#include "../particle_properties_map.hpp"
#include "../utils.hpp"
#include <iostream>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction data class for calculating velocity samples from a filtered
 * Maxwellian distribution given a fluid temperature and flow speed. The sampled
 * distribution is formally sigma(|v-u|)f_M(v), where sigma is a cross-section
 * evaluated at the relative speed |v-u| of the neutrals (v) and ions (u). The
 * filtering is performed using a rejection method.
 *
 * @tparam ndim The velocity space dimensionality for both the particles and the
 * fields
 * @tparam CROSS_SECTION The typename corresponding to the cross-section class
 * used
 */
template <size_t ndim, typename CROSS_SECTION>
struct FilteredMaxwellianOnDevice
    : public ReactionDataBaseOnDevice<ndim, HostAtomicBlockKernelRNG<REAL>> {

  /**
   * @brief Constructor for FilteredMaxwellianOnDevice.
   *
   * @param norm_ratio The ratio of the temperature and kinetic energy
   * normalisations. Specifically kT/mv^2 where m is the mass of the ions, and T
   * and v are the temperature and velocity normalisation constants
   * @param cross_section Cross section object to be used in the rejection method
   * sampling
   */
  FilteredMaxwellianOnDevice(const REAL &norm_ratio,
                             CROSS_SECTION cross_section)
      : norm_ratio(norm_ratio), cross_section(cross_section){};

  /**
   * @brief Function to calculate the sampled ion velocities from a filtered
   * Maxwellian
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel - assumed uniform
   *
   * @return A REAL-valued array of size ndim that contains the calculated sampled ion velocities.
   */
  std::array<REAL, ndim>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename HostAtomicBlockKernelRNG<REAL>::KernelType &kernel) const {
    auto fluid_temperature_dat =
        req_real_props.at(this->fluid_temperature_ind, index, 0);

    bool accepted = false;

    std::array<REAL, ndim> sampled_vels;
    for (int i = 0; i < ndim; i++) {
      sampled_vels[i] = 0;
    };
    // Sampling an even number of random numbers
    constexpr size_t num_req_samples = (ndim % 2 == 0) ? ndim : ndim + 1;

    std::array<REAL, num_req_samples> total_samples;

    std::array<REAL, ndim> neutral_vels;
    for (int i = 0; i < ndim; i++) {
      neutral_vels[i] = req_real_props.at(this->velocity_ind, index, i);
    }
    std::array<REAL, ndim> fluid_flows;
    for (int i = 0; i < ndim; i++) {
      fluid_flows[i] = req_real_props.at(this->fluid_flow_speed_ind, index, i);
    }
    int sample_counter = 0;
    bool is_kernel_valid = true;
    REAL rand1 = 0;
    REAL rand2 = 0;
    do {

      // Get the unit variance zero mean normal variates
      for (int i = 0; i < num_req_samples; i += 2) {

        rand1 = kernel.at(index, i, &is_kernel_valid);
        rand2 = kernel.at(index, i + 1, &is_kernel_valid);
        if (!is_kernel_valid) {
          break;
        }

        auto current_samples = utils::box_muller_transform(rand1, rand2);
        total_samples[i] = current_samples[0];
        total_samples[i + 1] = current_samples[1];
      };
      if (!is_kernel_valid) {
        req_int_props.at(this->panic_ind, index, 0) += 1;

        break;
      }
      // Calculate the relative velocity magnitude by rescaling the sampled
      // normal variables
      REAL relative_vel_sq = 0;
      for (int i = 0; i < ndim; i++) {
        sampled_vels[i] =
            fluid_temperature_dat * this->norm_ratio * total_samples[i] +
            fluid_flows[i];
        relative_vel_sq += (neutral_vels[i] - sampled_vels[i]) *
                           (neutral_vels[i] - sampled_vels[i]);
      }

      REAL relative_vel = std::sqrt(relative_vel_sq);
      REAL value_at = this->cross_section.get_value_at(relative_vel);
      REAL max_rate_val = this->cross_section.get_max_rate_val();

      rand1 = kernel.at(index, num_req_samples, &is_kernel_valid);
      if (!is_kernel_valid) {
        req_int_props.at(this->panic_ind, index, 0) += 1;

        break;
      }
      accepted = this->cross_section.accept_reject(relative_vel, rand1,
                                                   value_at, max_rate_val);

      sample_counter++;
    } while (!accepted);

    return sampled_vels;
  }

public:
  int fluid_temperature_ind, fluid_flow_speed_ind, velocity_ind, panic_ind;
  REAL norm_ratio;
  CROSS_SECTION cross_section;
};

/**
 * @brief Reaction data class for calculating velocity samples from a filtered
 * Maxwellian distribution given a fluid temperature and flow speed. The sampled
 * distribution is formally sigma(|v-u|)f_M(v), where sigma is a cross-section
 * evaluated at the relative speed |v-u| of the neutrals (v) and ions (u). The
 * filtering is performed using a rejection method.
 *
 * @tparam ndim The velocity space dimensionality for both the particles and the
 * fields
 * @tparam CROSS_SECTION The typename corresponding to the cross-section class
 * used
 */
template <size_t ndim, typename CROSS_SECTION = ConstantRateCrossSection>
struct FilteredMaxwellianSampler
    : public ReactionDataBase<ndim, HostAtomicBlockKernelRNG<REAL>> {

  constexpr static auto props = default_properties;

  constexpr static auto required_simple_real_props = std::array<int, 3>{
      props.fluid_temperature, props.fluid_flow_speed, props.velocity};

  constexpr static auto required_simple_int_props =
      std::array<int, 1>{props.panic};

  /**
   * @brief Constructor for FilteredMaxwellianSampler.
   *
   * @param norm_ratio The ratio of the temperature and kinetic energy
   * normalisations. Specifically kT/mv^2 where m is the mass of the ions, and T
   * and v are the temperature and velocity normalisation constants
   * @param cross_section Cross section object to be used in the rejection method
   * sampling
   * @param rng_kernel A shared pointer of a HostAtomicBlockKernelRNG<REAL> to be
   * set as the rng_kernel in ReactionDataBase.
   * @param properties_map (Optional) A std::map<int, std::string> object to be used when
   * remapping property names.
   */
  FilteredMaxwellianSampler(
      const REAL &norm_ratio, CROSS_SECTION cross_section,
      std::shared_ptr<HostAtomicBlockKernelRNG<REAL>> rng_kernel,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<ndim, HostAtomicBlockKernelRNG<REAL>>(
            Properties<INT>(required_simple_int_props),
            Properties<REAL>(required_simple_real_props),
            properties_map),
        device_obj(FilteredMaxwellianOnDevice<ndim, CROSS_SECTION>(
            norm_ratio, cross_section)) {

    static_assert(std::is_base_of_v<AbstractCrossSection, CROSS_SECTION>,
                  "Template parameter CROSS_SECITON is not derived from "
                  "AbstractCrossSection...");

    this->device_obj.fluid_flow_speed_ind =
        this->required_real_props.simple_prop_index(props.fluid_flow_speed,
                                                    this->properties_map);
    this->device_obj.fluid_temperature_ind =
        this->required_real_props.simple_prop_index(props.fluid_temperature,
                                                    this->properties_map);
    this->device_obj.panic_ind = this->required_int_props.simple_prop_index(
        props.panic, this->properties_map);
    this->device_obj.velocity_ind = this->required_real_props.simple_prop_index(
        props.velocity, this->properties_map);

    this->set_rng_kernel(rng_kernel);
  }

  /**
   * \overload
   * @brief Constructor which sets default values for the
   * cross_section and properties_map.
   *
   * @param norm_ratio The ratio of the temperature and kinetic energy
   * normalisations. Specifically kT/mv^2 where m is the mass of the ions, and T
   * and v are the temperature and velocity normalisation constants
   * @param rng_kernel A shared pointer of a HostAtomicBlockKernelRNG<REAL> to be
   * set as the rng_kernel in ReactionDataBase.
   */
  FilteredMaxwellianSampler(
      const REAL &norm_ratio,
      std::shared_ptr<HostAtomicBlockKernelRNG<REAL>> rng_kernel)
      : FilteredMaxwellianSampler(norm_ratio, ConstantRateCrossSection(0.0),
                                  rng_kernel) {}

private:
  FilteredMaxwellianOnDevice<ndim, CROSS_SECTION> device_obj;

public:
  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */

  FilteredMaxwellianOnDevice<ndim, CROSS_SECTION> get_on_device_obj() {
    return this->device_obj;
  }
};
}; // namespace VANTAGE::Reactions
#endif