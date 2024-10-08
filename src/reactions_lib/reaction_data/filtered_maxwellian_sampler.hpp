#pragma once
#include "cross_sections/constant_rate_cs.hpp"
#include "particle_properties_map.hpp"
#include <array>
#include <neso_particles.hpp>
#include <neso_particles/containers/rng/host_atomic_block_kernel_rng.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <utils.hpp>
#include <vector>

using namespace NESO::Particles;
namespace Reactions {
using namespace ParticlePropertiesIndices;

namespace FILTERED_MAXWALLIAN_SAMPLER {

const auto props = ParticlePropertiesIndices::default_properties;

const std::vector<int> required_simple_real_props = {
    props.fluid_temperature, props.fluid_flow_speed, props.velocity};
} // namespace FILTERED_MAXWALLIAN_SAMPLER

/**
 * @brief A struct that contains data and calc_data functions that are to be
 * stored on and used on a SYCL device.
 *
 * @tparam ndim The velocity space dimensionality for both the particles and the
 * fields
 * @tparam CROSS_SECTION The typename corresponding to the cross-section class
 * used
 * @param norm_ratio The ratio of the temperature and kinetic energy
 * normalisations. Specifically kT/mv^2 where m is the mass of the ions, and T
 * and v are the temperature and velocity normalisation constants
 * @param cross_section Cross section object to be used in the rejection method
 * sampling
 */
template <size_t ndim, typename CROSS_SECTION>
struct FilteredMaxwellianOnDevice
    : public ReactionDataBaseOnDevice<ndim, HostAtomicBlockKernelRNG<REAL>> {
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
   */
  std::array<REAL, ndim>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Read<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename HostAtomicBlockKernelRNG<REAL>::KernelType &kernel) const {
    auto fluid_temperature_dat =
        req_real_props.at(this->fluid_temperature_ind, index, 0);

    bool accepted = false;

    std::array<REAL, ndim> sampled_vels;
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
    while (!accepted) {

      // Get the unit variance zero mean normal variates
      for (int i = 0; i < num_req_samples; i += 2) {

        auto current_samples = utils::box_muller_transform(
            kernel.at(index, i), kernel.at(index, i + 1));
        total_samples[i] = current_samples[0];
        total_samples[i + 1] = current_samples[1];
      };
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

      accepted = this->cross_section.accept_reject(
          std::sqrt(relative_vel_sq), kernel.at(index, num_req_samples));
    }

    return sampled_vels;
  }

public:
  int fluid_temperature_ind, fluid_flow_speed_ind, velocity_ind;
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
 * @param norm_ratio The ratio of the temperature and kinetic energy
 * normalisations. Specifically kT/mv^2 where m is the mass of the ions, and T
 * and v are the temperature and velocity normalisation constants
 * @param cross_section Cross section object to be used in the rejection method
 * sampling
 *
 */
template <size_t ndim, typename CROSS_SECTION = ConstantRateCrossSection>
struct FilteredMaxwellianSampler
    : public ReactionDataBase<ndim, HostAtomicBlockKernelRNG<REAL>> {

  FilteredMaxwellianSampler(
      const REAL &norm_ratio, CROSS_SECTION cross_section,
      std::shared_ptr<HostAtomicBlockKernelRNG<REAL>> rng_kernel)
      : ReactionDataBase<ndim, HostAtomicBlockKernelRNG<REAL>>(Properties<REAL>(
            FILTERED_MAXWALLIAN_SAMPLER::required_simple_real_props,
            std::vector<Species>{}, std::vector<int>{})),
        device_obj(FilteredMaxwellianOnDevice<ndim, CROSS_SECTION>(
            norm_ratio, cross_section)) {

    static_assert(std::is_base_of_v<AbstractCrossSection, CROSS_SECTION>,
                  "Template parameter CROSS_SECITON is not derived from "
                  "AbstractCrossSection...");
    auto props = FILTERED_MAXWALLIAN_SAMPLER::props;

    this->device_obj.fluid_flow_speed_ind =
        this->required_real_props.simple_prop_index(props.fluid_flow_speed);
    this->device_obj.fluid_temperature_ind =
        this->required_real_props.simple_prop_index(props.fluid_temperature);
    this->device_obj.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity);

    this->set_rng_kernel(rng_kernel);
  }

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
}; // namespace Reactions
