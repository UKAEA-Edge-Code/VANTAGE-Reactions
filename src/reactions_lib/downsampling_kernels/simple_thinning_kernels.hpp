#ifndef REACTIONS_SIMPLE_THINNING_H
#define REACTIONS_SIMPLE_THINNING_H

#include "reactions_lib/downsampling_base.hpp"
#include <cmath>
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * Simple thinning sets particle weights to either k*weight with the probability
 * 1/k, and otherwise deletes the particle. This makes it conserve weight on
 * average, but otherwise does not have any conservation properties.
 */

/**
 * @brief On-device simple thinning kernel, requires an RNG kernel with a single
 * per particle uniform random variate
 *
 */
struct SimpleThinningOnDevice
    : DownsamplingKernelOnDeviceBase<0, HostPerParticleBlockRNG<REAL>> {

  SimpleThinningOnDevice() = default;

  /**
   * @brief Constructor for SimpleThinningOnDevice object
   *
   * @param thinning_ratio The probability of the particle being kept and its
   * weight increased by 1/thinning_ratio
   */
  SimpleThinningOnDevice(REAL thinning_ratio)
      : inverse_thinning_ratio(1 / thinning_ratio),
        thinning_ratio(thinning_ratio) {};

  /**
   * @brief Apply the thinning algorithm
   *
   * @param index LoopIndex accessor used for linear indexing
   * @param req_int_props SymVector Write access to required integer properties
   * @param req_real_props SymVector Write access to required real properties
   * @param rng_kernel RNG kernel access, if required
   */
  void apply_no_red(
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Write<REAL> &req_real_props,
      typename HostPerParticleBlockRNG<REAL>::KernelType &rng_kernel) const {

    bool is_kernel_valid = true;
    bool accepted =
        rng_kernel.at(index, 0, &is_kernel_valid) < 1 - this->thinning_ratio;
    req_int_props.at(this->panic_ind, 0) += is_kernel_valid ? 1 : 0;
    req_real_props.at(this->weight_ind, 0) *=
        accepted ? 0 : this->inverse_thinning_ratio;
    return;
  }

public:
  int weight_ind, panic_ind;
  REAL inverse_thinning_ratio, thinning_ratio;
};

/**
 * @brief Host-side simple thinning kernels, taking a thinning ratio < 0,
 * representing the probability of a particle being kept after thinning.
 *
 * Required properties are the particle weight and the panic flag (used to
 * report rng sampling issues)
 */
struct SimpleThinningKernels
    : DownsamplingKernelBase<DownsamplingMode::thinning,
                             DownsamplingReductionKernelOnDeviceBase<0, 0, 0>,
                             SimpleThinningOnDevice> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};

  constexpr static std::array<int, 1> required_simple_int_props = {props.panic};

  /**
   * @brief Simple thinning kernels constructor
   *
   * @param thinning_ratio The probability of the particle being kept and its
   * weight increased by 1/thinning_ratio
   * @param rng_kernel Uniform variate HostPerParticleBlockRNG kernel used to
   * sample the random number for comparison with the thinning ratio
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names
   */
  SimpleThinningKernels(
      REAL thinning_ratio,
      std::shared_ptr<HostPerParticleBlockRNG<REAL>> rng_kernel,
      std::map<int, std::string> properties_map = get_default_map())
      : DownsamplingKernelBase<DownsamplingMode::thinning,
                               DownsamplingReductionKernelOnDeviceBase<0, 0, 0>,
                               SimpleThinningOnDevice>(
            Properties<INT>(required_simple_int_props),
            Properties<REAL>(required_simple_real_props), properties_map) {

    this->set_rng_kernel(rng_kernel);
    this->downsampling_on_device_obj = SimpleThinningOnDevice(thinning_ratio);
    this->reduction_on_device_obj =
        DownsamplingReductionKernelOnDeviceBase<0, 0, 0>();

    this->downsampling_on_device_obj->weight_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.weight));

    this->downsampling_on_device_obj->panic_ind =
        this->required_int_props.find_index(
            this->properties_map.at(props.panic));
  }
};

/**
 * @brief Helper function for generating a simple thinning strategy
 *
 * @param template_group The template group sharing the domain and sycl target
 * of the particle group to which the transformation strategy is to be applied
 * @param thinning_ratio The probability of the particle being kept and its
 * weight increased by 1/thinning_ratio
 * @param rng_kernel Uniform variate HostPerParticleBlockRNG kernel used to
 * sample the random number for comparison with the thinning ratio
 * @param properties_map (Optional) A std::map<int, std::string> object to be
 * used when remapping property names
 */
inline std::shared_ptr<TransformationStrategy> make_simple_thinning_strategy(
    ParticleGroupSharedPtr template_group, REAL thinning_ratio,
    std::shared_ptr<HostPerParticleBlockRNG<REAL>> rng_kernel,
    const std::map<int, std::string> &properties_map = get_default_map()) {

  auto r = std::make_shared<DownsamplingStrategy<SimpleThinningKernels>>(
      template_group,
      SimpleThinningKernels(thinning_ratio, rng_kernel, properties_map), 1,
      properties_map);
  return std::dynamic_pointer_cast<TransformationStrategy>(r);
};
}; // namespace VANTAGE::Reactions
#endif
