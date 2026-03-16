#ifndef REACTIONS_SIMPLE_THINNING_H
#define REACTIONS_SIMPLE_THINNING_H

#include "reactions_lib/downsampling_base.hpp"
#include <cmath>
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

struct SimpleThinningOnDevice
    : DownsamplingKernelOnDeviceBase<0, HostPerParticleBlockRNG<REAL>> {

  SimpleThinningOnDevice() = default;

  SimpleThinningOnDevice(REAL thinning_ratio)
      : inverse_thinning_ratio(1 / thinning_ratio),
        thinning_ratio(thinning_ratio) {};

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

struct SimpleThinningKernels
    : DownsamplingKernelBase<DownsamplingMode::thinning,
                             ReductionKernelOnDeviceBase<0, 0, 0>,
                             SimpleThinningOnDevice> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};

  constexpr static std::array<int, 1> required_simple_int_props = {props.panic};

  SimpleThinningKernels(
      REAL thinning_ratio,
      std::shared_ptr<HostPerParticleBlockRNG<REAL>> rng_kernel,
      std::map<int, std::string> properties_map = get_default_map())
      : DownsamplingKernelBase<DownsamplingMode::thinning,
                               ReductionKernelOnDeviceBase<0, 0, 0>,
                               SimpleThinningOnDevice>(
            Properties<INT>(required_simple_int_props),
            Properties<REAL>(required_simple_real_props), properties_map) {

    this->set_rng_kernel(rng_kernel);
    this->downsampling_on_device_obj = SimpleThinningOnDevice(thinning_ratio);
    this->reduction_on_device_obj = ReductionKernelOnDeviceBase<0, 0, 0>();

    this->downsampling_on_device_obj->weight_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.weight));

    this->downsampling_on_device_obj->panic_ind =
        this->required_int_props.find_index(
            this->properties_map.at(props.panic));
  }
};

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
