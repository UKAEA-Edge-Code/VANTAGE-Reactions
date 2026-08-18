
#include "../include/reactions_lib/downsampling_kernels/simple_thinning_kernels.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

SimpleThinningKernels::SimpleThinningKernels(
    REAL thinning_ratio,
    std::shared_ptr<NP::HostPerParticleBlockRNG<REAL>> rng_kernel,
    std::map<int, std::string> properties_map)
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
      this->required_int_props.find_index(this->properties_map.at(props.panic));
}

std::shared_ptr<TransformationStrategy> make_simple_thinning_strategy(
    NP::ParticleGroupSharedPtr template_group, REAL thinning_ratio,
    std::shared_ptr<NP::HostPerParticleBlockRNG<REAL>> rng_kernel,
    const std::map<int, std::string> &properties_map) {

  auto r = std::make_shared<DownsamplingStrategy<SimpleThinningKernels>>(
      template_group,
      SimpleThinningKernels(thinning_ratio, rng_kernel, properties_map), 1,
      properties_map);
  return std::dynamic_pointer_cast<TransformationStrategy>(r);
}

} // namespace VANTAGE::Reactions
