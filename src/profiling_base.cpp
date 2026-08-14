
#include "../include/reactions_lib/profiling_base.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

std::string ProfilingBase::get_profiling_name() { return typeid(*this).name(); }

std::optional<NP::ProfileRegion>
ProfilingBase::start_profiling_region(NP::ParticleSubGroupSharedPtr &subgroup,
                                      const std::string key1) {
  auto &sycl_target = NP::get_particle_group(subgroup)->sycl_target;
  return sycl_target->profile_map.start_region(get_profiling_name(), key1,
                                               PROFILING_LEVEL);
}

void ProfilingBase::end_profiling_region(
    NP::ParticleSubGroupSharedPtr &subgroup,
    std::optional<NP::ProfileRegion> &region) {
  auto &sycl_target = get_particle_group(subgroup)->sycl_target;
  sycl_target->profile_map.end_region(region);
}

} // namespace VANTAGE::Reactions
