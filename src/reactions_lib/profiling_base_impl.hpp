#ifndef VANTAGE_REACTIONS_PROFILING_BASE_IMPL_H
#define VANTAGE_REACTIONS_PROFILING_BASE_IMPL_H

#include "profiling_base.hpp"
#include "vantage_inline.hpp"

namespace VANTAGE::Reactions {

VANTAGE_REACTIONS_INLINE std::string ProfilingBase::get_profiling_name() {
  return typeid(*this).name();
}

VANTAGE_REACTIONS_INLINE std::optional<NESO::Particles::ProfileRegion>
ProfilingBase::start_profiling_region(
    NESO::Particles::ParticleSubGroupSharedPtr &subgroup,
    const std::string key1) {
  auto &sycl_target =
      NESO::Particles::get_particle_group(subgroup)->sycl_target;
  return sycl_target->profile_map.start_region(get_profiling_name(), key1,
                                               PROFILING_LEVEL);
}

VANTAGE_REACTIONS_INLINE void ProfilingBase::end_profiling_region(
    NESO::Particles::ParticleSubGroupSharedPtr &subgroup,
    std::optional<NESO::Particles::ProfileRegion> &region) {
  auto &sycl_target = get_particle_group(subgroup)->sycl_target;
  sycl_target->profile_map.end_region(region);
}

} // namespace VANTAGE::Reactions

#endif
