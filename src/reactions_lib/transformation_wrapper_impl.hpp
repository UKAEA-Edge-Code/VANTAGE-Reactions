#ifndef VANTAGE_REACTIONS_TRANSFORM_WRAPPER_IMPL_H
#define VANTAGE_REACTIONS_TRANSFORM_WRAPPER_IMPL_H

#include "transformation_wrapper.hpp"
#include "vantage_inline.hpp"

namespace VANTAGE::Reactions {

VANTAGE_REACTIONS_INLINE ParticleSubGroupSharedPtr
MarkingStrategy::make_marker_subgroup_v(
    ParticleSubGroupSharedPtr particle_group) {
  // This function should never actually be called. If it is called and we do
  // not have a return value then the calling function will receive an
  // undefined value. By setting a value we at least know what the returned
  // value is and can pick one that is detectable. By returning a nullptr the
  // calling code will hopefully segfault.
  return nullptr;
}

VANTAGE_REACTIONS_INLINE ParticleSubGroupSharedPtr
MarkingStrategy::make_marker_subgroup(
    ParticleSubGroupSharedPtr particle_group) {
  auto r0 =
      this->start_profiling_region(particle_group, "make_marker_subgroup");
  auto sub_group = this->make_marker_subgroup_v(particle_group);
  this->end_profiling_region(particle_group, r0);
  return sub_group;
}

VANTAGE_REACTIONS_INLINE void
TransformationStrategy::transform_v(ParticleSubGroupSharedPtr target_subgroup) {
}

VANTAGE_REACTIONS_INLINE void
TransformationStrategy::transform(ParticleSubGroupSharedPtr target_subgroup) {
  auto r0 = this->start_profiling_region(target_subgroup, "transform");
  this->transform_v(target_subgroup);
  this->end_profiling_region(target_subgroup, r0);
}

VANTAGE_REACTIONS_INLINE TransformationWrapper::TransformationWrapper(
    std::vector<std::shared_ptr<MarkingStrategy>> marking_strategy,
    std::shared_ptr<TransformationStrategy> transformation_strategy)
    : marking_strat(marking_strategy),
      transformation_strat(transformation_strategy) {}

VANTAGE_REACTIONS_INLINE TransformationWrapper::TransformationWrapper(
    std::shared_ptr<TransformationStrategy> transformation_strategy)
    : transformation_strat(transformation_strategy) {}

VANTAGE_REACTIONS_INLINE void TransformationWrapper::add_marking_strategy(
    std::shared_ptr<MarkingStrategy> marking_strategy) {
  this->marking_strat.push_back(marking_strategy);
}

} // namespace VANTAGE::Reactions

#endif
