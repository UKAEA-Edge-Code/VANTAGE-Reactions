
#include "../include/reactions_lib/transformation_wrapper.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

NP::ParticleSubGroupSharedPtr MarkingStrategy::make_marker_subgroup_v(
    NP::ParticleSubGroupSharedPtr particle_group) {
  // This function should never actually be called. If it is called and we do
  // not have a return value then the calling function will receive an
  // undefined value. By setting a value we at least know what the returned
  // value is and can pick one that is detectable. By returning a nullptr the
  // calling code will hopefully segfault.
  return nullptr;
}

NP::ParticleSubGroupSharedPtr MarkingStrategy::make_marker_subgroup(
    NP::ParticleSubGroupSharedPtr particle_group) {
  auto r0 =
      this->start_profiling_region(particle_group, "make_marker_subgroup");
  auto sub_group = this->make_marker_subgroup_v(particle_group);
  this->end_profiling_region(particle_group, r0);
  return sub_group;
}

void TransformationStrategy::transform_v(
    NP::ParticleSubGroupSharedPtr target_subgroup) {}

void TransformationStrategy::transform(
    NP::ParticleSubGroupSharedPtr target_subgroup) {
  auto r0 = this->start_profiling_region(target_subgroup, "transform");
  this->transform_v(target_subgroup);
  this->end_profiling_region(target_subgroup, r0);
}

TransformationWrapper::TransformationWrapper(
    std::vector<std::shared_ptr<MarkingStrategy>> marking_strategy,
    std::shared_ptr<TransformationStrategy> transformation_strategy)
    : marking_strat(marking_strategy),
      transformation_strat(transformation_strategy) {}

TransformationWrapper::TransformationWrapper(
    std::shared_ptr<TransformationStrategy> transformation_strategy)
    : transformation_strat(transformation_strategy) {}

void TransformationWrapper::add_marking_strategy(
    std::shared_ptr<MarkingStrategy> marking_strategy) {
  this->marking_strat.push_back(marking_strategy);
}

} // namespace VANTAGE::Reactions
