#ifndef VANTAGE_REACTIONS_COMMON_TRANSFORMATIONS_IMPL_H
#define VANTAGE_REACTIONS_COMMON_TRANSFORMATIONS_IMPL_H

#include "common_transformations.hpp"
#include "vantage_inline.hpp"

namespace VANTAGE::Reactions {

VANTAGE_REACTIONS_INLINE void SimpleRemovalTransformationStrategy::transform_v(
    ParticleSubGroupSharedPtr target_subgroup) {
  auto particle_group = target_subgroup->get_particle_group();

  particle_group->remove_particles(target_subgroup);
}

VANTAGE_REACTIONS_INLINE CompositeTransform::CompositeTransform(
    std::vector<std::shared_ptr<TransformationStrategy>> components)
    : components(components) {}

VANTAGE_REACTIONS_INLINE void
CompositeTransform::transform_v(ParticleSubGroupSharedPtr target_subgroup) {
  for (auto &comp : this->components) {
    comp->transform(target_subgroup);
  }
}

VANTAGE_REACTIONS_INLINE void CompositeTransform::add_transformation(
    std::shared_ptr<TransformationStrategy> strat) {
  this->components.push_back(strat);
}

} // namespace VANTAGE::Reactions

#endif
