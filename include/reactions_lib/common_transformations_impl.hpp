#ifndef VANTAGE_REACTIONS_COMMON_TRANSFORMATIONS_IMPL_H
#define VANTAGE_REACTIONS_COMMON_TRANSFORMATIONS_IMPL_H

#include "common_transformations.hpp"

namespace VANTAGE::Reactions {

void SimpleRemovalTransformationStrategy::transform_v(
    ParticleSubGroupSharedPtr target_subgroup) {
  auto particle_group = target_subgroup->get_particle_group();

  particle_group->remove_particles(target_subgroup);
}

CompositeTransform::CompositeTransform(
    std::vector<std::shared_ptr<TransformationStrategy>> components)
    : components(components) {}

void CompositeTransform::transform_v(
    ParticleSubGroupSharedPtr target_subgroup) {
  for (auto &comp : this->components) {
    comp->transform(target_subgroup);
  }
}

void CompositeTransform::add_transformation(
    std::shared_ptr<TransformationStrategy> strat) {
  this->components.push_back(strat);
}

} // namespace VANTAGE::Reactions

#endif
