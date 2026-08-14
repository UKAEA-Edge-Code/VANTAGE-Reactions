#include "../include/reactions_lib/common_transformations.hpp"

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

// Template instantiations
template class CellwiseAccumulator<REAL>;
template class CellwiseAccumulator<INT>;
template class WeightedCellwiseAccumulator<REAL>;
template class WeightedCellwiseAccumulator<INT>;

template class ParticleDatZeroer<REAL>;
template class ParticleDatZeroer<INT>;

template class CellwiseDistributor<REAL>;
template class CellwiseDistributor<INT>;

} // namespace VANTAGE::Reactions
