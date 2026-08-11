
#include "../include/reactions_lib/common_transformations.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

void SimpleRemovalTransformationStrategy::transform_v(
    NP::ParticleSubGroupSharedPtr target_subgroup) {
  auto particle_group = target_subgroup->get_particle_group();

  particle_group->remove_particles(target_subgroup);
}

CompositeTransform::CompositeTransform(
    std::vector<std::shared_ptr<TransformationStrategy>> components)
    : components(components) {}

void CompositeTransform::transform_v(
    NP::ParticleSubGroupSharedPtr target_subgroup) {
  for (auto &comp : this->components) {
    comp->transform(target_subgroup);
  }
}

void CompositeTransform::add_transformation(
    std::shared_ptr<TransformationStrategy> strat) {
  this->components.push_back(strat);
}

// Template instantiations
template class CellwiseAccumulator<NP::REAL>;
template class CellwiseAccumulator<NP::INT>;
template class WeightedCellwiseAccumulator<NP::REAL>;
template class WeightedCellwiseAccumulator<NP::INT>;

template class ParticleDatZeroer<NP::REAL>;
template class ParticleDatZeroer<NP::INT>;

template class CellwiseDistributor<NP::REAL>;
template class CellwiseDistributor<NP::INT>;

} // namespace VANTAGE::Reactions
