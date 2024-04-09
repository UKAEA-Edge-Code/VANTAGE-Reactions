#ifndef COMMON_TRANSFORMATIONS_H
#define COMMON_TRANSFORMATIONS_H
#include <neso_particles.hpp>
#include <transformation_wrapper.hpp>

using namespace NESO::Particles;

namespace Reactions {
/**
 * @brief No operations transformation strategy
 */
struct NoOpTransformationStrategy : TransformationStrategy {
  NoOpTransformationStrategy() = default;
};
/**
 * @brief Simple transformation strategy that will remove all particles in the
 * passed ParticleSubGroup
 *
 */
struct SimpleRemovalTransformationStrategy : TransformationStrategy {

  SimpleRemovalTransformationStrategy() = default;

  /**
   * @brief Remove all particle in given subgroup
   *
   * @param target_subgroup ParticleSubgroup to remove
   */
  void transform(ParticleSubGroupSharedPtr target_subgroup) {
    auto particle_group = target_subgroup->get_particle_group();

    particle_group->remove_particles(target_subgroup);
  }
};
} // namespace Reactions
#endif
