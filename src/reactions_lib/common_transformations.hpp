#include <memory>
#include <neso_particles.hpp>
#include <vector>
#include <transformation_wrapper.hpp>

using namespace NESO::Particles;

/**
 * @brief Simple transformation strategy that will remove all particles in the passed ParticleSubGroup
 * 
 */
struct SimpleRemovalTransformationStrategy: TransformationStrategy {

      SimpleRemovalTransformationStrategy() = default;

      /**
       * @brief Remove all particle in given subgroup
       * 
       * @param target_subgroup ParticleSubgroup to remove
       */
      void transform(ParticleSubGroupSharedPtr target_subgroup)
      {
        auto particle_group = target_subgroup->get_particle_group();

        particle_group->remove_particles(target_subgroup);
      }
};
