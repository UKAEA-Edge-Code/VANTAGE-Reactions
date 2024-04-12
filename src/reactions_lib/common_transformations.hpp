#ifndef COMMON_TRANSFORMATIONS_H
#define COMMON_TRANSFORMATIONS_H
#include "containers/local_array.hpp"
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

/**
 * @brief Transformation Strategy containing multiple other transformations,
 * applied in order of addition
 */
struct CompositeTransform : TransformationStrategy {

  CompositeTransform() = default;

  CompositeTransform(
      std::vector<std::shared_ptr<TransformationStrategy>> components)
      : components(components) {}
  /**
   * @brief Apply all children of this transform in order of addition
   *
   * @param target_subgroup Particle subgroup to apply the transform to
   */
  void transform(ParticleSubGroupSharedPtr target_subgroup) {
    for (auto &comp : this->components) {
      comp->transform(target_subgroup);
    }
  }

  /**
   * @brief Add a transformation to the composite
   *
   * @param strat TransformationStrategy to be added (will be applied after
   * previously added strategies are added)
   */
  void add_transformation(std::shared_ptr<TransformationStrategy> strat) {
    this->components.push_back(strat);
  }

private:
  std::vector<std::shared_ptr<TransformationStrategy>> components;
};
template <typename T> struct ParticleDatZeroer : TransformationStrategy {

  ParticleDatZeroer() = delete;

  ParticleDatZeroer(std::vector<std::string> dat_names) {

    for (auto name : dat_names) {
      this->dats.push_back(Sym<T>(name));
    }
  }
  /**
   * @brief Zero all particle dats with names stored in the transform
   *
   * @param target_subgroup Particle subgroup to apply the transform to
   */
  void transform(ParticleSubGroupSharedPtr target_subgroup) {

    std::vector<INT> num_comps_vec;
    auto particle_group = target_subgroup->get_particle_group();
    for (auto &dat : dats) {
      auto particle_dat = particle_group->get_dat(dat);

      num_comps_vec.push_back(particle_dat->ncomp);
    }

    auto comp_nums = std::make_shared<LocalArray<INT>>(
        target_subgroup->get_particle_group()->sycl_target, num_comps_vec);

    auto k_len = size(this->dats);
    auto loop = particle_loop(
        "zeroer_loop", target_subgroup,
        [=]( auto vars, auto comp_nums) {
          for (auto i = 0; i < k_len; i++) {
            for (auto j = 0; j < comp_nums.at(i); j++) {
              vars.at(i, j) = 0;
            }
          }
        },
        // The ->get_particle_group() is temporary until sym_vector accepts
        // ParticleSubGroup as an argument
        Access::write(
            sym_vector<T>(target_subgroup->get_particle_group(), this->dats)),
        Access::read(comp_nums));

    loop->execute();
  }

private:
  std::vector<Sym<T>> dats;
};
} // namespace Reactions
#endif
