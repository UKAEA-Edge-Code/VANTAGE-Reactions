#ifndef TRANSFORM_WRAPPER_H
#define TRANSFORM_WRAPPER_H

#include "particle_sub_group.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

namespace Reactions {

/**
 * @brief Abstract base class for marking strategies. All marking strategies
 produce a ParticleSubGroupSharedPtr from another ParticleSubGroupSharedPtr
 using some selection criterion.
 *
 */
struct MarkingStrategy {

  virtual ParticleSubGroupSharedPtr
  make_marker_subgroup(ParticleSubGroupSharedPtr particle_group){};
};

/**
 * @brief Helper function for generating shared pointers of marking strategies
 * by passing constructor arguments to individual derived classes
 *
 * @tparam MarkingStrategyDerived The class name of the derived class of
 * MarkingStrategy
 * @param args Argument pack to be passed to the constructor of
 * MarkingStrategyDerived
 * @return std::shared_ptr<MarkingStrategy>
 */
template <typename MarkingStrategyDerived, typename... ARGS>
inline std::shared_ptr<MarkingStrategy> make_marking_strategy(ARGS... args) {
  auto r = std::make_shared<MarkingStrategyDerived>(args...);
  return std::dynamic_pointer_cast<MarkingStrategy>(r);
}

/**
 * @brief Abstract base class for transformation strategies. All transformation
 * strategies take a ParticleSubGroupSharedPtr and perform an arbitrary
 * transformation on it.
 *
 */
struct TransformationStrategy {

  TransformationStrategy() = default;

  virtual void transform(ParticleSubGroupSharedPtr target_subgroup){};
};

/**
 * @brief Helper function for generating shared pointers of transformation
 * strategies by passing constructor arguments to individual derived classes
 *
 * @tparam TransformationStrategyDerived The class name of the derived class of
 * TransformationStrategy
 * @param args Argument pack to be passed to the constructor of
 * TransformationStrategyDerived
 * @return std::shared_ptr<TransformationStrategy>
 */
template <typename TransformationStrategyDerived, typename... ARGS>
inline std::shared_ptr<TransformationStrategy>
make_transformation_strategy(ARGS... args) {
  auto r = std::make_shared<TransformationStrategyDerived>(args...);
  return std::dynamic_pointer_cast<TransformationStrategy>(r);
}

/**
 * @brief Wrapper class containing a marking and a transformation strategy to be
 * applied to a ParticleGroup. Its responsibility is to apply the two strategies
 * in order to transform those particles in a ParticleGroup that satisfy some
 * condition.
 *
 */
struct TransformationWrapper {

  TransformationWrapper() = delete;

  TransformationWrapper(
      std::shared_ptr<TransformationStrategy> transformation_strategy)
      : transformation_strat(transformation_strategy) {}

  TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>> marking_strategy,
      std::shared_ptr<TransformationStrategy> transformation_strategy)
      : marking_strat(marking_strategy),
        transformation_strat(transformation_strategy) {}

  /**
   * @brief Applies the marking and transformation strategies to a given
   * ParticleGroup, transforming those particles that satisfy some condition.
   *
   * @param target_group ParticleGroup to transform
   */
  void transform(ParticleGroupSharedPtr target_group) {

    this->transform(target_group, -1);
  }

  /**
   * @brief Applies the marking and transfomation strategies to a given
   * ParticleGroup, transforming those particle that satisfy some condition in a
   * given cell.
   *
   * @param target_group ParticleGroup to transform
   * @param cell_id Local cell id to restrict the transformation to
   */
  void transform(ParticleGroupSharedPtr target_group, INT cell_id) {

    auto marker_subgroup = make_shared<ParticleSubGroup>(target_group);
    if (cell_id >= 0) {
      auto cell_num = target_group->domain->mesh->get_cell_count();
      NESOASSERT(
          cell_id < cell_num,
          "Transformation wrapper transform called with cell id out of range");
      // TODO: Update with efficient cell-wise subgroup construction once
      // available in NP
      auto marker_subgroup = std::make_shared<ParticleSubGroup>(
          target_group, [=](auto cell_idx) { return cell_idx[0] == cell_id; },
          Access::read(Sym<INT>("CELL_ID")));
    }
    for (auto &strat : this->marking_strat) {
      marker_subgroup = strat->make_marker_subgroup(marker_subgroup);
    }

    this->transformation_strat->transform(marker_subgroup);
  }
  /**
   * @brief Add marking strategy to transfomation wrapper, adding its condition
   * to the wrapper.
   *
   * @param marking_strategy Strategy to be added
   */
  void add_marking_strategy(std::shared_ptr<MarkingStrategy> marking_strategy) {
    this->marking_strat.push_back(marking_strategy);
  }

private:
  std::vector<std::shared_ptr<MarkingStrategy>> marking_strat;
  std::shared_ptr<TransformationStrategy> transformation_strat;
};

/**
 * @brief SYCL CRTP base marking strategy host type. Each derived type should be
 * paired with a device type derived from MarkingFunctionWrapperBase which
 * contains only device copyable types.
 *
 * @tparam MarkingStrategyDerived CRTP template argument
 */
template <typename MarkingStrategyDerived>
struct MarkingStrategyBase : MarkingStrategy {

  MarkingStrategyBase() = default;

  /**
   * @brief Construct a new Marking Strategy Base object
   *
   * @param required_dats_real_read Standard vector of Sym<REAL>s representing
   * those real-valued NESO-Particles ParticleDats to be passed to device type
   * for determining marking function return
   * @param required_dats_int_read Standard vector of Sym<INT>s representing
   * those integer-valued NESO-Particles ParticleDats to be passed to device
   * type for determining marking function return
   */
  MarkingStrategyBase(const std::vector<Sym<REAL>> required_dats_real_read,
                      const std::vector<Sym<INT>> required_dats_int_read)
      : required_particle_dats_real(required_dats_real_read),
        required_particle_dats_int(required_dats_int_read) {}

  ParticleSubGroupSharedPtr
  make_marker_subgroup(ParticleSubGroupSharedPtr particle_sub_group) override {

    const auto &underlying = static_cast<MarkingStrategyDerived &>(*this);
    auto device_type = underlying.get_device_data();

    auto marker_subgroup = std::make_shared<ParticleSubGroup>(
        particle_sub_group,
        [=](auto req_reals, auto req_ints) {
          return device_type.marking_condition(req_reals, req_ints);
        },
        Access::read(sym_vector<REAL>(particle_sub_group->get_particle_group(),
                                      this->required_particle_dats_real)),
        Access::read(sym_vector<INT>(particle_sub_group->get_particle_group(),
                                     this->required_particle_dats_int)));
    return marker_subgroup;
  }

private:
  std::vector<Sym<REAL>>
      required_particle_dats_real; //!< vector of symbols associated with
                                   //!< real-valued read-only ParticleDats
                                   //!< needed to determine which particles get
                                   //!< marked
  std::vector<Sym<INT>>
      required_particle_dats_int; //!< vector of symbols associated with
                                  //!< integer-valued read-only ParticleDats
                                  //!< needed to determine which particles get
                                  //!< marked
};

/**
 * @brief Device type associated with classes derived from MarkingStrategyBase.
 * Classes derived from this and other device types should never be
 * constructable outside of their corresponding host type.
 *
 * @tparam MarkingFunctionWrapperDerived CRTP template argument
 */
template <typename MarkingFunctionWrapperDerived>
struct MarkingFunctionWrapperBase {

  MarkingFunctionWrapperBase() = default;

  /**
   * @brief Marking condition applied particle-by-particle. To be overriden by
   * the derived function following SYCL CRTP.
   *
   * @param real_vars Read-only accessor to a list of real-valued ParticleDats.
   * Use real_vars.at(v_idx,c_idx) to access the c_idx-th component of v_idx-th
   * ParticleDat in the list
   * @param int_vars  Read-only accessor to a list of integer-valued
   * ParticleDats. Use int_vars.at(v_idx,c_idx) to access the c_idx-th component
   * of v_idx-th ParticleDat in the list
   * @return bool The return value of the marking_condition function on the
   * derived type.
   */
  bool marking_condition(Access::SymVector::Read<REAL> &real_vars,
                         Access::SymVector::Read<INT> &int_vars) const {
    const auto &underlying =
        static_cast<const MarkingFunctionWrapperDerived &>(*this);

    return underlying.template marking_condition(real_vars, int_vars);
  }
};
} // namespace Reactions
#endif
