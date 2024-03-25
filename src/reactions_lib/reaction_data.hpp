#pragma once
#include "containers/sym_vector.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;

/**
 * @brief SYCL CRTP base reaction data object.
 *
 * @tparam ReactionDataDerived The typename of the derived class of
 * ReactionDataBase
 */

template <typename ReactionDataDerived>

struct ReactionDataBase {

  ReactionDataBase() = default;

  /**
   * @brief Function to calculate the rates of the reaction that the
   * ReactionDataDerived-type object belongs to. To be overridden by the
   * function in the ReactionDataDerived-type object following SYCL CRTP.

   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param vars Read-only accessor to a list of real-valued ParticleDats.
   * Use real_vars.at(v_idx,c_idx) to access the c_idx-th component of v_idx-th
   * ParticleDat in the list
   * @return REAL (type-aliased to double) The calculated reaction rate from
   * the overriding function on the derived type.
   */
  // TODO: Extend the interface to take in integer syms?
  REAL calc_rate(Access::LoopIndex::Read &index,
                 Access::SymVector::Read<INT> &vars) const {
    const auto &underlying = static_cast<const ReactionDataDerived &>(*this);

    return underlying.template calc_rate(index, vars);
  }

  /**
   * @brief Function to calculate the rates of the reaction that the
   * ReactionDataDerived-type object belongs to. To be overridden by the
   * function in the ReactionDataDerived-type object following SYCL CRTP.

   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param vars Read-only accessor to a list of int-valued ParticleDats.
   * Use int_vars.at(v_idx,c_idx) to access the c_idx-th component of v_idx-th
   * ParticleDat in the list
   * @return REAL (type-aliased to double) The calculated reaction rate from
   * the overriding function on the derived type.
   */
  REAL calc_rate(Access::LoopIndex::Read &index,
                 Access::SymVector::Read<REAL> &vars) const {
    const auto &underlying = static_cast<const ReactionDataDerived &>(*this);

    return underlying.template calc_rate(index, vars);
  }

  /**
   * @brief Function to calculate the rates of the reaction that the
   * ReactionDataDerived-type object belongs to. To be overridden by the
   * function in the ReactionDataDerived-type object following SYCL CRTP.

   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param int_vars Read-only accessor to a list of int-valued ParticleDats.
   * Use int_vars.at(v_idx,c_idx) to access the c_idx-th component of v_idx-th
   * ParticleDat in the list
   * @param real_vars Read-only accessor to a list of real-valued ParticleDats.
   * Use real_vars.at(v_idx,c_idx) to access the c_idx-th component of v_idx-th
   * ParticleDat in the list
   * @return REAL (type-aliased to double) The calculated reaction rate from
   * the overriding function on the derived type.
   */
  REAL calc_rate(Access::LoopIndex::Read &index,
                 Access::SymVector::Read<INT> &int_vars,
                 Access::SymVector::Read<REAL> &real_vars) const {
    const auto &underlying = static_cast<const ReactionDataDerived &>(*this);

    return underlying.template calc_rate(index, int_vars, real_vars);
  }
};
