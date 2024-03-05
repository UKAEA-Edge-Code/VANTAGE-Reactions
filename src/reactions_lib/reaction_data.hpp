#pragma once
#include "particle_group.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

/**
 * @brief SYCL CRTP base reaction data object.
 *
 * @tparam ReactionDataDerived The typename of the derived class of
 * ReactionDataBase
 * @param dt Optional parameter (default value = 0.0)
 */

template <typename ReactionDataDerived>

struct ReactionDataBase {

  ReactionDataBase() : dt(0.0) {}

  ReactionDataBase(REAL dt_) : dt(dt_) {}

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
  REAL calc_rate(Access::LoopIndex::Read &index,
                 Access::SymVector::Read<REAL> &vars) const {
    const auto &underlying = static_cast<const ReactionDataDerived &>(*this);

    return underlying.template calc_rate(index, vars);
  }

  /**
   * @brief Getter function for dt.
   *
   * @return REAL (type-aliased to double) The value of dt.
   */
  const REAL &get_dt() const { return dt; }

private:
  const REAL dt;
};