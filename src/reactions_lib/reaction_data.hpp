#include "particle_group.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

template <typename ReactionDataDerived>

struct ReactionDataBase {

  ReactionDataBase() : dt(0.0) {}

  ReactionDataBase(
    REAL dt_
  ) : dt(dt_) {}

  REAL calc_rate(Access::LoopIndex::Read& index,Access::SymVector::Read<REAL>& vars) const {
    const auto& underlying = static_cast<const ReactionDataDerived&>(*this);

    return underlying.template calc_rate(index,vars);
  }

  const REAL& get_dt() const {
    return dt;
  }

  private:
    const REAL dt;
};