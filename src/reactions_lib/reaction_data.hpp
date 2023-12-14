#include "particle_group.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

template <typename ReactionDataDerived>

struct ReactionDataBase {

  ReactionDataBase() = default;

  REAL calc_rate(Access::LoopIndex::Read& index,Access::SymVector::Read<REAL>& vars) const {
    const auto& underlying = static_cast<const ReactionDataDerived&>(*this);

    return underlying.template calc_rate(index,vars);
  }
};