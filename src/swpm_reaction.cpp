#include "../include/reactions_lib/swpm_reaction.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

void AbstractPairReaction::calculate_rates(
    NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList>
        &pair_list,
    INT cell_idx_start, INT cell_idx_end) {}

void AbstractPairReaction::apply(
    NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList>
        &pair_list,
    INT cell_idx_start, INT cell_idx_end, double dt,
    NP::ParticleGroupSharedPtr child_group) {}

void AbstractPairReaction::set_max_buffer_size(size_t max_size) {}

void AbstractPairReaction::set_max_num_coll_cells(size_t max_cells) {}

} // namespace VANTAGE::Reactions
