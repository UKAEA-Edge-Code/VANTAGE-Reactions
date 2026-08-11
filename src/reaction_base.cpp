
#include "../include/reactions_lib/reaction_base.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

AbstractReaction::AbstractReaction(
    NP::SYCLTargetSharedPtr sycl_target,
    const std::map<int, std::string> &properties_map)
    : sycl_target_stored(sycl_target),
      device_rate_buffer(
          std::make_shared<NP::LocalArray<NP::REAL>>(sycl_target, 0, 0.0)),
      pre_req_data(
          std::make_shared<NP::NDLocalArray<NP::REAL, 2>>(sycl_target, 0, 0)),
      max_buffer_size(16384 *
                      NP::get_env_size_t("REACTIONS_CELL_BLOCK_SIZE", 256)) {

  NESOWARN(map_subset_check(properties_map),
           "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsistencies with indexing of properties.");

  this->total_reaction_rate = NP::Sym<NP::REAL>(
      properties_map.at(default_properties.tot_reaction_rate));
  this->weight_sym =
      NP::Sym<NP::REAL>(properties_map.at(default_properties.weight));

  this->pre_req_data->fill(0.0);
}

void AbstractReaction::calculate_rates(
    NP::ParticleSubGroupSharedPtr particle_sub_group, NP::INT cell_idx_start,
    NP::INT cell_idx_end) {
  auto r0 = this->start_profiling_region(particle_sub_group, "calculate_rates");
  this->calculate_rates_v(particle_sub_group, cell_idx_start, cell_idx_end);
  this->end_profiling_region(particle_sub_group, r0);
}

void AbstractReaction::apply(NP::ParticleSubGroupSharedPtr particle_sub_group,
                             NP::INT cell_idx_start, NP::INT cell_idx_end,
                             double dt, NP::ParticleGroupSharedPtr child_group,
                             bool full_weight) {
  auto r0 = this->start_profiling_region(particle_sub_group, "apply");
  this->apply_v(particle_sub_group, cell_idx_start, cell_idx_end, dt,
                child_group, full_weight);
  this->end_profiling_region(particle_sub_group, r0);
}

} // namespace VANTAGE::Reactions
