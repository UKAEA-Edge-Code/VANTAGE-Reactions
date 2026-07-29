#ifndef VANTAGE_REACTIONS_REACTION_BASE_IMPL_H
#define VANTAGE_REACTIONS_REACTION_BASE_IMPL_H

#include "reaction_base.hpp"
#include "vantage_inline.hpp"

namespace VANTAGE::Reactions {

VANTAGE_REACTIONS_INLINE AbstractReaction::AbstractReaction(
    SYCLTargetSharedPtr sycl_target,
    const std::map<int, std::string> &properties_map)
    : sycl_target_stored(sycl_target),
      device_rate_buffer(
          std::make_shared<LocalArray<REAL>>(sycl_target, 0, 0.0)),
      pre_req_data(std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, 0, 0)),
      max_buffer_size(16384 *
                      get_env_size_t("REACTIONS_CELL_BLOCK_SIZE", 256)) {

  NESOWARN(map_subset_check(properties_map),
           "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsistencies with indexing of properties.");

  this->total_reaction_rate =
      Sym<REAL>(properties_map.at(default_properties.tot_reaction_rate));
  this->weight_sym = Sym<REAL>(properties_map.at(default_properties.weight));

  this->pre_req_data->fill(0.0);
}

VANTAGE_REACTIONS_INLINE void
AbstractReaction::calculate_rates(ParticleSubGroupSharedPtr particle_sub_group,
                                  INT cell_idx_start, INT cell_idx_end) {
  auto r0 = this->start_profiling_region(particle_sub_group, "calculate_rates");
  this->calculate_rates_v(particle_sub_group, cell_idx_start, cell_idx_end);
  this->end_profiling_region(particle_sub_group, r0);
}

VANTAGE_REACTIONS_INLINE void
AbstractReaction::apply(ParticleSubGroupSharedPtr particle_sub_group,
                        INT cell_idx_start, INT cell_idx_end, double dt,
                        ParticleGroupSharedPtr child_group, bool full_weight) {
  auto r0 = this->start_profiling_region(particle_sub_group, "apply");
  this->apply_v(particle_sub_group, cell_idx_start, cell_idx_end, dt,
                child_group, full_weight);
  this->end_profiling_region(particle_sub_group, r0);
}

} // namespace VANTAGE::Reactions

#endif
