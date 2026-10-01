#include "../include/reactions_lib/collision_cell_manager.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

CollisionCellManager::CollisionCellManager(
    NP::SYCLTargetSharedPtr sycl_target,
    std::shared_ptr<AbstractCollCellHierarchy> coll_cell_hierarchy,
    std::vector<INT> species_ids,
    const std::map<int, std::string> &properties_map)
    : coll_cell_hierarchy(coll_cell_hierarchy), species_ids(species_ids) {

  this->coll_cell_partition =
      std::make_shared<NP::DSMC::CollisionCellPartition>(
          sycl_target, coll_cell_hierarchy->get_num_coll_cells().size(),
          species_ids);

  this->reduction_obj = std::make_shared<NP::DSMC::CollisionCellRateReduction>(
      this->coll_cell_partition);
  this->reduction_obj->setup(0);

  this->num_coll_cells = this->coll_cell_hierarchy->get_num_coll_cells();
  // This makes sure that the first construct call marks all cells as having
  // been resized for reduction purposes
  std::fill(this->num_coll_cells.begin(), this->num_coll_cells.end(), 0);
  this->cell_change_mask = std::vector<int>(this->num_coll_cells.size(), 1);
  this->species_id_sym =
      NP::Sym<INT>(properties_map.at(default_properties.internal_state));
  this->coll_cell_sym =
      NP::Sym<INT>(properties_map.at(default_properties.collision_cell_id));
  this->cell_id_sym =
      NP::Sym<INT>(properties_map.at(default_properties.cell_id));
};

std::shared_ptr<NP::DSMC::CollisionCellPartition>
CollisionCellManager::get_cell_partition() {

  return this->coll_cell_partition;
};

NP::NDHostArraySharedPtr<int, 2>
CollisionCellManager::get_npart_coll_cell(NP::ParticleSubGroupSharedPtr target,
                                          INT species_id) {

  this->coll_cell_partition->get_num_unmasked_particles(species_id,
                                                        this->num_particles);
  return this->num_particles;
};

void CollisionCellManager::bin_particles(NP::ParticleSubGroupSharedPtr target) {

  if (!this->partition_valid) {
    this->coll_cell_hierarchy->bin_particles(target, this->coll_cell_sym);
  }
}

std::vector<int> CollisionCellManager::get_num_coll_cells() {
  return this->num_coll_cells;
}

NP::NDHostArraySharedPtr<REAL, 2>
CollisionCellManager::get_coll_cell_volumes() {

  return this->coll_cell_hierarchy->get_coll_cell_volumes();
}

void CollisionCellManager::set_coll_cell_linear_resolution(
    std::vector<REAL> resolutions) {

  this->coll_cell_hierarchy->set_coll_cell_linear_resolution(resolutions);
}

void CollisionCellManager::invalidate_partition() {
  this->partition_valid = false;
}

void CollisionCellManager::update_rate_reduction(int reaction_index,
                                                 REAL default_value) {
  this->reduction_obj->update(reaction_index, this->cell_change_mask,
                              default_value);
};

void CollisionCellManager::update_rate_reduction(
    NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList>
        &pair_list,
    int cell_start, int cell_end,
    NP::LocalArraySharedPtr<REAL> &device_rate_buffer, int reaction_index) {
  this->reduction_obj->update(reaction_index, pair_list, cell_start, cell_end,
                              this->coll_cell_sym, 0, device_rate_buffer);
};

void CollisionCellManager::setup_rate_reduction(int n_reactions) {
  this->reduction_obj->setup(n_reactions);
};

void CollisionCellManager::get_rate_reduction(
    NP::NDLocalArraySharedPtr<REAL, 2> &accumulated_rates) {
  this->reduction_obj->get(accumulated_rates);
}

void CollisionCellManager::construct_cell_partition(
    NP::ParticleSubGroupSharedPtr target) {
  auto new_num_coll_cells = this->coll_cell_hierarchy->get_num_coll_cells();

  std::transform(new_num_coll_cells.begin(), new_num_coll_cells.end(),
                 this->num_coll_cells.begin(), this->cell_change_mask.begin(),
                 [](int x, int y) { return x != y; });

  if (std::any_of(this->cell_change_mask.begin(), this->cell_change_mask.end(),
                  [](int x) { return x > 0; })) {
    this->partition_valid = false;
  }
  this->num_coll_cells = new_num_coll_cells;
  if (!this->partition_valid) {

    this->partition_valid = true;
    this->coll_cell_partition->construct(target, this->num_coll_cells,
                                         this->species_id_sym, 0,
                                         this->coll_cell_sym, 0);
    this->reduction_obj->resize();
  }
};

std::vector<INT> CollisionCellManager::get_species_ids() {
  return this->species_ids;
}

// Explicit instantiations of the CollisionCellManager member function
// templates. These are the only T used by the SWPM/pair path; see the matching
// extern template declarations in collision_cell_manager.hpp.
template void CollisionCellManager::coll_cellwise_max<REAL>(
    NP::ParticleSubGroupSharedPtr, NP::Sym<REAL>, int,
    NP::NDLocalArraySharedPtr<REAL, 2> &);
template NP::NDLocalArraySharedPtr<int, 2>
    CollisionCellManager::get_empty_coll_cellwise_data<int>(
        NP::SYCLTargetSharedPtr);
template NP::NDLocalArraySharedPtr<REAL, 2>
    CollisionCellManager::get_empty_coll_cellwise_data<REAL>(
        NP::SYCLTargetSharedPtr);
template void CollisionCellManager::resize_coll_cellwise_data<int>(
    NP::SYCLTargetSharedPtr, NP::NDLocalArraySharedPtr<int, 2> &);
template void CollisionCellManager::resize_coll_cellwise_data<REAL>(
    NP::SYCLTargetSharedPtr, NP::NDLocalArraySharedPtr<REAL, 2> &);

}; // namespace VANTAGE::Reactions
