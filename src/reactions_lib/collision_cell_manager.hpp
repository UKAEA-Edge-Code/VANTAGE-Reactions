#ifndef REACTIONS_COLLISION_CELL_MANAGER_H
#define REACTIONS_COLLISION_CELL_MANAGER_H
#include "particle_properties_map.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

struct AbstractCollCellHierarchy {

  virtual void bin_particles(ParticleSubGroupSharedPtr target,
                             Sym<INT> coll_cell_sym) = 0;

  virtual std::vector<int> get_num_coll_cells() = 0;

  virtual NDHostArraySharedPtr<REAL, 2> get_coll_cell_volumes() = 0;

  virtual void
  set_coll_cell_linear_resolution(std::vector<REAL> resolutions) = 0;
};

template <typename CollCellHierarchyDerived, typename... ARGS>
inline std::shared_ptr<AbstractCollCellHierarchy>
make_coll_cell_hierarchy(ARGS &&...args) {
  auto r =
      std::make_shared<CollCellHierarchyDerived>(std::forward<ARGS>(args)...);
  return std::dynamic_pointer_cast<AbstractCollCellHierarchy>(r);
}

struct CollisionCellManager {

  CollisionCellManager() = delete;

  CollisionCellManager(
      SYCLTargetSharedPtr sycl_target,
      std::shared_ptr<AbstractCollCellHierarchy> coll_cell_hierarchy,
      std::vector<INT> species_ids,
      const std::map<int, std::string> &properties_map = get_default_map())
      : coll_cell_hierarchy(coll_cell_hierarchy), species_ids(species_ids) {

    this->coll_cell_partition = std::make_shared<DSMC::CollisionCellPartition>(
        sycl_target, coll_cell_hierarchy->get_num_coll_cells().size(),
        species_ids);
    this->species_id_sym =
        Sym<INT>(properties_map.at(default_properties.internal_state));
    this->coll_cell_sym =
        Sym<INT>(properties_map.at(default_properties.collision_cell_id));
  };

  std::shared_ptr<DSMC::CollisionCellPartition>
  get_cell_partition(ParticleSubGroupSharedPtr target) {

    this->construct_cell_partition(target);
    return this->coll_cell_partition;
  };

  NDHostArraySharedPtr<int, 2>
  get_npart_coll_cell(ParticleSubGroupSharedPtr target, INT species_id) {
    this->construct_cell_partition(target);

    this->coll_cell_partition->get_num_unmasked_particles(species_id,
                                                          this->num_particles);
    return this->num_particles;
  };

  void bin_particles(ParticleSubGroupSharedPtr target) {

    this->coll_cell_hierarchy->bin_particles(target, this->coll_cell_sym);
  }

  std::vector<int> get_num_coll_cells() {

    return this->coll_cell_hierarchy->get_num_coll_cells();
  }

  NDHostArraySharedPtr<REAL, 2> get_coll_cell_volumes() {

    return this->coll_cell_hierarchy->get_coll_cell_volumes();
  }

  void set_coll_cell_linear_resolution(std::vector<REAL> resolutions) {

    this->coll_cell_hierarchy->set_coll_cell_linear_resolution(resolutions);
  }

  void invalidate_partition() { this->partition_valid = false; }

private:
  void construct_cell_partition(ParticleSubGroupSharedPtr target) {

    if (!this->partition_valid) {

      this->partition_valid = true;
      this->coll_cell_partition->construct(
          target, this->coll_cell_hierarchy->get_num_coll_cells(),
          this->species_id_sym, 0, this->coll_cell_sym, 0);
    }
  };

  bool partition_valid = false;

protected:
  std::shared_ptr<DSMC::CollisionCellPartition> coll_cell_partition;
  std::shared_ptr<AbstractCollCellHierarchy> coll_cell_hierarchy;
  Sym<INT> species_id_sym;
  Sym<INT> coll_cell_sym;
  std::vector<INT> species_ids;

  NDHostArraySharedPtr<int, 2> num_particles;
};
}; // namespace VANTAGE::Reactions
#endif
