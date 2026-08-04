#ifndef REACTIONS_COLLISION_CELL_MANAGER_H
#define REACTIONS_COLLISION_CELL_MANAGER_H
#include "particle_properties_map.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Abstract base class for a collision cell hierarchy
 *
 */
struct AbstractCollCellHierarchy {

  /**
   * @brief Bin particles into collision cells
   *
   * @param target The subgroup containing particles to bin
   * @param coll_cell_sym The name of the collision cell sym
   */
  virtual void bin_particles(ParticleSubGroupSharedPtr target,
                             Sym<INT> coll_cell_sym) = 0;

  /**
   * @brief Get the number of collision cells per mesh cell
   */
  virtual std::vector<int> get_num_coll_cells() = 0;

  /**
   * @brief Get cell volumes per mesh and collision cell
   */
  virtual NDHostArraySharedPtr<REAL, 2> get_coll_cell_volumes() = 0;

  /**
   * @brief Set the linear resolution per mesh cell. The individual collision
   * cells per mesh cell should all have linear extents less than the specified
   * resolution.
   *
   * @param resolutions The resolution (largest allowed collision cell linear
   * extent) per mesh cell
   */
  virtual void
  set_coll_cell_linear_resolution(std::vector<REAL> resolutions) = 0;
};

/**
 * @brief Helper function for generating shared pointers of collision cell
 * hierarchies for passing to CollisionCellManager objects
 *
 * @tparam CollCellHierarchyDerived The class name of the derived class of
 * AbstractCollCellHierarchy
 * @param args Argument pack to be passed on to the derived class constructor
 */
template <typename CollCellHierarchyDerived, typename... ARGS>
inline std::shared_ptr<AbstractCollCellHierarchy>
make_coll_cell_hierarchy(ARGS &&...args) {
  auto r =
      std::make_shared<CollCellHierarchyDerived>(std::forward<ARGS>(args)...);
  return std::dynamic_pointer_cast<AbstractCollCellHierarchy>(r);
}

/**
 * @brief Manager class for collision cell binning and partition construction.
 *
 */
struct CollisionCellManager {

  CollisionCellManager() = delete;

  /**
   * @brief Constructor for CollisionCellManager
   *
   * @param sycl_target Compute device used by this manager. Should coincide
   * with the device used by the collision cell hierarchy
   * @param coll_cell_hierarchy Collision cell hierarchy defining particle
   * binning and collision cell volumes
   * @param species_ids Vector of integer ids for species managed by this
   * manager
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (here collision cell and species id
   * syms)
   */
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

  /**
   * @brief Get the current CollisionCellPartition object or construct it if
   * needed
   *
   * @param target Particle subgroup for which the partition is constructed
   */
  std::shared_ptr<DSMC::CollisionCellPartition>
  get_cell_partition(ParticleSubGroupSharedPtr target) {

    this->construct_cell_partition(target);
    return this->coll_cell_partition;
  };

  /**
   * @brief Get the number of particles per mesh and collision cell.
   *
   * @param target Particle subgroup for which to get the number of particles
   * @param species_id Species for which to get the number of particles
   */
  NDHostArraySharedPtr<int, 2>
  get_npart_coll_cell(ParticleSubGroupSharedPtr target, INT species_id) {
    this->construct_cell_partition(target);

    this->coll_cell_partition->get_num_unmasked_particles(species_id,
                                                          this->num_particles);
    return this->num_particles;
  };

  /**
   * @brief Bin particles in collision cells
   *
   * @param target Particle subgroup containing particles to be binned
   */
  void bin_particles(ParticleSubGroupSharedPtr target) {

    this->coll_cell_hierarchy->bin_particles(target, this->coll_cell_sym);
  }

  /**
   * @brief Get the number of collision cells per mesh cell
   *
   */
  std::vector<int> get_num_coll_cells() {

    return this->coll_cell_hierarchy->get_num_coll_cells();
  }

  /**
   * @brief Get cell volumes per mesh and collision cell
   */
  NDHostArraySharedPtr<REAL, 2> get_coll_cell_volumes() {

    return this->coll_cell_hierarchy->get_coll_cell_volumes();
  }

  /**
   * @brief Set the maximum linear extent for collision cells per mesh cell.
   *
   * @param resolutions Maximum linear extent for collision cells per mesh cell.
   */
  void set_coll_cell_linear_resolution(std::vector<REAL> resolutions) {

    this->coll_cell_hierarchy->set_coll_cell_linear_resolution(resolutions);
  }

  /**
   * @brief Invalidate the collision cell partition. Should be called whenever
   * particles are added or removed or the particle to collision cell map is
   * otherwise invalidated.
   */
  void invalidate_partition() { this->partition_valid = false; }

private:
  /**
   * @brief Construct the collision cell partition for a given group if the
   * current is not valid
   *
   * @param target Particle subgroup for which to construct the partition
   */
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
