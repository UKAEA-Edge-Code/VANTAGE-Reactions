#ifndef REACTIONS_COLLISION_CELL_MANAGER_H
#define REACTIONS_COLLISION_CELL_MANAGER_H
#include "particle_properties_map.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

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
  virtual void bin_particles(NP::ParticleSubGroupSharedPtr target,
                             NP::Sym<INT> coll_cell_sym) = 0;

  /**
   * @brief Get the number of collision cells per mesh cell
   */
  virtual std::vector<int> get_num_coll_cells() = 0;

  /**
   * @brief Get cell volumes per mesh and collision cell
   */
  virtual NP::NDHostArraySharedPtr<REAL, 2> get_coll_cell_volumes() = 0;

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

  /**
   * @brief Return the current shape, i.e. n_cell,max_num_coll_cells
   */
  virtual std::tuple<int, int> get_current_mesh_dims() = 0;
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
 * Also handles reaction rate reduction across multiple reactions.
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
   * used when remapping property names (here mesh, collision cell, and species
   * id syms)
   */
  CollisionCellManager(
      NP::SYCLTargetSharedPtr sycl_target,
      std::shared_ptr<AbstractCollCellHierarchy> coll_cell_hierarchy,
      std::vector<INT> species_ids,
      const std::map<int, std::string> &properties_map = get_default_map());

  /**
   * @brief Get the current CollisionCellPartition object or construct it if
   * needed
   *
   * @param target Particle subgroup for which the partition is constructed
   */
  std::shared_ptr<NP::DSMC::CollisionCellPartition> get_cell_partition();

  /**
   * @brief Get the number of particles per mesh and collision cell.
   *
   * @param target Particle subgroup for which to get the number of particles
   * @param species_id Species for which to get the number of particles
   */
  NP::NDHostArraySharedPtr<int, 2>
  get_npart_coll_cell(NP::ParticleSubGroupSharedPtr target, INT species_id);

  /**
   * @brief Bin particles in collision cells
   *
   * @param target Particle subgroup containing particles to be binned
   */
  void bin_particles(NP::ParticleSubGroupSharedPtr target);

  /**
   * @brief Get the number of collision cells per mesh cell
   *
   */
  std::vector<int> get_num_coll_cells();

  /**
   * @brief Get cell volumes per mesh and collision cell
   */
  NP::NDHostArraySharedPtr<REAL, 2> get_coll_cell_volumes();

  /**
   * @brief Set the maximum linear extent for collision cells per mesh cell.
   *
   * @param resolutions Maximum linear extent for collision cells per mesh cell.
   */
  void set_coll_cell_linear_resolution(std::vector<REAL> resolutions);

  /**
   * @brief Invalidate the collision cell partition. Should be called whenever
   * particles are added or removed or the particle to collision cell map is
   * otherwise invalidated.
   */
  void invalidate_partition();

  template <typename T>
  void coll_cellwise_max(NP::ParticleSubGroupSharedPtr target, NP::Sym<T> sym,
                         int component,
                         NP::NDLocalArraySharedPtr<T, 2> &buffer) {

    this->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                    buffer);
    buffer->fill(-std::numeric_limits<REAL>::max());

    int k_component = component;

    NP::particle_loop(
        "coll_cellwise_max", target,
        [=](auto reduction_sym, auto reduction_buffer, auto cell,
            auto coll_cell) {
          reduction_buffer.fetch_max(cell[0], coll_cell[0],
                                     reduction_sym[k_component]);
        },
        NP::Access::read(sym), NP::Access::max(buffer),
        NP::Access::read(this->cell_id_sym),
        NP::Access::read(this->coll_cell_sym))
        ->execute();
  };

  /**
   * @brief Return a new NDLocalArraySharedPtr conforming to the expected mesh
   * cell,collision cell size
   *
   * @param sycl_target Device to use when constructing
   */
  template <typename T>
  NP::NDLocalArraySharedPtr<T, 2>
  get_empty_coll_cellwise_data(NP::SYCLTargetSharedPtr sycl_target) {

    auto expected_shape = this->coll_cell_hierarchy->get_current_mesh_dims();
    return std::make_shared<NP::NDLocalArray<T, 2>>(
        sycl_target, std::get<0>(expected_shape), std::get<1>(expected_shape));
  }

  /**
   * @brief Resize and existing (if nullptr) or construct a new
   * NDLocalArraySharedPtr conforming to the expected mesh cell,collision cell
   * size
   *
   * @param sycl_target Device to use when constructing
   */
  template <typename T>
  void resize_coll_cellwise_data(NP::SYCLTargetSharedPtr sycl_target,
                                 NP::NDLocalArraySharedPtr<T, 2> &data) {
    if (data == nullptr) {
      data = this->get_empty_coll_cellwise_data<T>(sycl_target);
    }

    auto expected_shape = this->coll_cell_hierarchy->get_current_mesh_dims();
    auto shape = data->index.shape;
    if (shape[0] != std::get<0>(expected_shape) ||
        shape[1] != std::get<1>(expected_shape)) {

      data = this->get_empty_coll_cellwise_data<T>(sycl_target);
    }
  }

  /**
   * @brief Update the rate reduction object with a fixed value for a given
   * reaction index
   *
   * @param reaction_index Reaction index for which to update the rate reduction
   * buffer
   * @param default_value The default value to update with
   */
  void update_rate_reduction(int reaction_index, REAL default_value);

  /**
   * @brief Update the rate reduction object for a given reaction index by
   * applying a collision-cell-wise max based on the device rate buffer
   *
   * @param pair_list Pair list corresponding to the pairs used to calculate the
   * device rate buffer
   * @param cell_start Starting cell index for the update (for blockwise
   * updates)
   * @param cell_end Final cell index for the update (for blockwise updates)
   * @param device_rate_buffer LocalArraySharedPtr for the pairwise reaction
   * rate buffer for the given reactions
   * @param reaction_index Reaction index for which to update the rate reduction
   * buffer
   */
  void update_rate_reduction(
      NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList>
          &pair_list,
      int cell_start, int cell_end,
      NP::LocalArraySharedPtr<REAL> &device_rate_buffer, int reaction_index);

  /**
   * @brief Set up the rate reduction buffers
   *
   * @param n_reactions The number of reactions controlled by the reaction
   * controller using this collision cell manager
   */
  void setup_rate_reduction(int n_reactions);

  /**
   * @brief Perform a reaction-wise addition reduction on the max rate buffers
   * providing <sigma*v_r>_max per collision cell
   *
   * @param accumulated_rates Buffer into which to save the reduction result
   */
  void
  get_rate_reduction(NP::NDLocalArraySharedPtr<REAL, 2> &accumulated_rates);

  /**
   * @brief Construct the collision cell partition for a given group if the
   * current is not valid
   *
   * @param target Particle subgroup for which to construct the partition
   */
  void construct_cell_partition(NP::ParticleSubGroupSharedPtr target);

  std::vector<INT> get_species_ids();

private:
  std::shared_ptr<NP::DSMC::CollisionCellPartition> coll_cell_partition;
  std::shared_ptr<AbstractCollCellHierarchy> coll_cell_hierarchy;
  NP::Sym<INT> species_id_sym;
  NP::Sym<INT> coll_cell_sym;
  NP::Sym<INT> cell_id_sym;
  std::vector<INT> species_ids;
  std::vector<int> num_coll_cells;
  std::vector<int> cell_change_mask;

  std::shared_ptr<NP::DSMC::CollisionCellRateReduction> reduction_obj;

  bool partition_valid = false;
  NP::NDHostArraySharedPtr<int, 2> num_particles;
};

// Explicit instantiations of the CollisionCellManager member function templates
// above, shipped in the compiled library. T is only ever one of the two
// collision-cell scalar types: 'int' (particle counts: N_a/N_b, q_hat input,
// the pair counters in SWPMReactionController) and REAL (rates, volumes,
// weights). 'INT' (int64_t) is not used with these helpers.
extern template void CollisionCellManager::coll_cellwise_max<REAL>(
    NP::ParticleSubGroupSharedPtr, NP::Sym<REAL>, int,
    NP::NDLocalArraySharedPtr<REAL, 2> &);
extern template NP::NDLocalArraySharedPtr<int, 2>
    CollisionCellManager::get_empty_coll_cellwise_data<int>(
        NP::SYCLTargetSharedPtr);
extern template NP::NDLocalArraySharedPtr<REAL, 2>
    CollisionCellManager::get_empty_coll_cellwise_data<REAL>(
        NP::SYCLTargetSharedPtr);
extern template void CollisionCellManager::resize_coll_cellwise_data<int>(
    NP::SYCLTargetSharedPtr, NP::NDLocalArraySharedPtr<int, 2> &);
extern template void CollisionCellManager::resize_coll_cellwise_data<REAL>(
    NP::SYCLTargetSharedPtr, NP::NDLocalArraySharedPtr<REAL, 2> &);

}; // namespace VANTAGE::Reactions
#endif
