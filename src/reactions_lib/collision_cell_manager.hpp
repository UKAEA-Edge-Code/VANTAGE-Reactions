#ifndef REACTIONS_COLLISION_CELL_MANAGER_H
#define REACTIONS_COLLISION_CELL_MANAGER_H
#include "particle_properties_map.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <neso_particles.hpp>
#include <tuple>
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
      SYCLTargetSharedPtr sycl_target,
      std::shared_ptr<AbstractCollCellHierarchy> coll_cell_hierarchy,
      std::vector<INT> species_ids,
      const std::map<int, std::string> &properties_map = get_default_map())
      : coll_cell_hierarchy(coll_cell_hierarchy), species_ids(species_ids) {

    this->coll_cell_partition = std::make_shared<DSMC::CollisionCellPartition>(
        sycl_target, coll_cell_hierarchy->get_num_coll_cells().size(),
        species_ids);

    this->reduction_obj = std::make_shared<DSMC::CollisionCellRateReduction>(
        this->coll_cell_partition);
    this->reduction_obj->setup(0);

    this->num_coll_cells = this->coll_cell_hierarchy->get_num_coll_cells();
    // This makes sure that the first construct call marks all cells as having
    // been resized for reduction purposes
    std::fill(this->num_coll_cells.begin(), this->num_coll_cells.end(), 0);
    this->cell_change_mask = std::vector<int>(this->num_coll_cells.size(), 1);
    this->species_id_sym =
        Sym<INT>(properties_map.at(default_properties.internal_state));
    this->coll_cell_sym =
        Sym<INT>(properties_map.at(default_properties.collision_cell_id));
    this->cell_id_sym = Sym<INT>(properties_map.at(default_properties.cell_id));
  };

  /**
   * @brief Get the current CollisionCellPartition object or construct it if
   * needed
   *
   * @param target Particle subgroup for which the partition is constructed
   */
  std::shared_ptr<DSMC::CollisionCellPartition> get_cell_partition() {

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

    if (!this->partition_valid) {
      this->coll_cell_hierarchy->bin_particles(target, this->coll_cell_sym);
    }
  }

  /**
   * @brief Get the number of collision cells per mesh cell
   *
   */
  std::vector<int> get_num_coll_cells() { return this->num_coll_cells; }

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

  template <typename T>
  void coll_cellwise_max(ParticleSubGroupSharedPtr target, Sym<T> sym,
                         int component, NDLocalArraySharedPtr<T, 2> &buffer) {

    this->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                    buffer);
    buffer->fill(-std::numeric_limits<REAL>::max());

    int k_component = component;

    particle_loop(
        "coll_cellwise_max", target,
        [=](auto reduction_sym, auto reduction_buffer, auto cell,
            auto coll_cell) {
          reduction_buffer.fetch_max(cell[0], coll_cell[0],
                                     reduction_sym[k_component]);
        },
        Access::read(sym), Access::max(buffer), Access::read(this->cell_id_sym),
        Access::read(this->coll_cell_sym))
        ->execute();
  };

  /**
   * @brief Return a new NDLocalArraySharedPtr conforming to the expected mesh
   * cell,collision cell size
   *
   * @param sycl_target Device to use when constructing
   */
  template <typename T>
  NDLocalArraySharedPtr<T, 2>
  get_empty_coll_cellwise_data(SYCLTargetSharedPtr sycl_target) {

    auto expected_shape = this->coll_cell_hierarchy->get_current_mesh_dims();
    return std::make_shared<NDLocalArray<T, 2>>(
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
  void resize_coll_cellwise_data(SYCLTargetSharedPtr sycl_target,
                                 NDLocalArraySharedPtr<T, 2> &data) {
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
  void update_rate_reduction(int reaction_index, REAL default_value) {
    this->reduction_obj->update(reaction_index, this->cell_change_mask,
                                default_value);
  };

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
      CellwisePairListAbsolute<ParticleGroup, CellwisePairList> &pair_list,
      int cell_start, int cell_end,
      LocalArraySharedPtr<REAL> &device_rate_buffer, int reaction_index) {
    this->reduction_obj->update(reaction_index, pair_list, cell_start, cell_end,
                                this->coll_cell_sym, 0, device_rate_buffer);
  };

  /**
   * @brief Set up the rate reduction buffers
   *
   * @param n_reactions The number of reactions controlled by the reaction
   * controller using this collision cell manager
   */
  void setup_rate_reduction(int n_reactions) {
    this->reduction_obj->setup(n_reactions);
  };

  /**
   * @brief Perform a reaction-wise addition reduction on the max rate buffers
   * providing <sigma*v_r>_max per collision cell
   *
   * @param accumulated_rates Buffer into which to save the reduction result
   */
  void get_rate_reduction(NDLocalArraySharedPtr<REAL, 2> &accumulated_rates) {
    this->reduction_obj->get(accumulated_rates);
  }

  /**
   * @brief Construct the collision cell partition for a given group if the
   * current is not valid
   *
   * @param target Particle subgroup for which to construct the partition
   */
  void construct_cell_partition(ParticleSubGroupSharedPtr target) {
    auto new_num_coll_cells = this->coll_cell_hierarchy->get_num_coll_cells();

    std::transform(new_num_coll_cells.begin(), new_num_coll_cells.end(),
                   this->num_coll_cells.begin(), this->cell_change_mask.begin(),
                   [](int x, int y) { return x != y; });

    if (std::any_of(this->cell_change_mask.begin(),
                    this->cell_change_mask.end(),
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

  std::vector<INT> get_species_ids() { return this->species_ids; }

private:
  std::shared_ptr<DSMC::CollisionCellPartition> coll_cell_partition;
  std::shared_ptr<AbstractCollCellHierarchy> coll_cell_hierarchy;
  Sym<INT> species_id_sym;
  Sym<INT> coll_cell_sym;
  Sym<INT> cell_id_sym;
  std::vector<INT> species_ids;
  std::vector<int> num_coll_cells;
  std::vector<int> cell_change_mask;

  std::shared_ptr<DSMC::CollisionCellRateReduction> reduction_obj;

  bool partition_valid = false;
  NDHostArraySharedPtr<int, 2> num_particles;
};
}; // namespace VANTAGE::Reactions
#endif
