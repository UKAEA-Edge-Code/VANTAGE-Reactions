#ifndef REACTIONS_REACTION_CONTROLLER_H
#define REACTIONS_REACTION_CONTROLLER_H
#include "common_markers.hpp"
#include "common_transformations.hpp"
#include "particle_properties_map.hpp"
#include "reaction_base.hpp"
#include "transformation_wrapper.hpp"
#include <ios>
#include <iostream>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/particle_group.hpp>
#include <neso_particles/particle_sub_group/particle_sub_group_base.hpp>
#include <neso_particles/typedefs.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * @brief Enum class containing possible modes for the ReactionController
 */
enum class ControllerMode {

  standard_mode,  /**< Standard mode, where every reaction is applied on part of
                     the ingoing particle's weight, with some weight potentially
                     not participating in any reaction*/
  semi_dsmc_mode, /**< Semi-deterministic Direct Simulation Monte Carlo (DSMC)
                    method, where MC is used to get which particles go through a
                    reaction, and then all possible reactions are applied to
                    those particles, consuming them completely. */
  surface_mode    /**< Surface reaction mode, where every reaction is applied to
                    all particles in the passed subgroup, with 100% of the
                    weight of each particle participating */

};

/**
 * @brief A reaction controller that orchestrates the application of reactions
 * to a given ParticleGroup or ParticleSubGroup.
 *
 * @param parent_transform TransformationWrapper(s) informing how parent
 * particles are to be handled
 * @param child_transform TransformationWrapper(s) informing how descendant
 * products are to be handled
 * @param auto_clean_tot_rate_buffer Automatically flush the total rate buffer.
 * Defaults to true.
 * @param properties_map Optional remapping of default properties (panic flag,
 * internal_state, and total rate)
 */
struct ReactionController {

  /**
   * @brief Constructor for ReactionController.
   *
   * @param parent_transform Vector of TransformationWrappers informing how
   * parent particles are to be handled
   * @param child_transform Vector of TransformationWrappers informing how
   * descendant products are to be handled
   * @param auto_clean_tot_rate_buffer Automatically flush the total rate
   * buffer. Defaults to true.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (eg. panic flag, internal_state, and
   * total rate)
   */
  ReactionController(
      std::vector<std::shared_ptr<TransformationWrapper>> parent_transform,
      std::vector<std::shared_ptr<TransformationWrapper>> child_transform,
      bool auto_clean_tot_rate_buffer = true,
      const std::map<int, std::string> &properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for ReactionController with no parent and child
   * transformation strategies.
   *
   * @param auto_clean_tot_rate_buffer Automatically flush the total rate
   * buffer. Defaults to true.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (eg. panic flag, internal_state, and
   * total rate)
   */
  ReactionController(
      bool auto_clean_tot_rate_buffer = true,
      const std::map<int, std::string> &properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for ReactionController with no parent transformation
   * strategies.
   *
   * @param child_transform A TransformationWrapper informing how descendant
   * products are to be handled
   * @param auto_clean_tot_rate_buffer Automatically flush the total rate
   * buffer. Defaults to true.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (eg. panic flag, internal_state, and
   * total rate)
   */
  ReactionController(
      std::shared_ptr<TransformationWrapper> child_transform,
      bool auto_clean_tot_rate_buffer = true,
      const std::map<int, std::string> &properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for ReactionController.
   *
   * @param parent_transform A TransformationWrapper informing how parent
   * particles are to be handled
   * @param child_transform A TransformationWrapper informing how descendant
   * products are to be handled
   * @param auto_clean_tot_rate_buffer Automatically flush the total rate
   * buffer. Defaults to true.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (eg. panic flag, internal_state, and
   * total rate)
   */
  ReactionController(
      std::shared_ptr<TransformationWrapper> parent_transform,
      std::shared_ptr<TransformationWrapper> child_transform,
      bool auto_clean_tot_rate_buffer = true,
      const std::map<int, std::string> &properties_map = get_default_map());

  /**
   * @brief Function to populate the sub_group_selectors map and
   * parent_ids, child_ids sets, as well as set the buffer sizes used.
   */
  void controller_pre_process();

public:
  /**
   * @brief Function to add reactions to a stored vector of AbstractReaction
   * pointers.
   *
   * @param reaction The reaction to be added
   */
  void add_reaction(std::shared_ptr<AbstractReaction> reaction);

  /**
   * @brief Set the maximum number of particles per cell (used in determining
   * the buffer size for reaction data
   *
   * @param max_num_parts Maximum number of particles per cell
   */
  void set_max_particles_per_cell(size_t max_num_parts) {
    this->max_particles_per_cell = max_num_parts;
  }

  /**
   * @brief Set the number of cells per cell block, determines how many cells
   * each reaction runs its loops over at a time, and determines the maximum
   * reaction data buffer size together with the maximum number of particles per
   * cell (block size times maximum number of particles per cell)
   *
   * @param cell_block_size Number of cells to apply reactions to at a time (set
   * to a lower number in case of memory issues)
   */
  void set_cell_block_size(size_t cell_block_size) {
    this->cell_block_size = cell_block_size;
  }
  void set_auto_clean_tot_rate_buffer(const bool &auto_clean_setting) {
    this->auto_clean_tot_rate_buffer = auto_clean_setting;
  }
  const bool &get_auto_clean_tot_rate_buffer() {
    return this->auto_clean_tot_rate_buffer;
  }

  /**
   * @brief Apply parent transform on the target group or subgroup
   *
   * @param target The ParticleGroup or ParticleSubGroup to apply the transforms
   * to
   */
  template <typename PARENT>
  void apply_parent_transforms(std::shared_ptr<PARENT> target) {
    auto particle_group = get_particle_group(target);
    auto target_as_subgroup = particle_sub_group(target);
    this->apply_parent_transforms_impl(target_as_subgroup, particle_group);
  }

  /**
   * @brief Applies all reactions that have been added prior to calling this
   * function. The reactions are effectively applied at the same time and the
   * result should not depend on the ordering of the reactions. Any reaction
   * products are added to the designated group (can be different to the parent
   * group) and they are transformed according to the child_transform
   * transformation wrapper. Parents are transformed according to the
   * parent_transform transformation wrapper.
   *
   * @param target The ParticleGroup or ParticleSubGroup to apply the
   * reactions to.
   * @param dt The current time step size.
   * @param product_group The ParticleGroup into which to add the products,
   * should have the same spec as the parent.
   * @param controller_mode The mode to run the controller in. Either
   * standard_mode (default) or semi_dsmc_mode.
   */
  template <typename PARENT>
  void apply(std::shared_ptr<PARENT> target, double dt,
             ParticleGroupSharedPtr product_group,
             ControllerMode controller_mode = ControllerMode::standard_mode) {
    auto particle_group = get_particle_group(target);
    auto target_as_subgroup = particle_sub_group(target);
    const bool is_particle_group = std::is_same<ParticleGroup, PARENT>::value;
    this->apply_impl(target_as_subgroup, particle_group, dt, product_group,
                     controller_mode, is_particle_group);
  }

  /**
   * @brief Applies all reactions that have been added prior to calling this
   * function. The reactions are effectively applied at the same time and the
   * result should not depend on the ordering of the reactions. Any reaction
   * products are added and they are transformed according to the
   * child_transform transformation wrapper. Parents are transformed according
   * to the parent_transform transformation wrapper.
   *
   * @param target The ParticleGroup or ParticleSubGroup to apply the
   * reactions to.
   * @param dt The current time step size.
   * @param controller_mode The mode to run the controller in. Either
   * standard_mode (default) or semi_dsmc_mode.
   */
  template <typename PARENT>
  void apply(std::shared_ptr<PARENT> target, double dt,
             ControllerMode controller_mode = ControllerMode::standard_mode) {
    ParticleGroupSharedPtr particle_group = get_particle_group(target);
    this->apply(target, dt, particle_group, controller_mode);
  }

  void
  set_rng_kernel(std::shared_ptr<HostPerParticleBlockRNG<REAL>> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }
  std::shared_ptr<HostPerParticleBlockRNG<REAL>> get_rng_kernel() {
    NESOASSERT(this->rng_kernel != nullptr,
               "RNG kernel is nullptr, was set_rng_kernel called?");
    return this->rng_kernel;
  }

private:
  void apply_parent_transforms_impl(ParticleSubGroupSharedPtr target,
                                    ParticleGroupSharedPtr particle_group);
  void apply_impl(ParticleSubGroupSharedPtr target,
                  ParticleGroupSharedPtr particle_group, double dt,
                  ParticleGroupSharedPtr product_group,
                  ControllerMode controller_mode, bool is_particle_group);

  std::map<int, std::shared_ptr<MarkingStrategy>> sub_group_selectors;
  std::map<int, ParticleSubGroupSharedPtr> species_groups;
  std::map<int, ParticleSubGroupSharedPtr> reacted_species_groups;
  ParticleGroupSharedPtr reference_particle_group = nullptr;

  std::set<int> parent_ids;
  std::set<int> child_ids;

  std::vector<std::shared_ptr<AbstractReaction>> reactions;
  std::vector<std::shared_ptr<TransformationWrapper>> parent_transform;
  std::vector<std::shared_ptr<TransformationWrapper>> child_transform;

  std::shared_ptr<MarkingStrategy> reacted_marker;
  Sym<INT> id_sym;
  Sym<INT> panic_flag;
  Sym<INT> reacted_flag;
  Sym<REAL> tot_rate_buffer;
  Sym<REAL> weight_sym;
  std::shared_ptr<TransformationWrapper> rate_buffer_zeroer;
  bool auto_clean_tot_rate_buffer;
  std::shared_ptr<HostPerParticleBlockRNG<REAL>> rng_kernel;
  size_t cell_block_size = 256;
  size_t max_particles_per_cell = 16384;
  std::shared_ptr<ParticleGroupTemporary> particle_group_temporary;

  void setup_particle_group_temporary() {
    this->particle_group_temporary = std::make_shared<ParticleGroupTemporary>();
  }
};

} // namespace VANTAGE::Reactions
#endif
