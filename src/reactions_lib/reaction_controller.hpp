#pragma once
#include "common_markers.hpp"
#include "common_transformations.hpp"
#include "particle_properties_map.hpp"
#include "reaction_base.hpp"
#include "transformation_wrapper.hpp"
#include <iostream>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/particle_group.hpp>
#include <neso_particles/particle_sub_group/particle_sub_group.hpp>
#include <neso_particles/particle_sub_group/particle_sub_group_utility.hpp>

using namespace NESO::Particles;

namespace Reactions {

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

  ReactionController(
      std::vector<std::shared_ptr<TransformationWrapper>> parent_transform,
      std::vector<std::shared_ptr<TransformationWrapper>> child_transform,
      bool auto_clean_tot_rate_buffer = true,
      const std::map<int, std::string> &properties_map = get_default_map())
      : parent_transform(parent_transform), child_transform(child_transform),
        id_sym(Sym<INT>(properties_map.at(default_properties.internal_state))),
        tot_rate_buffer(
            Sym<REAL>(properties_map.at(default_properties.tot_reaction_rate))),
        panic_flag(Sym<INT>(properties_map.at(default_properties.panic))),
        reacted_flag(
            Sym<INT>(properties_map.at(default_properties.reacted_flag))),
        auto_clean_tot_rate_buffer(auto_clean_tot_rate_buffer) {
    auto zeroer = make_transformation_strategy<ParticleDatZeroer<REAL>>(
        std::vector<std::string>{tot_rate_buffer.name});
    this->rate_buffer_zeroer = std::make_shared<TransformationWrapper>(
        std::dynamic_pointer_cast<TransformationStrategy>(zeroer));
    this->setup_particle_group_temporary();
    this->reacted_marker =
        make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
            this->reacted_flag, 1);
    auto rng_lambda = [&]() -> REAL { return 0; };
    this->rng_kernel =
        std::make_shared<HostPerParticleBlockRNG<REAL>>(rng_lambda, 0);
  }

  ReactionController(
      bool auto_clean_tot_rate_buffer = true,
      const std::map<int, std::string> &properties_map = get_default_map())
      : ReactionController(
            std::vector<std::shared_ptr<TransformationWrapper>>{},
            std::vector<std::shared_ptr<TransformationWrapper>>{},
            auto_clean_tot_rate_buffer, properties_map){};

  ReactionController(
      std::shared_ptr<TransformationWrapper> child_transform,
      bool auto_clean_tot_rate_buffer = true,
      const std::map<int, std::string> &properties_map = get_default_map())
      : ReactionController(
            std::vector<std::shared_ptr<TransformationWrapper>>{},
            std::vector{child_transform}, auto_clean_tot_rate_buffer,
            properties_map){};

  ReactionController(
      std::shared_ptr<TransformationWrapper> parent_transform,
      std::shared_ptr<TransformationWrapper> child_transform,
      bool auto_clean_tot_rate_buffer = true,
      const std::map<int, std::string> &properties_map = get_default_map())
      : ReactionController(std::vector{parent_transform},
                           std::vector{child_transform},
                           auto_clean_tot_rate_buffer, properties_map){};

  /**
   * @brief Function to populate the sub_group_selectors map and
   * parent_ids, child_ids sets, as well as set the buffer sizes used.
   */
  void controller_pre_process() {
    for (int r = 0; r < this->reactions.size(); r++) {
      this->reactions[r]->set_max_buffer_size(this->max_particles_per_cell *
                                              this->cell_block_size);
      if (!this->reactions[r]->get_in_states().empty()) {
        auto in_states = this->reactions[r]->get_in_states();

        for (int in_state : in_states) {
          this->parent_ids.insert(in_state);

          this->sub_group_selectors.emplace(std::make_pair(
              in_state,
              make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
                  this->id_sym, in_state)));
        }
      }

      if (!this->reactions[r]->get_out_states().empty()) {
        auto out_states = this->reactions[r]->get_out_states();

        for (int out_state : out_states) {
          this->child_ids.insert(out_state);

          this->sub_group_selectors.emplace(std::make_pair(
              out_state,
              make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
                  this->id_sym, out_state)));
        }
      }
    }
  }

public:
  /**
   * @brief Function to add reactions to a stored vector of AbstractReaction
   * pointers.
   *
   * @param reaction The reaction to be added
   */
  void add_reaction(std::shared_ptr<AbstractReaction> reaction) {
    this->reactions.push_back(reaction);
    this->controller_pre_process();
  }

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
   */
  template <typename PARENT>
  void apply_reactions(
      std::shared_ptr<PARENT> target, double dt,
      ControllerMode controller_mode = ControllerMode::standard_mode) {

    ParticleGroupSharedPtr particle_group;

    if constexpr (std::is_same<ParticleGroup, PARENT>::value) {
      particle_group = target;
    } else {

      particle_group = get_particle_group(target);
    }

    const size_t cell_count = particle_group->domain->mesh->get_cell_count();

    // Ensure that the total rate buffer is flushed before the reactions are
    // applied
    if (this->auto_clean_tot_rate_buffer) {
      this->rate_buffer_zeroer->transform(target);
    }

    NESOASSERT(particle_group->contains_dat(this->id_sym, 1),
               "ParticleGroup passed to controller does not contain expected "
               "ID dat, or the dat has wrong dimensionality");
    NESOASSERT(particle_group->contains_dat(this->tot_rate_buffer, 1),
               "ParticleGroup passed to controller does not contain expected "
               "total rate dat, or the dat has wrong dimensionality");
    NESOASSERT(particle_group->contains_dat(this->panic_flag, 1),
               "ParticleGroup passed to controller does not contain expected "
               "panic flag dat, or the dat has wrong dimensionality");

    NESOASSERT(particle_group->contains_dat(this->reacted_flag, 1),
               "ParticleGroup passed to controller does not contain expected "
               "reacted flag dat, or the dat has wrong dimensionality");
    NESOASSERT(this->reactions.size() > 0,
               "ReactionController.apply_reactions(...) cannot be called "
               "without adding at "
               "least one reaction to the ReactionController object (via "
               "ReactionController.add_reaction(...)).");

    bool use_full_weight = false;

    switch (controller_mode) {

    case ControllerMode::semi_dsmc_mode:
      use_full_weight = true;

      break;

    case ControllerMode::surface_mode:
      use_full_weight = true;

      break;

    default:
      break;
    }

    for (int r = 0; r < this->reactions.size(); r++) {
      if (!this->reactions[r]->get_in_states().empty()) {
        auto in_states = this->reactions[r]->get_in_states();

        for (int in_state : in_states) {
          this->species_groups.emplace(std::make_pair(
              in_state,
              this->sub_group_selectors[in_state]->make_marker_subgroup(
                  particle_sub_group(target))));
        }

        switch (controller_mode) {

        case ControllerMode::semi_dsmc_mode: {

          for (int in_state : in_states) {
            this->reacted_species_groups.emplace(std::make_pair(
                in_state, this->reacted_marker->make_marker_subgroup(
                              this->species_groups[in_state])));
          }
          break;
        }

        default: {
          for (int in_state : in_states) {
            this->reacted_species_groups.emplace(
                std::make_pair(in_state, this->species_groups[in_state]));
          }
          break;
        }
        }
      }
    }

    auto child_group = this->particle_group_temporary->get(particle_group);

    for (int i = 0; i < cell_count; i += this->cell_block_size) {

      for (int r = 0; r < this->reactions.size(); r++) {

        INT in_state = this->reactions[r]->get_in_states()[0];

        this->reactions[r]->run_rate_loop(
            this->species_groups[in_state], i,
            std::min(i + this->cell_block_size, cell_count));
      }

      switch (controller_mode) {

      case ControllerMode::semi_dsmc_mode: {

        // marking loop
        auto loop = particle_loop(
            "reacted_loop", target,
            [=](auto index, auto reacted_flag, auto total_reaction_rate,
                auto kernel) {
              reacted_flag.at(0) =
                  (1 - Kernel::exp(-total_reaction_rate.at(0) * dt)) >
                          kernel.at(index, 0)
                      ? 1
                      : 0;
            },
            Access::read(ParticleLoopIndex{}),
            Access::write(this->reacted_flag),
            Access::read(this->tot_rate_buffer),
            Access::read(this->rng_kernel));

        loop->execute(i, std::min(i + this->cell_block_size, cell_count));
        rate_buffer_zeroer->transform(
            target, i, std::min(i + this->cell_block_size, cell_count));

        for (int r = 0; r < this->reactions.size(); r++) {

          INT in_state = this->reactions[r]->get_in_states()[0];

          this->reactions[r]->run_rate_loop(
              this->reacted_species_groups[in_state], i,
              std::min(i + this->cell_block_size, cell_count));
        }

        break;
      }

      default: {
        break;
      }
      }

      for (int r = 0; r < reactions.size(); r++) {
        INT in_state = this->reactions[r]->get_in_states()[0];

        this->reactions[r]->descendant_product_loop(
            this->reacted_species_groups[in_state], i,
            std::min(i + this->cell_block_size, cell_count), dt, child_group,
            use_full_weight);
      }

      for (auto it = this->child_ids.begin(); it != this->child_ids.end();
           it++) {
        for (auto tr : this->child_transform) {
          auto transform_buffer = std::make_shared<TransformationWrapper>(*tr);
          transform_buffer->add_marking_strategy(
              this->sub_group_selectors[*it]);
          transform_buffer->transform(
              child_group, i, std::min(i + this->cell_block_size, cell_count));
        }
      }
    }

    for (auto it = this->parent_ids.begin(); it != this->parent_ids.end();
         it++) {
      for (auto tr : this->parent_transform) {
        auto transform_buffer = std::make_shared<TransformationWrapper>(*tr);
        transform_buffer->add_marking_strategy(this->sub_group_selectors[*it]);
        transform_buffer->transform(target);
      }
    }
    if (this->child_ids.size() > 0) {
      particle_group->add_particles_local(child_group);
    }
    this->particle_group_temporary->restore(particle_group, child_group);
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
  std::map<int, std::shared_ptr<MarkingStrategy>> sub_group_selectors;
  std::map<int, ParticleSubGroupSharedPtr> species_groups;
  std::map<int, ParticleSubGroupSharedPtr> reacted_species_groups;

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
  std::shared_ptr<TransformationWrapper> rate_buffer_zeroer;
  bool auto_clean_tot_rate_buffer;
  std::shared_ptr<HostPerParticleBlockRNG<REAL>> rng_kernel;
  size_t cell_block_size = 256;
  size_t max_particles_per_cell = 16384;
  std::shared_ptr<ParticleGroupTemporary> particle_group_temporary;

  inline void setup_particle_group_temporary() {
    this->particle_group_temporary = std::make_shared<ParticleGroupTemporary>();
  }
};
} // namespace Reactions
