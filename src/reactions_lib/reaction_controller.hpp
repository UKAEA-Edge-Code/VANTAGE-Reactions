#pragma once
#include "transformation_wrapper.hpp"
#include <memory>
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace Reactions {

/**
 * @brief A reaction controller that orchestrates the application of reactions
 * to a given ParticleGroup.
 *
 * @param parent_transform TransformationWrapper(s) informing how parent
 * particles are to be handled
 * @param child_transform TransformationWrapper(s) informing how descendant
 * products are to be handled
 * @param id_sym Symbol index of the integer ParticleDat that will be used
 * to specify which particles reactions will be applied to
 * @param tot_rate_buffer Symbol of the total reaction rate ParticleDat that
 * will be automatically flushed
 * @param auto_clean_tot_rate_buffer Automatically flush the total rate buffer.
 * Defaults to true.
 */
struct ReactionController {

  ReactionController(Sym<INT> id_sym, Sym<REAL> tot_rate_buffer,
                     bool auto_clean_tot_rate_buffer = true)
      : id_sym(id_sym), tot_rate_buffer(tot_rate_buffer),
        auto_clean_tot_rate_buffer(auto_clean_tot_rate_buffer) {
    auto zeroer = make_transformation_strategy<ParticleDatZeroer<REAL>>(
        std::vector<std::string>{tot_rate_buffer.name});
    this->rate_buffer_zeroer = std::make_shared<TransformationWrapper>(
        std::dynamic_pointer_cast<TransformationStrategy>(zeroer));
    this->setup_particle_group_temporary();
  }

  ReactionController(std::shared_ptr<TransformationWrapper> child_transform,
                     Sym<INT> id_sym, Sym<REAL> tot_rate_buffer,
                     bool auto_clean_tot_rate_buffer = true)
      : child_transform(std::vector{child_transform}), id_sym(id_sym),
        tot_rate_buffer(tot_rate_buffer),
        auto_clean_tot_rate_buffer(auto_clean_tot_rate_buffer) {
    auto zeroer = make_transformation_strategy<ParticleDatZeroer<REAL>>(
        std::vector<std::string>{tot_rate_buffer.name});
    this->rate_buffer_zeroer = std::make_shared<TransformationWrapper>(
        std::dynamic_pointer_cast<TransformationStrategy>(zeroer));
    this->setup_particle_group_temporary();
  }

  ReactionController(std::shared_ptr<TransformationWrapper> parent_transform,
                     std::shared_ptr<TransformationWrapper> child_transform,
                     Sym<INT> id_sym, Sym<REAL> tot_rate_buffer,
                     bool auto_clean_tot_rate_buffer = true)
      : parent_transform(std::vector{parent_transform}),
        child_transform(std::vector{child_transform}), id_sym(id_sym),
        tot_rate_buffer(tot_rate_buffer),
        auto_clean_tot_rate_buffer(auto_clean_tot_rate_buffer) {
    auto zeroer = make_transformation_strategy<ParticleDatZeroer<REAL>>(
        std::vector<std::string>{tot_rate_buffer.name});
    this->rate_buffer_zeroer = std::make_shared<TransformationWrapper>(
        std::dynamic_pointer_cast<TransformationStrategy>(zeroer));
    this->setup_particle_group_temporary();
  }

  ReactionController(
      std::vector<std::shared_ptr<TransformationWrapper>> parent_transform,
      std::vector<std::shared_ptr<TransformationWrapper>> child_transform,
      Sym<INT> id_sym, Sym<REAL> tot_rate_buffer,
      bool auto_clean_tot_rate_buffer = true)
      : parent_transform(parent_transform), child_transform(child_transform),
        id_sym(id_sym), tot_rate_buffer(tot_rate_buffer),
        auto_clean_tot_rate_buffer(auto_clean_tot_rate_buffer) {
    auto zeroer = make_transformation_strategy<ParticleDatZeroer<REAL>>(
        std::vector<std::string>{tot_rate_buffer.name});
    this->rate_buffer_zeroer = std::make_shared<TransformationWrapper>(
        std::dynamic_pointer_cast<TransformationStrategy>(zeroer));
    this->setup_particle_group_temporary();
  }

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
   * @param particle_group The ParticleGroup to apply the reactions to.
   * @param dt The current time step size.
   */
  void apply_reactions(ParticleGroupSharedPtr particle_group, double dt) {
    const size_t cell_count = particle_group->domain->mesh->get_cell_count();

    // Ensure that the total rate buffer is flushed before the reactions are
    // applied
    if (this->auto_clean_tot_rate_buffer) {
      this->rate_buffer_zeroer->transform(particle_group);
    }

    NESOASSERT(this->reactions.size() > 0,
               "ReactionController.apply_reactions(...) cannot be called "
               "without adding at "
               "least one reaction to the ReactionController object (via "
               "ReactionController.add_reaction(...)).");

    for (int r = 0; r < this->reactions.size(); r++) {
      if (!this->reactions[r]->get_in_states().empty()) {
        auto in_states = this->reactions[r]->get_in_states();

        for (int in_state : in_states) {
          this->species_groups.emplace(std::make_pair(
              in_state,
              this->sub_group_selectors[in_state]->make_marker_subgroup(
                  std::make_shared<ParticleSubGroup>(particle_group))));
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

      for (int r = 0; r < reactions.size(); r++) {
        INT in_state = this->reactions[r]->get_in_states()[0];

        this->reactions[r]->descendant_product_loop(
            this->species_groups[in_state], i,
            std::min(i + this->cell_block_size, cell_count), dt, child_group);
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
        transform_buffer->transform(particle_group);
      }
    }
    if (this->child_ids.size() > 0) {
      particle_group->add_particles_local(child_group);
    }
    this->particle_group_temporary->restore(particle_group, child_group);
  }

private:
  std::map<int, std::shared_ptr<MarkingStrategy>> sub_group_selectors;
  std::map<int, ParticleSubGroupSharedPtr> species_groups;

  std::set<int> parent_ids;
  std::set<int> child_ids;

  std::vector<std::shared_ptr<AbstractReaction>> reactions;
  std::vector<std::shared_ptr<TransformationWrapper>> parent_transform;
  std::vector<std::shared_ptr<TransformationWrapper>> child_transform;

  Sym<INT> id_sym;
  Sym<REAL> tot_rate_buffer;
  std::shared_ptr<TransformationWrapper> rate_buffer_zeroer;
  bool auto_clean_tot_rate_buffer;
  size_t cell_block_size = 256;
  size_t max_particles_per_cell = 16384;
  std::shared_ptr<ParticleGroupTemporary>  particle_group_temporary;
  
  inline void setup_particle_group_temporary(){
    this->particle_group_temporary = std::make_shared<ParticleGroupTemporary>();
  }

};
} // namespace Reactions
