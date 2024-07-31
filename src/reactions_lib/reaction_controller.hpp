#pragma once
#include "common_markers.hpp"
#include "common_transformations.hpp"
#include "reaction_base.hpp"
#include "transformation_wrapper.hpp"
#include <map>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/typedefs.hpp>
#include <set>
#include <vector>

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
  }

  /**
   * @brief Function to populate the sub_group_selectors map and
   * parent_ids, child_ids sets.
   */
  void controller_pre_process() {
    for (int r = 0; r < this->reactions.size(); r++) {
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
   * @tparam ReactionType The derived type of the reaction object to be added.
   * @param reaction The reaction to be added
   */
  void add_reaction(std::shared_ptr<AbstractReaction> reaction) {
    this->reactions.push_back(reaction);
    this->controller_pre_process();
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
    const int cell_count = particle_group->domain->mesh->get_cell_count();

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

    auto child_group = std::make_shared<ParticleGroup>(
        particle_group->domain, particle_group->get_particle_spec(),
        particle_group->sycl_target);

    for (int i = 0; i < cell_count; i++) {

      for (int r = 0; r < this->reactions.size(); r++) {

        INT in_state = this->reactions[r]->get_in_states()[0];

        this->reactions[r]->run_rate_loop(this->species_groups[in_state], i);
      }

      for (int r = 0; r < reactions.size(); r++) {
        INT in_state = this->reactions[r]->get_in_states()[0];

        this->reactions[r]->descendant_product_loop(
            this->species_groups[in_state], i, dt, child_group);
      }

      for (auto it = this->child_ids.begin(); it != this->child_ids.end(); it++) {
        for (auto tr : this->child_transform) {
          auto transform_buffer = std::make_shared<TransformationWrapper>(*tr);
          transform_buffer->add_marking_strategy(this->sub_group_selectors[*it]);
          transform_buffer->transform(child_group, i);
        }
      }
    }

    for (auto it = this->parent_ids.begin(); it != this->parent_ids.end(); it++) {
      for (auto tr : this->parent_transform) {
        auto transform_buffer = std::make_shared<TransformationWrapper>(*tr);
        transform_buffer->add_marking_strategy(this->sub_group_selectors[*it]);
        transform_buffer->transform(particle_group);
      }
    }
    if (this->child_ids.size() > 0) {
      particle_group->add_particles_local(child_group);
    }
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
};
} // namespace Reactions
