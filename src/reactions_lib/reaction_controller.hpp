#pragma once
#include "common_markers.hpp"
#include "particle_group.hpp"
#include "particle_sub_group.hpp"
#include "reaction_base.hpp"
#include "transformation_wrapper.hpp"
#include <map>
#include <memory>
#include <set>
#include <vector>

using namespace NESO::Particles;

namespace Reactions {

/**
 * @brief A reaction controller that orchestrates the application of reactions
 * to a given ParticleGroup.
 *
 * @param child_transform TransformationWrapper informing how descendant
 * products are to be handled
 * @param id_sym Symbol index of the integer ParticleDat that will be used
 * to specify which particles reactions will be applied to
 */
struct ReactionController {

  ReactionController() = default;

  ReactionController(Sym<INT> id_sym) : id_sym(id_sym) {}

  ReactionController(std::shared_ptr<TransformationWrapper> child_transform,
                     Sym<INT> id_sym)
      : child_transform(child_transform), id_sym(id_sym) {}

public:
  /**
   * @brief Function to add reactions to a stored vector of AbstractReaction
   * pointers.
   *
   * @tparam ReactionType The derived type of the reaction object to be added.
   * @param reaction The reaction to be added
   */
  void add_reaction(std::shared_ptr<AbstractReaction> reaction) {
    reactions.push_back(reaction);
  }

  /**
   * @brief Applies all reactions that have been added prior to calling this
   * function. Internally, run_rate_loop and descendant_product_loop are called
   * for each particle and any relevant descendants are handled and added back
   * to the given ParticleGroup
   *
   * @param particle_group The ParticleGroup to apply the reactions to.
   * @param dt The current time step size.
   */
  void apply_reactions(ParticleGroupSharedPtr particle_group, double dt) {
    const int cell_count = particle_group->domain->mesh->get_cell_count();
    std::map<int, std::shared_ptr<MarkingStrategy>> sub_group_selectors;

    std::map<int, ParticleSubGroupSharedPtr> species_groups;

    std::set<int> child_ids;

    auto child_group = std::make_shared<ParticleGroup>(
        particle_group->domain, particle_group->get_particle_spec(),
        particle_group->sycl_target);

    for (int r = 0; r < reactions.size(); r++) {
      INT in_state = reactions[r]->get_in_states()[0];

      auto in_it = species_groups.find(in_state);

      if (in_it == species_groups.end()) {

        sub_group_selectors.emplace(std::make_pair(
            in_state,
            make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(
                id_sym, in_state)));

        species_groups.emplace(std::make_pair(
            in_state, sub_group_selectors[in_state]->make_marker_subgroup(
                          std::make_shared<ParticleSubGroup>(particle_group))));
      }

      if (!reactions[r]->get_out_states().empty()) {
        auto out_states = reactions[r]->get_out_states();

        for (int out_state : out_states) {
          child_ids.insert(out_state);
          auto out_it = sub_group_selectors.find(out_state);

          if (out_it == sub_group_selectors.end()) {

            sub_group_selectors.emplace(std::make_pair(
                out_state, make_marking_strategy<
                               ComparisonMarkerSingle<EqualsComp<INT>, INT>>(
                               id_sym, out_state)));
          }
        }
      }
    }

    for (int i = 0; i < cell_count; i++) {

      for (int r = 0; r < reactions.size(); r++) {

        INT in_state = reactions[r]->get_in_states()[0];

        reactions[r]->run_rate_loop(species_groups[in_state], i);
      }

      for (int r = 0; r < reactions.size(); r++) {
        INT in_state = reactions[r]->get_in_states()[0];

        reactions[r]->descendant_product_loop(species_groups[in_state], i, dt,
                                              child_group);
      }

      for (auto it = child_ids.begin(); it != child_ids.end(); it++) {
        auto transform_buffer =
            std::make_shared<TransformationWrapper>(*child_transform);
        transform_buffer->add_marking_strategy(sub_group_selectors[*it]);
        transform_buffer->transform(child_group);
        
      }
      if (child_ids.size() > 0) {
          particle_group->add_particles_local(child_group);
      }
    }
  }

private:
  std::vector<std::shared_ptr<AbstractReaction>> reactions;
  std::shared_ptr<TransformationWrapper> child_transform;

  Sym<INT> id_sym;
};
} // namespace Reactions

/*
ReactionController :
loop over cells {
    loop over Reactions {
        generate reaction_sub_groups
    }
    loop over Reactions {
        run_rate_loop(...)
    }
    loop over Reactions {
        apply_kernel (currently descendant_products)
    }
    handle cell-wise products (need add_particles_local(DescendantProducts,
ParentGroup))
}

*/
