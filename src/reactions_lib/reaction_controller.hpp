#pragma once
#include "common_markers.hpp"
#include "particle_group.hpp"
#include "particle_sub_group.hpp"
#include "reaction_base.hpp"
#include "reaction_data.hpp"
#include "transformation_wrapper.hpp"
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

using namespace NESO::Particles;

namespace Reactions {

/**
 * @brief A reaction controller that orchestrates the application of reactions
 * to a given ParticleGroup.
 *
 * @param child_transform TransformationStrategy informing how descendant
 * products are to be handled (eg. via merging or via removal)
 * @param id_sym Symbol index of the integer ParticleDat that will be used
 * to specify which particles reactions will be applied to
 */
struct ReactionController {

  ReactionController() = default;

  ReactionController(Sym<INT> id_sym) : id_sym(id_sym) {}

  ReactionController(TransformationStrategy child_transform, Sym<INT> id_sym)
      : child_transform(child_transform), id_sym(id_sym) {}

public:
  /**
   * @brief Function to add reactions to a stored vector of AbstractReaction
   * pointers.
   *
   * @tparam ReactionType The derived type of the reaction object to be added.
   * @param reaction The reaction to be added
   */
  template <typename ReactionType> void add_reaction(ReactionType &reaction) {
    static_assert(
        std::is_base_of<AbstractReaction,
                        ReactionType>()); //!< This is necessary due to the
                                          //!< inherent flexibility of
                                          //!< templating
    reactions.push_back(&reaction);
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
    int cell_count = particle_group->domain->mesh->get_cell_count();
    std::map<int, std::shared_ptr<MarkingStrategy>> sub_group_selectors;

    std::map<int, ParticleSubGroupSharedPtr> species_groups;

    for (int r = 0; r < reactions.size(); r++) {
      INT in_state = reactions[r]->get_in_states()[0];

      auto it = species_groups.find(in_state);

      if (it == species_groups.end()) {

        sub_group_selectors.emplace(std::make_pair(
            in_state,
            make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(
                id_sym, in_state)));

        species_groups.emplace(std::make_pair(
            in_state, sub_group_selectors[in_state]->make_marker_subgroup(
                          std::make_shared<ParticleSubGroup>(particle_group))));
      }
    }

    for (int i = 0; i < cell_count; i++) {
      auto child_group = std::make_shared<ParticleGroup>(
          particle_group->domain, particle_group->get_particle_spec(),
          particle_group->sycl_target);

      for (int r = 0; r < reactions.size(); r++) {
        int in_state = reactions[r]->get_in_states()[0];

        reactions[r]->run_rate_loop(species_groups[in_state], i);
      }

      for (int r = 0; r < reactions.size(); r++) {
        int in_state = reactions[r]->get_in_states()[0];

        reactions[r]->descendant_product_loop(species_groups[in_state], i, dt,
                                              child_group);
      }

      for (auto select_it = sub_group_selectors.begin();
           select_it != sub_group_selectors.end(); ++select_it) {
        child_transform.transform(select_it->second->make_marker_subgroup(
            make_shared<ParticleSubGroup>(child_group)));
      }

      particle_group->add_particles_local(child_group);
    }
  }

private:
  std::vector<AbstractReaction *> reactions;
  TransformationStrategy child_transform;
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