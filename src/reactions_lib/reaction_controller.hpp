#pragma once
#include "common_markers.hpp"
#include "containers/cell_dat_const.hpp"
#include "loop/particle_loop.hpp"
#include "particle_group.hpp"
#include "particle_sub_group.hpp"
#include "reaction_base.hpp"
#include "reaction_data.hpp"
#include "transformation_wrapper.hpp"
#include <algorithm>
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

  ReactionController(std::shared_ptr<TransformationStrategy> child_transform,
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
  // TODO: Revert back to std::shared_ptr<AbstractReaction>
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

    std::map<int, ParticleGroupSharedPtr> child_groups;

    auto tot_weight_stored = this->tot_weight_per_child_per_cell;

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
        INT out_state = reactions[r]->get_out_states()[0];

        auto out_it = child_groups.find(out_state);

        if (out_it == child_groups.end()) {
          child_groups.emplace(
              std::make_pair(out_state, std::make_shared<ParticleGroup>(
                                            particle_group->domain,
                                            particle_group->get_particle_spec(),
                                            particle_group->sycl_target)));

          tot_weight_per_child_per_cell.emplace(std::make_pair(
              out_state, make_shared<CellDatConst<REAL>>(
                             particle_group->sycl_target, cell_count, 1, 1)));
        }
      } else {
        child_groups.emplace(std::make_pair(
            -1, std::make_shared<ParticleGroup>(
                    particle_group->domain, particle_group->get_particle_spec(),
                    particle_group->sycl_target)));

        tot_weight_per_child_per_cell.emplace(std::make_pair(
            -1, make_shared<CellDatConst<REAL>>(particle_group->sycl_target,
                                                cell_count, 1, 1)));
      }
    }

    for (int i = 0; i < cell_count; i++) {

      for (int r = 0; r < reactions.size(); r++) {

        INT in_state = reactions[r]->get_in_states()[0];

        reactions[r]->run_rate_loop(species_groups[in_state], i);
      }

      for (int r = 0; r < reactions.size(); r++) {
        INT in_state = reactions[r]->get_in_states()[0];

        if (!reactions[r]->get_out_states().empty()) {
          INT out_state = reactions[r]->get_out_states()[0];

          reactions[r]->descendant_product_loop(species_groups[in_state], i, dt,
                                                child_groups[out_state]);

          // This is done before the child_transform is applied to calculate the
          // total weight of children of a specific internal state(/species id)
          // per cell before they are merged together.
          particle_loop(
              "child_weight_agg", child_groups[out_state],
              [=](auto TotW, auto W) { TotW.fetch_add(0, 0, W[0]); },
              Access::add(tot_weight_per_child_per_cell[out_state]),
              Access::read(Sym<REAL>("COMPUTATIONAL_WEIGHT")))
              ->execute(i);

        } else {
          reactions[r]->descendant_product_loop(species_groups[in_state], i, dt,
                                                child_groups[-1]);

          particle_loop(
              "child_weight_agg", child_groups[-1],
              [=](auto TotW, auto W) { TotW.fetch_add(0, 0, W[0]); },
              Access::add(tot_weight_per_child_per_cell[-1]),
              Access::read(Sym<REAL>("COMPUTATIONAL_WEIGHT")))
              ->execute(i);
        }
      }
    }

    if (child_groups.find(-1) == child_groups.end()) {
      for (auto it = child_groups.begin(); it != child_groups.end(); it++) {
        child_transform->transform(
            std::make_shared<ParticleSubGroup>(it->second));

        particle_group->add_particles_local(it->second);
      }
    }
  }

  std::shared_ptr<CellDatConst<REAL>>
  get_tot_weight_per_cell(int output_state) {
    return tot_weight_per_child_per_cell[output_state];
  }

private:
  std::vector<std::shared_ptr<AbstractReaction>> reactions;
  std::shared_ptr<TransformationStrategy> child_transform;

  // Not quite sure about this approach but it seems to work fine
  std::map<int, std::shared_ptr<CellDatConst<REAL>>>
      tot_weight_per_child_per_cell;

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