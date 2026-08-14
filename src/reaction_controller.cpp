#include "../include/reactions_lib/reaction_controller.hpp"

namespace VANTAGE::Reactions {

ReactionController::ReactionController(
    std::vector<std::shared_ptr<TransformationWrapper>> parent_transform,
    std::vector<std::shared_ptr<TransformationWrapper>> child_transform,
    bool auto_clean_tot_rate_buffer,
    const std::map<int, std::string> &properties_map)
    : cell_block_size(get_env_size_t("REACTIONS_CELL_BLOCK_SIZE", 256)),
      parent_transform(parent_transform), child_transform(child_transform),
      auto_clean_tot_rate_buffer(auto_clean_tot_rate_buffer) {

  NESOWARN(map_subset_check(properties_map),
           "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

  this->id_sym = Sym<INT>(properties_map.at(default_properties.internal_state));
  this->weight_sym = Sym<REAL>(properties_map.at(default_properties.weight));
  this->tot_rate_buffer =
      Sym<REAL>(properties_map.at(default_properties.tot_reaction_rate));
  this->panic_flag = Sym<INT>(properties_map.at(default_properties.panic));
  this->reacted_flag =
      Sym<INT>(properties_map.at(default_properties.reacted_flag));

  auto zeroer = make_transformation_strategy<ParticleDatZeroer<REAL>>(
      std::vector<std::string>{tot_rate_buffer.name});
  this->rate_buffer_zeroer = std::make_shared<TransformationWrapper>(
      std::dynamic_pointer_cast<TransformationStrategy>(zeroer));
  this->setup_particle_group_temporary();
  this->reacted_marker = make_direct_marking_strategy(
      "reacted_marker", [=](auto reacted) { return reacted[0] == 1; },
      Access::read(this->reacted_flag));
  auto rng_lambda = [&]() -> REAL { return 0; };
  this->rng_kernel =
      std::make_shared<HostPerParticleBlockRNG<REAL>>(rng_lambda, 0);
}

ReactionController::ReactionController(
    bool auto_clean_tot_rate_buffer,
    const std::map<int, std::string> &properties_map)
    : ReactionController(std::vector<std::shared_ptr<TransformationWrapper>>{},
                         std::vector<std::shared_ptr<TransformationWrapper>>{},
                         auto_clean_tot_rate_buffer, properties_map) {}

ReactionController::ReactionController(
    std::shared_ptr<TransformationWrapper> child_transform,
    bool auto_clean_tot_rate_buffer,
    const std::map<int, std::string> &properties_map)
    : ReactionController(std::vector<std::shared_ptr<TransformationWrapper>>{},
                         std::vector{child_transform},
                         auto_clean_tot_rate_buffer, properties_map) {}

ReactionController::ReactionController(
    std::shared_ptr<TransformationWrapper> parent_transform,
    std::shared_ptr<TransformationWrapper> child_transform,
    bool auto_clean_tot_rate_buffer,
    const std::map<int, std::string> &properties_map)
    : ReactionController(std::vector{parent_transform},
                         std::vector{child_transform},
                         auto_clean_tot_rate_buffer, properties_map) {}

void ReactionController::controller_pre_process() {
  for (int r = 0; r < this->reactions.size(); r++) {
    this->reactions[r]->set_max_buffer_size(this->max_particles_per_cell *
                                            this->cell_block_size);
    if (!this->reactions[r]->get_in_states().empty()) {
      auto in_states = this->reactions[r]->get_in_states();

      for (int in_state : in_states) {
        this->parent_ids.insert(in_state);

        this->sub_group_selectors.emplace(std::make_pair(
            in_state, make_direct_marking_strategy(
                          "species_selector_" + std::to_string(in_state),
                          [=](auto id) { return id[0] == in_state; },
                          Access::read(this->id_sym))));
      }
    }

    if (!this->reactions[r]->get_out_states().empty()) {
      auto out_states = this->reactions[r]->get_out_states();

      for (int out_state : out_states) {
        this->child_ids.insert(out_state);

        this->sub_group_selectors.emplace(std::make_pair(
            out_state, make_direct_marking_strategy(
                           "species_selector_" + std::to_string(out_state),
                           [=](auto id) { return id[0] == out_state; },
                           Access::read(this->id_sym))));
      }
    }
  }
}

void ReactionController::add_reaction(
    std::shared_ptr<AbstractReaction> reaction) {
  this->reactions.push_back(reaction);
  this->controller_pre_process();
}

void ReactionController::apply_parent_transforms_impl(
    ParticleSubGroupSharedPtr target, ParticleGroupSharedPtr particle_group) {

  if (this->reference_particle_group == nullptr) {
    this->reference_particle_group = particle_group;
  }

  NESOASSERT(
      particle_group == this->reference_particle_group,
      "Particle group passed to apply_parent_transform is not the same as "
      "recorded reference group.");

  for (auto it = this->parent_ids.begin(); it != this->parent_ids.end(); it++) {
    for (auto tr : this->parent_transform) {
      auto transform_buffer = std::make_shared<TransformationWrapper>(*tr);
      transform_buffer->add_marking_strategy(this->sub_group_selectors[*it]);
      transform_buffer->transform(target);
    }
  }
}

void ReactionController::apply_impl(ParticleSubGroupSharedPtr target,
                                    ParticleGroupSharedPtr particle_group,
                                    double dt,
                                    ParticleGroupSharedPtr product_group,
                                    ControllerMode controller_mode,
                                    bool is_particle_group) {

  if (this->reference_particle_group == nullptr) {
    this->reference_particle_group = particle_group;
  }

  NESOASSERT(particle_group == this->reference_particle_group,
             "Particle group passed to apply is not the same as "
             "recorded reference group.");

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
             "ReactionController.apply(...) cannot be called "
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

        if (is_particle_group) {
          this->species_groups.emplace(std::make_pair(
              in_state,
              this->sub_group_selectors[in_state]->make_marker_subgroup(
                  target)));
        } else {

          this->species_groups[in_state] =
              this->sub_group_selectors[in_state]->make_marker_subgroup(target);
        }
      }

      switch (controller_mode) {

      case ControllerMode::semi_dsmc_mode: {

        for (int in_state : in_states) {
          if (is_particle_group) {
            this->reacted_species_groups.emplace(std::make_pair(
                in_state, this->reacted_marker->make_marker_subgroup(
                              this->species_groups[in_state])));
          } else {

            this->reacted_species_groups[in_state] =
                this->reacted_marker->make_marker_subgroup(
                    this->species_groups[in_state]);
          }
        }
        break;
      }

      default: {
        for (int in_state : in_states) {
          if (is_particle_group) {
            this->reacted_species_groups.emplace(
                std::make_pair(in_state, this->species_groups[in_state]));
          } else {
            this->reacted_species_groups[in_state] =
                this->species_groups[in_state];
          }
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

      this->reactions[r]->calculate_rates(
          this->species_groups[in_state], i,
          std::min(i + this->cell_block_size, cell_count));
    }

    switch (controller_mode) {

    case ControllerMode::semi_dsmc_mode: {

      // marking loop
      auto loop = particle_loop(
          "reacted_loop", target,
          [=](auto index, auto reacted_flag, auto total_reaction_rate,
              auto weight, auto kernel) {
            reacted_flag.at(0) =
                (1 - Kernel::exp(-total_reaction_rate.at(0) * dt / weight[0])) >
                        kernel.at(index, 0)
                    ? 1
                    : 0;
          },
          Access::read(ParticleLoopIndex{}), Access::write(this->reacted_flag),
          Access::read(this->tot_rate_buffer), Access::read(this->weight_sym),
          Access::read(this->rng_kernel));

      loop->execute(i, std::min(i + this->cell_block_size, cell_count));
      rate_buffer_zeroer->transform(
          target, i, std::min(i + this->cell_block_size, cell_count));

      for (int r = 0; r < this->reactions.size(); r++) {

        INT in_state = this->reactions[r]->get_in_states()[0];

        this->reactions[r]->calculate_rates(
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

      this->reactions[r]->apply(this->reacted_species_groups[in_state], i,
                                std::min(i + this->cell_block_size, cell_count),
                                dt, child_group, use_full_weight);
    }

    for (auto it = this->child_ids.begin(); it != this->child_ids.end(); it++) {
      for (auto tr : this->child_transform) {
        auto transform_buffer = std::make_shared<TransformationWrapper>(*tr);
        transform_buffer->add_marking_strategy(this->sub_group_selectors[*it]);
        transform_buffer->transform(
            child_group, i, std::min(i + this->cell_block_size, cell_count));
      }
    }
  }

  if (controller_mode != ControllerMode::surface_mode) {
    this->apply_parent_transforms_impl(target, particle_group);
  }

  if (this->child_ids.size() > 0) {
    product_group->add_particles_local(child_group);
  }
  this->particle_group_temporary->restore(particle_group, child_group);
}

} // namespace VANTAGE::Reactions
