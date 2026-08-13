#ifndef REACTIONS_SWPM_REACTION_CONTROLLER_H
#define REACTIONS_SWPM_REACTION_CONTROLLER_H
#include "collision_cell_manager.hpp"
#include "common_transformations.hpp"
#include "particle_properties_map.hpp"
#include "swpm_coll_specification_abstract.hpp"
#include "swpm_reaction.hpp"
#include "transformation_wrapper.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <tuple>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

template <typename SWPM_SPEC_T, typename RNG_GEN_T>
struct SWPMReactionController {

  SWPMReactionController(
      int species_a_id, int species_b_id,
      std::shared_ptr<SWPM_SPEC_T> swpm_specification,
      std::shared_ptr<RNG_GEN_T> rng_generation_fun,
      std::shared_ptr<CollisionCellManager> coll_cell_manager,
      std::vector<std::shared_ptr<TransformationWrapper>> parent_transform,
      std::vector<std::shared_ptr<TransformationWrapper>> child_transform,
      bool auto_clean_tot_rate_buffer = true,
      bool add_noise_to_partial_collisions = true,
      const std::map<int, std::string> &properties_map = get_default_map())
      : cell_block_size(get_env_size_t("REACTIONS_CELL_BLOCK_SIZE", 256)),
        parent_transform(parent_transform), child_transform(child_transform),
        auto_clean_tot_rate_buffer(auto_clean_tot_rate_buffer),
        add_noise_to_partial_collisions(add_noise_to_partial_collisions),
        rng_generation_fun(rng_generation_fun),
        swpm_specification(swpm_specification),
        coll_cell_manager(coll_cell_manager) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

    this->reactant_set = std::make_tuple(species_a_id, species_b_id);

    // TODO: add consistency check with species in ccm
    this->id_sym =
        Sym<INT>(properties_map.at(default_properties.internal_state));
    this->weight_sym = Sym<REAL>(properties_map.at(default_properties.weight));
    this->tot_rate_buffer =
        Sym<REAL>(properties_map.at(default_properties.tot_reaction_rate));

    auto zeroer = make_transformation_strategy<ParticleDatZeroer<REAL>>(
        std::vector<std::string>{tot_rate_buffer.name});

    this->rate_buffer_zeroer = std::make_shared<TransformationWrapper>(
        std::dynamic_pointer_cast<TransformationStrategy>(zeroer));
    this->setup_particle_group_temporary();

    this->reactant_selector = make_direct_marking_strategy(
        "reactant_selector",
        [=](auto id) { return id[0] == species_a_id || id[0] == species_b_id; },
        Access::read(this->id_sym));
  }

  void controller_pre_process() {
    this->coll_cell_manager->setup_rate_reduction(this->reactions.size());
    for (int r = 0; r < this->reactions.size(); r++) {
      this->reactions[r]->set_max_buffer_size(this->max_pairs_per_cell *
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

  /**
   * @brief Function to add reactions to a stored vector of AbstractPairReaction
   * pointers.
   *
   * @param reaction The reaction to be added
   */
  void add_reaction(std::shared_ptr<AbstractPairReaction> reaction) {
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
    this->max_pairs_per_cell = max_num_parts;
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

  void set_default_rel_vel(REAL rel_vel) { this->default_rel_vel = rel_vel; }

  /**
   * @brief Apply parent transform on the target group or subgroup
   *
   * @param target The ParticleGroup or ParticleSubGroup to apply the transforms
   * to
   */
  template <typename PARENT>
  void apply_parent_transforms(std::shared_ptr<PARENT> target) {

    ParticleGroupSharedPtr particle_group = get_particle_group(target);

    if (this->reference_particle_group == nullptr) {
      this->reference_particle_group = particle_group;
    }

    NESOASSERT(
        particle_group == this->reference_particle_group,
        "Particle group passed to apply_parent_transform is not the same as "
        "recorded reference group.");

    for (auto it = this->parent_ids.begin(); it != this->parent_ids.end();
         it++) {
      for (auto tr : this->parent_transform) {
        auto transform_buffer = std::make_shared<TransformationWrapper>(*tr);
        transform_buffer->add_marking_strategy(this->sub_group_selectors[*it]);
        transform_buffer->transform(target);
      }
    }
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
             ParticleGroupSharedPtr product_group) {

    ParticleGroupSharedPtr particle_group = get_particle_group(target);

    if (this->reference_particle_group == nullptr) {
      this->reference_particle_group = particle_group;
    }

    NESOASSERT(particle_group == this->reference_particle_group,
               "Particle group passed to apply is not the same as "
               "recorded reference group.");
    const size_t cell_count = particle_group->domain->mesh->get_cell_count();

    NESOASSERT(particle_group->contains_dat(this->id_sym, 1),
               "ParticleGroup passed to controller does not contain expected "
               "ID dat, or the dat has wrong dimensionality");
    NESOASSERT(particle_group->contains_dat(this->tot_rate_buffer, 1),
               "ParticleGroup passed to controller does not contain expected "
               "total rate dat, or the dat has wrong dimensionality");

    NESOASSERT(this->reactions.size() > 0,
               "SWPMReactionController.apply(...) cannot be called "
               "without adding at "
               "least one reaction to the SWPMReactionController object (via "
               "SWPMReactionController.add_reaction(...)).");

    auto reactant_subgroup = this->reactant_selector->make_marker_subgroup(
        particle_sub_group(target));

    // Ensure that the total rate buffer is flushed before the reactions are
    // applied
    if (this->auto_clean_tot_rate_buffer) {
      this->rate_buffer_zeroer->transform(reactant_subgroup);
    }

    this->coll_cell_manager->bin_particles(reactant_subgroup);
    this->coll_cell_manager->construct_cell_partition(reactant_subgroup);

    for (int r = 0; r < this->reactions.size(); r++) {

      this->coll_cell_manager->update_rate_reduction(
          r, this->reactions[r]->get_sigma_v_bound(this->default_rel_vel));
    }

    this->coll_cell_manager->get_rate_reduction(this->sigma_v_bound);

    this->swpm_specification->calculate_exponential_parameter(
        reactant_subgroup, this->coll_cell_manager,
        std::get<0>(this->reactant_set), std::get<1>(this->reactant_set),
        this->sigma_v_bound, this->exponential_parameter,
        this->timestep_bounds);

    NDHostArraySharedPtr<REAL, 2> h_timestep_bounds;
    this->timestep_bounds->get(h_timestep_bounds);
    std::vector<REAL> timestep_bounds_vec;
    h_timestep_bounds->get(timestep_bounds_vec);

    // TODO - change from hardcoded:
    // Set here to prevent cases where all particles are consumed in pairs
    REAL max_fraction_pairs = 0.8;

    REAL dt_max = std::min(
        dt, max_fraction_pairs * *std::min_element(timestep_bounds_vec.begin(),
                                                   timestep_bounds_vec.end()));

    REAL current_time = 0;
    auto num_pairs = this->coll_cell_manager->get_empty_coll_cellwise_data<int>(
        particle_group->sycl_target);
    auto noise = this->coll_cell_manager->get_empty_coll_cellwise_data<REAL>(
        particle_group->sycl_target);
    noise->fill(0);
    REAL used_dt;

    if (this->sampler == nullptr) {
      this->sampler = std::make_shared<DSMC::PairSamplerNoReplacement>(
          particle_group->sycl_target, cell_count, this->rng_generation_fun);
    };

    auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
        particle_group, particle_group, this->sampler);

    while (current_time < dt) {

      if (current_time > 0) {
        // TODO: add resolution control features
        this->coll_cell_manager->bin_particles(reactant_subgroup);
        this->coll_cell_manager->construct_cell_partition(reactant_subgroup);
        for (int r = 0; r < this->reactions.size(); r++) {

          this->coll_cell_manager->update_rate_reduction(
              r, this->reactions[r]->get_sigma_v_bound(this->default_rel_vel));
        }
        this->coll_cell_manager->get_rate_reduction(this->sigma_v_bound);

        this->swpm_specification->calculate_exponential_parameter(
            reactant_subgroup, this->coll_cell_manager,
            std::get<0>(this->reactant_set), std::get<1>(this->reactant_set),
            this->sigma_v_bound, this->exponential_parameter,
            this->timestep_bounds);
      }

      used_dt = std::min(dt_max, dt - current_time);

      if (this->add_noise_to_partial_collisions) {
        noise->fill(this->rng_generation_fun);
      }

      // TODO: add random noise to floored value to avoid systematic bias
      nd_local_array_loop_element_wise(
          num_pairs,
          [=](REAL rate_bound, REAL R) {
            return rate_bound > 0 ? sycl::floor(rate_bound * used_dt + R) : 0;
          },
          this->exponential_parameter, noise);

      this->sampler->sample(this->coll_cell_manager->get_cell_partition(),
                            std::get<0>(this->reactant_set),
                            std::get<1>(this->reactant_set), num_pairs);

      auto child_group = this->particle_group_temporary->get(particle_group);

      for (int i = 0; i < cell_count; i += this->cell_block_size) {
        for (int r = 0; r < this->reactions.size(); r++) {

          this->reactions[r]->calculate_rates(
              pair_list, i, std::min(i + this->cell_block_size, cell_count));

          auto buffer = this->reactions[r]->get_device_rate_buffer();

          this->coll_cell_manager->update_rate_reduction(
              pair_list, i, std::min(i + this->cell_block_size, cell_count),
              buffer, r);
        }

        this->swpm_specification->calculate_weight_transfer(
            pair_list, i, std::min(i + this->cell_block_size, cell_count));

        for (int r = 0; r < this->reactions.size(); r++) {

          this->reactions[r]->apply(
              pair_list, i, std::min(i + this->cell_block_size, cell_count), dt,
              child_group);
        }

        for (auto it = this->child_ids.begin(); it != this->child_ids.end();
             it++) {
          for (auto tr : this->child_transform) {
            auto transform_buffer =
                std::make_shared<TransformationWrapper>(*tr);
            transform_buffer->add_marking_strategy(
                this->sub_group_selectors[*it]);
            transform_buffer->transform(
                child_group, i,
                std::min(i + this->cell_block_size, cell_count));
          }
        }
      }

      this->apply_parent_transforms(target);

      if (this->child_ids.size() > 0) {
        product_group->add_particles_local(child_group);
      }
      this->particle_group_temporary->restore(particle_group, child_group);

      // TODO: can this be relaxed?
      this->coll_cell_manager->invalidate_partition();

      current_time += used_dt;
    }
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
  void apply(std::shared_ptr<PARENT> target, double dt) {

    ParticleGroupSharedPtr particle_group = get_particle_group(target);

    this->apply(target, dt, particle_group);
  }

private:
  std::map<int, std::shared_ptr<MarkingStrategy>> sub_group_selectors;
  std::map<int, ParticleSubGroupSharedPtr> species_groups;
  std::map<int, ParticleSubGroupSharedPtr> reacted_species_groups;
  ParticleGroupSharedPtr reference_particle_group = nullptr;

  std::tuple<int, int> reactant_set;

  std::shared_ptr<SWPM_SPEC_T> swpm_specification;
  std::shared_ptr<RNG_GEN_T> rng_generation_fun;
  std::shared_ptr<DSMC::PairSamplerNoReplacement> sampler;
  std::shared_ptr<CollisionCellManager> coll_cell_manager;
  std::set<int> parent_ids;
  std::set<int> child_ids;

  std::vector<std::shared_ptr<AbstractPairReaction>> reactions;
  std::vector<std::shared_ptr<TransformationWrapper>> parent_transform;
  std::vector<std::shared_ptr<TransformationWrapper>> child_transform;

  std::shared_ptr<MarkingStrategy> reactant_selector;

  Sym<INT> id_sym;
  Sym<INT> panic_flag;
  Sym<INT> reacted_flag;
  Sym<REAL> tot_rate_buffer;
  Sym<REAL> weight_sym;
  std::shared_ptr<TransformationWrapper> rate_buffer_zeroer;
  bool auto_clean_tot_rate_buffer;
  size_t cell_block_size = 256;
  size_t max_pairs_per_cell = 16384;
  std::shared_ptr<ParticleGroupTemporary> particle_group_temporary;

  NDLocalArraySharedPtr<REAL, 2> sigma_v_bound;
  NDLocalArraySharedPtr<REAL, 2> exponential_parameter;
  NDLocalArraySharedPtr<REAL, 2> timestep_bounds;

  REAL default_rel_vel = 1.0;

  bool add_noise_to_partial_collisions;

  inline void setup_particle_group_temporary() {
    this->particle_group_temporary = std::make_shared<ParticleGroupTemporary>();
  }
};
} // namespace VANTAGE::Reactions
#endif
