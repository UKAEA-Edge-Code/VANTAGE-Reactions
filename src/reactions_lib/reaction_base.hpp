#pragma once
#include "containers/descendant_products.hpp"
#include "containers/local_array.hpp"
#include "containers/sym_vector.hpp"
#include "loop/access_descriptors.hpp"
#include "particle_group.hpp"
#include "particle_spec.hpp"
#include "particle_sub_group.hpp"
#include "reaction_data.hpp"
#include "reaction_kernels.hpp"
#include "typedefs.hpp"
#include <array>
#include <cstring>
#include <memory>
#include <neso_particles.hpp>
#include <vector>
#include <particle_properties_map.hpp>

using namespace NESO::Particles;

namespace Reactions {

/**
 * @brief Abstract base class for reactions. All reactions operate on
 * particles in a ParticleSubGroup in a given cell and modify the
 * ParticleSubGroup and, depending on the reaction, produce and
 * process descendants.
 *
 * @param sycl_target Compute device used by the instance.
 * @param total_rate_dat Symbol index for a ParticleDat that's used to track
 * the cumulative weighted reaction rate modification imposed on all of the
 * particles in the ParticleSubGroup passed to run_rate_loop(...).
 * @param required_dats_real_read Symbol indices for real-valued ParticleDats
 * that are required to be read by either run_rate_loop(...) or
 * descendant_particle_loop(...)
 * @param required_dats_real_write Symbol indices for real-valued ParticleDats
 * that are required to be written by either run_rate_loop(...) or
 * descendant_particle_loop(...)
 * @param required_dats_int_read Symbol indices for integer-valued
 * ParticleDats that are required to be read by either run_rate_loop(...) or
 * descendant_particle_loop(...)
 * @param required_dats_int_write Symbol indices for integer-valued
 * ParticleDats that are required to be written by either run_rate_loop(...) or
 * descendant_particle_loop(...)
 */
struct AbstractReaction {
  AbstractReaction() = default;

  AbstractReaction(SYCLTargetSharedPtr sycl_target,
                   const Sym<REAL> total_rate_dat)
      : sycl_target_stored(sycl_target), total_reaction_rate(total_rate_dat),
        device_rate_buffer(LocalArray<REAL>(sycl_target, 0, 0.0)),
        pre_req_data(LocalArray<REAL>(sycl_target, 0, 0.0)) {}

public:
  /**
   * @brief Virtual functions to be overidden by an implementation in a derived
   * struct.
   */
  virtual void run_rate_loop(ParticleSubGroupSharedPtr particle_sub_group,
                             INT cell_idx) {}

  virtual void
  descendant_product_loop(ParticleSubGroupSharedPtr particle_sub_group,
                          INT cell_idx, double dt,
                          ParticleGroupSharedPtr child_group) {}

  virtual std::vector<int> get_in_states() { return std::vector<int>{0}; }

  virtual std::vector<int> get_out_states() { return std::vector<int>{0}; }

  virtual void pre_calc_req_data(int cell_idx) {}

  virtual void flush_buffer() {}

protected:
  /**
   * @brief Setters and getters for private members.
   */

  const std::vector<REAL> &get_rate_buffer() { return rate_buffer; }

  void set_rate_buffer(const std::vector<REAL> &rate_buffer_) {
    rate_buffer = rate_buffer_;
  }

  const Sym<REAL> &get_total_reaction_rate() { return total_reaction_rate; }

  void set_total_reaction_rate(const Sym<REAL> &total_reaction_rate_) {
    total_reaction_rate = total_reaction_rate_;
  }

  const LocalArray<REAL> &get_device_rate_buffer() {
    return device_rate_buffer;
  }

  void set_device_rate_buffer(LocalArray<REAL> &device_rate_buffer_) {
    this->device_rate_buffer = device_rate_buffer_;
  }

  const SYCLTargetSharedPtr &get_sycl_target() { return sycl_target_stored; }

  const LocalArray<REAL> &get_pre_req_data() const { return pre_req_data; }

  const Sym<REAL> &get_weight_sym() const { return weight_sym; }

  template <typename PROP_TYPE>
  std::vector<Sym<PROP_TYPE>> build_sym_vector(ParticleSpec particle_spec, const char** required_properties, const int num_props) {

    std::vector<Sym<PROP_TYPE>> syms = {};

    for (int iprop = 0; iprop < num_props; iprop++) {
      auto req_prop = required_properties[iprop];
      std::vector<const char *> possible_names;
      try {
        possible_names = ParticlePropertiesIndices::default_map.at(req_prop);
      } catch (std::out_of_range) {
        std::cout << "No instances of " << req_prop
                  << " found in keys of default_map..." << std::endl;
      }
      for (auto &possible_name : possible_names) {
        if constexpr (std::is_same_v<PROP_TYPE, INT>) {
        for (auto &int_prop : particle_spec.properties_int) {
          if (strcmp(int_prop.name.c_str(), possible_name) == 0) {
            syms.push_back(Sym<INT>(int_prop.name));
          }
        }}
        else if constexpr (std::is_same_v<PROP_TYPE, REAL>) {
        for (auto &real_prop : particle_spec.properties_real) {
          if (strcmp(real_prop.name.c_str(), possible_name) == 0) {
            syms.push_back(Sym<REAL>(real_prop.name));
          }
        }}
      }
    }

    return syms;
  }

private:
  std::vector<REAL> rate_buffer;
  Sym<REAL> total_reaction_rate;
  LocalArray<REAL> device_rate_buffer;
  SYCLTargetSharedPtr sycl_target_stored;
  LocalArray<REAL> pre_req_data; //!< Real-valued local array for storing
                                 //!< any pre-requisite data relating to a
                                 //!< derived reaction.
  Sym<REAL> weight_sym = Sym<REAL>("COMPUTATIONAL_WEIGHT");
};

/**
 * @brief Base linear reaction type. Specifically meant for
 * reactions that only involve a single particle at the start of the reaction.
 *
 * @tparam num_products_per_parent The number of products produced per parent
 * by the derived linear reaction.
 * @tparam ReactionData typename for reaction_data constructor argument
 * @tparam ReactionKernels template class for reaction_kernels constructor
 * argument
 * @param sycl_target Compute device used by the instance.
 * @param total_rate_dat Symbol index for a ParticleDat that's used to track
 * the cumulative weighted reaction rate modification imposed on all of the
 * particles in the ParticleSubGroup passed to run_rate_loop(...).
 * @param required_dats_real_read Symbol indices for real-valued ParticleDats
 * that are required to be read by either run_rate_loop(...) or
 * descendant_particle_loop(...)
 * @param required_dats_real_write Symbol indices for real-valued ParticleDats
 * that are required to be written by either run_rate_loop(...) or
 * descendant_particle_loop(...)
 * @param required_dats_int_read Symbol indices for integer-valued
 * ParticleDats that are required to be read by either run_rate_loop(...) or
 * descendant_particle_loop(...)
 * @param required_dats_int_write Symbol indices for integer-valued
 * ParticleDats that are required to be written by either run_rate_loop(...) or
 * descendant_particle_loop(...)
 * @param in_state Integer specifying the ID of the species on
 * which the derived reaction is acting on.
 * @param out_states Array of integers specifying the species IDs of the
 * descendants produced by the derived reaction.
 * @param real_desecendant_particles_props Array of ParticleProp<REAL> that
 * specify the REAL particle props to be modified on the descendant products.
 * @param int_desecendant_particles_props Array of ParticleProp<INT> that
 * specify the INT particle props to be modified on the descendant products.
 * @param reaction_data ReactionData object to be used in run_rate_loop.
 * @param reaction_kernels ReactionKernels object to be used in
 * descendant_product_loop.
 */
template <INT num_products_per_parent, typename ReactionData,
          template <INT> class ReactionKernels>
struct LinearReactionBase : public AbstractReaction {

  LinearReactionBase() = default;

  LinearReactionBase(
      SYCLTargetSharedPtr sycl_target, const Sym<REAL> total_rate_dat, int in_state,
      std::array<int, num_products_per_parent> out_states,
      std::vector<ParticleProp<REAL>> real_descendant_particles_props,
      std::vector<ParticleProp<INT>> int_descendant_particles_props,
      ReactionData reaction_data,
      ReactionKernels<num_products_per_parent> reaction_kernels)
      : AbstractReaction(sycl_target, total_rate_dat),
        in_state(in_state), out_states(out_states),
        real_descendant_particles_props(real_descendant_particles_props),
        int_descendant_particles_props(int_descendant_particles_props),
        reaction_data(reaction_data), reaction_kernels(reaction_kernels) {
    // These assertions are necessary since the typenames for ReactionData and
    // ReactionKernels could be any type and for run_rate_loop and
    // descendant_product_loop to operate correctly, ReactionData and
    // ReactionKernels have to be derived from ReactionKernelsBase and
    // AbstractReactionKernels respectively
    static_assert(std::is_base_of_v<ReactionDataBase, ReactionData>,
                  "Template parameter ReactionData is not derived from "
                  "ReactionDataBase...");
    static_assert(
        std::is_base_of_v<ReactionKernelsBase<num_products_per_parent>,
                          ReactionKernels<num_products_per_parent>>,
        "Template parameter ReactionKernels is not derived from "
        "ReactionKernelsBase...");

    auto descendant_particles_spec = ParticleSpec();
    for (auto ireal_prop : this->get_real_descendant_props()) {
      descendant_particles_spec.push(ireal_prop);
    }
    for (auto iint_prop : this->get_int_descendant_props()) {
      descendant_particles_spec.push(iint_prop);
    }

    // Product matrix spec for descendant particles that specifies which
    // properties of the descendant particles are to be modified in this
    // reaction upon creation of the descendant particles
    auto descendant_matrix_spec =
        product_matrix_spec(descendant_particles_spec);

    this->descendant_particles = std::make_shared<DescendantProducts>(
        this->get_sycl_target(), descendant_matrix_spec,
        num_products_per_parent);
  }

  /**
   * @brief Calculates the reaction rates for all particles in the given
   * particle sub group and cell. Stores the total rate for all particles
   * within a property assigned to each particle (all particles know the
   * total reaction rate) and stores the rate for each particle within a
   * buffer.
   * @param particle_sub_group A ParticleSubGroupSharedPtr that contains
   * particles with the relevant species ID out of the full ParticleGroup
   * @param cell_idx The id of the cell over which to run the principle
   * ParticleLoop to calculate reaction rates.
   */
  void run_rate_loop(ParticleSubGroupSharedPtr particle_sub_group,
                     INT cell_idx) {
    auto reaction_data_buffer = this->reaction_data;

    auto sycl_target_stored = this->get_sycl_target();
    auto device_rate_buffer = this->get_device_rate_buffer();

    try {
      // The ->get_particle_group() is temporary since ParticleSubGroup doesn't
      // have a sycl_target member
      if (particle_sub_group->get_particle_group()->sycl_target !=
          sycl_target_stored) {
        throw;
      }
    } catch (...) {
      std::cout << "sycl_target assigned to particle_group is not the same as "
                   "the sycl_target passed to Reaction object..."
                << std::endl;
    }

    std::vector<Sym<INT>> particle_int_syms =
        this->template build_sym_vector<INT>(
                particle_sub_group->get_particle_group()->get_particle_spec(),
                reaction_data_buffer.get_required_particle_int_props(),
                reaction_data_buffer.get_num_particle_int_props());

    std::vector<Sym<REAL>> particle_real_syms =
        this->template build_sym_vector<REAL>(
                particle_sub_group->get_particle_group()->get_particle_spec(),
                reaction_data_buffer.get_required_particle_real_props(),
                reaction_data_buffer.get_num_particle_real_props());

    std::vector<Sym<INT>> field_int_syms =
        this->template build_sym_vector<INT>(
                particle_sub_group->get_particle_group()->get_particle_spec(),
                reaction_data_buffer.get_required_field_int_props(),
                reaction_data_buffer.get_num_field_int_props());

    std::vector<Sym<REAL>> field_real_syms =
        this->template build_sym_vector<REAL>(
                particle_sub_group->get_particle_group()->get_particle_spec(),
                reaction_data_buffer.get_required_field_real_props(),
                reaction_data_buffer.get_num_field_real_props());

    auto loop = particle_loop(
        "calc_rate_loop", particle_sub_group,
        [=](auto particle_index, auto req_part_ints, auto req_part_reals, auto req_field_ints, auto req_field_reals, auto tot_rate,
            auto buffer, auto weight) {
          INT current_count = particle_index.get_loop_linear_index();
          REAL rate = reaction_data_buffer.calc_rate(particle_index, req_part_ints, req_part_reals, req_field_ints, req_field_reals);
          buffer[current_count] = rate * weight.at(0);
          tot_rate[0] += rate * weight.at(0);
        },
        Access::read(ParticleLoopIndex{}),
        // The ->get_particle_group() is temporary until sym_vector accepts
        // ParticleSubGroup as an argument
        Access::read(sym_vector<INT>(particle_sub_group->get_particle_group(),
                                     particle_int_syms)),
        Access::read(sym_vector<REAL>(particle_sub_group->get_particle_group(),
                                      particle_real_syms)),
        Access::read(sym_vector<INT>(particle_sub_group->get_particle_group(),
                                      field_int_syms)),
        Access::read(sym_vector<REAL>(particle_sub_group->get_particle_group(),
                                      field_real_syms)),
        Access::write(this->get_total_reaction_rate()),
        Access::write(device_rate_buffer),
        Access::read(this->get_weight_sym()));

    loop->execute(cell_idx);

    this->set_rate_buffer(device_rate_buffer.get());

    return;
  }

  /**
   * @brief Creates and processes any descendant products from the reaction
   * and modifies the appropriate background fields and/or parent particle
   * properties based on a weight modification calculation that utilises
   * results from run_rate_loop(...)
   *
   * @param particle_sub_group ParticleSubGroupSharedPtr that contains
   * particles with the relevant species ID out of the full ParticleGroup
   * @param cell_idx The id of the cell over which to run the principle
   * ParticleLoop to calculate reaction rates.
   * @param dt The current time step size.
   * @param child_group ParticleGroupSharedPtr that contains a particle group
   * into which descendants are placed after generation.
   */
  void descendant_product_loop(ParticleSubGroupSharedPtr particle_sub_group,
                               INT cell_idx, double dt,
                               ParticleGroupSharedPtr child_group) {
    auto sycl_target_stored = this->get_sycl_target();
    auto device_rate_buffer = this->get_device_rate_buffer();

    auto reaction_kernel_buffer = this->reaction_kernels;

    std::array<int, num_products_per_parent> out_states_arr = this->out_states;
    try {
      // The ->get_particle_group() is temporary since ParticleSubGroup doesn't
      // have a sycl_target member
      if (particle_sub_group->get_particle_group()->sycl_target !=
          sycl_target_stored) {
        throw;
      }
    } catch (...) {
      std::cout << "sycl_target assigned to particle_group is not the same as "
                   "the sycl_target passed to Reaction object..."
                << std::endl;
    }

    std::vector<Sym<INT>> particle_int_syms =
        this->template build_sym_vector<INT>(
                particle_sub_group->get_particle_group()->get_particle_spec(),
                reaction_kernel_buffer.get_required_particle_int_props(),
                reaction_kernel_buffer.get_num_particle_int_props());

    std::vector<Sym<REAL>> particle_real_syms =
        this->template build_sym_vector<REAL>(
                particle_sub_group->get_particle_group()->get_particle_spec(),
                reaction_kernel_buffer.get_required_particle_real_props(),
                reaction_kernel_buffer.get_num_particle_real_props());

    std::vector<Sym<INT>> field_int_syms =
        this->template build_sym_vector<INT>(
                particle_sub_group->get_particle_group()->get_particle_spec(),
                reaction_kernel_buffer.get_required_field_int_props(),
                reaction_kernel_buffer.get_num_field_int_props());

    std::vector<Sym<REAL>> field_real_syms =
        this->template build_sym_vector<REAL>(
                particle_sub_group->get_particle_group()->get_particle_spec(),
                reaction_kernel_buffer.get_required_field_real_props(),
                reaction_kernel_buffer.get_num_field_real_props());

    this->pre_calc_req_data(cell_idx);

    auto loop = particle_loop(
        "descendant_products_loop", particle_sub_group,
        [=](auto descendant_particle, auto particle_index, auto req_part_ints,
            auto req_part_reals, auto req_field_ints, auto req_field_reals,
            auto rate_buffer, auto pre_req_data, auto weight,
            auto total_reaction_rate) {
          INT current_count = particle_index.get_loop_linear_index();
          REAL rate = rate_buffer.at(current_count);

          REAL deltaweight = dt * rate;
          REAL total_deltaweight = dt * total_reaction_rate.at(0);
          REAL modified_weight = std::min(
              deltaweight, deltaweight * (weight.at(0) / total_deltaweight));

          for (int childx = 0; childx < num_products_per_parent; childx++) {

            descendant_particle.set_parent(particle_index, childx);
          }

          reaction_kernel_buffer.scattering_kernel(
              modified_weight, particle_index, descendant_particle,
              req_part_ints, req_part_reals, req_field_ints, req_field_reals,
              out_states_arr, pre_req_data, dt);

          reaction_kernel_buffer.weight_kernel(
              modified_weight, particle_index, descendant_particle,
              req_part_ints, req_part_reals, req_field_ints, req_field_reals,
              out_states_arr, pre_req_data, dt);

          reaction_kernel_buffer.transformation_kernel(
              modified_weight, particle_index, descendant_particle,
              req_part_ints, req_part_reals, req_field_ints, req_field_reals,
              out_states_arr, pre_req_data, dt);

          reaction_kernel_buffer.feedback_kernel(
              modified_weight, particle_index, descendant_particle,
              req_part_ints, req_part_reals, req_field_ints, req_field_reals,
              out_states_arr, pre_req_data, dt);
        },
        Access::write(this->descendant_particles),
        Access::read(ParticleLoopIndex{}),
        // The ->get_particle_group() is temporary until sym_vector accepts
        // ParticleSubGroup as an argument
        Access::write(sym_vector<INT>(particle_sub_group->get_particle_group(),
                                      particle_int_syms)),
        Access::write(sym_vector<REAL>(particle_sub_group->get_particle_group(),
                                       particle_real_syms)),
        Access::write(sym_vector<INT>(particle_sub_group->get_particle_group(),
                                      field_int_syms)),
        Access::write(sym_vector<REAL>(particle_sub_group->get_particle_group(),
                                       field_real_syms)),
        Access::read(device_rate_buffer),
        Access::read(this->get_pre_req_data()),
        Access::read(this->get_weight_sym()),
        Access::read(this->get_total_reaction_rate()));

    this->descendant_particles->reset(
        particle_sub_group->get_npart_cell(cell_idx));

    loop->execute(cell_idx);

    child_group->add_particles_local(this->descendant_particles,
                                     particle_sub_group->get_particle_group());

    return;
  }

  /**
   * @brief Flushes the stored rate_buffer and by setting all values to 0.0.
   */
  void flush_buffer() override {
    std::vector<REAL> zeroed_rate_buffer = AbstractReaction::get_rate_buffer();
    std::fill(zeroed_rate_buffer.begin(), zeroed_rate_buffer.end(), 0.0);
    AbstractReaction::set_rate_buffer(zeroed_rate_buffer);
  }

  /**
   * @brief Creates an empty rate buffer of a specified size
   *
   * @param buffer_size Size of the empty buffer that needs to be created and
   * stored.
   */
  void flush_buffer(size_t buffer_size) {
    std::vector<REAL> empty_rate_buffer(buffer_size);
    AbstractReaction::set_rate_buffer(empty_rate_buffer);
    auto empty_device_rate_buffer =
        LocalArray<REAL>(AbstractReaction::get_sycl_target(), buffer_size, 0);
    AbstractReaction::set_device_rate_buffer(empty_device_rate_buffer);
    this->flush_buffer();
  }

  /**
   * @brief Getter for in_states that define which species the reaction is to
   * be applied to.
   * @return std::vector<int> Integer vector of species IDs.
   */
  std::vector<int> get_in_states() {
    return std::vector<int>{in_state}; }

  /**
   * @brief Getter for out_states that define which species the reaction is to
   * produce.
   * @return std::vector<int> Integer vector of species IDs.
   */
  std::vector<int> get_out_states() {
    return std::vector<int>(out_states.begin(), out_states.end());
  }

  /**
   * @brief Getter for real_descendant_particles_props that define which
   * REAL properties are modified for descendant particles.
   * @return std::vector<ParticleProp<REAL>> Vector of REAL ParticleProps.
   */
  std::vector<ParticleProp<REAL>> get_real_descendant_props() {
    return std::vector<ParticleProp<REAL>>(
        real_descendant_particles_props.begin(),
        real_descendant_particles_props.end());
  }

  /**
   * @brief Getter for int_descendant_particles_props that define which
   * INT properties are modified for descendant particles.
   * @return std::vector<ParticleProp<INT>> Vector of INT ParticleProps.
   */
  std::vector<ParticleProp<INT>> get_int_descendant_props() {
    return std::vector<ParticleProp<INT>>(
        int_descendant_particles_props.begin(),
        int_descendant_particles_props.end());
  }

private:
  int in_state;
  std::array<int, num_products_per_parent> out_states;
  std::vector<ParticleProp<REAL>> real_descendant_particles_props;
  std::vector<ParticleProp<INT>> int_descendant_particles_props;
  ReactionData reaction_data;
  ReactionKernels<num_products_per_parent> reaction_kernels;
  std::shared_ptr<DescendantProducts> descendant_particles;
};
} // namespace Reactions
