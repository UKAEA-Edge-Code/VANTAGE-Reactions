#pragma once
#include "data_calculator.hpp"
#include "reaction_data.hpp"
#include "reaction_kernels.hpp"
#include "utils.hpp"
#include <array>
#include <cstring>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/compute_target.hpp>
#include <neso_particles/loop/access_descriptors.hpp>
#include <neso_particles/particle_group.hpp>
#include <neso_particles/particle_sub_group.hpp>
#include <neso_particles/typedefs.hpp>
#include <optional>
#include <particle_properties_map.hpp>
#include <vector>

#define MAX_BUFFER_SIZE 16384

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
 */
struct AbstractReaction {
  AbstractReaction() = default;

  AbstractReaction(SYCLTargetSharedPtr sycl_target,
                   const Sym<REAL> total_rate_dat,
                   Sym<REAL> weight_sym = Sym<REAL>("WEIGHT"))
      : sycl_target_stored(sycl_target), total_reaction_rate(total_rate_dat),
        device_rate_buffer(
            std::make_shared<LocalArray<REAL>>(sycl_target, 0, 0.0)),
        pre_req_data(
            std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, 0, 0, 0.0)),
        weight_sym(weight_sym) {}

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

  // TODO: This probably needs changing now
  virtual void pre_calc_req_data(int cell_idx) {}

  virtual void flush_buffer() {}

  virtual void flush_pre_req_data() {}

protected:
  /**
   * @brief Setters and getters for private members.
   */

  const Sym<REAL> &get_total_reaction_rate() { return total_reaction_rate; }

  void set_total_reaction_rate(const Sym<REAL> &total_reaction_rate_) {
    total_reaction_rate = total_reaction_rate_;
  }

  const LocalArraySharedPtr<REAL> &get_device_rate_buffer() {
    return device_rate_buffer;
  }

  const size_t &get_device_rate_buffer_size() {
    return this->device_rate_buffer->size;
  }

  void set_device_rate_buffer(LocalArraySharedPtr<REAL> &device_rate_buffer_) {
    this->device_rate_buffer = device_rate_buffer_;
  }

  const SYCLTargetSharedPtr &get_sycl_target() { return sycl_target_stored; }

  const NDLocalArraySharedPtr<REAL, 2> &get_pre_req_data() {
    return pre_req_data;
  }

  void set_pre_req_data(NDLocalArraySharedPtr<REAL, 2> &pre_req_data_) {
    this->pre_req_data = pre_req_data_;
  }

  const Sym<REAL> &get_weight_sym() const { return weight_sym; }

private:
  Sym<REAL> total_reaction_rate;
  LocalArraySharedPtr<REAL> device_rate_buffer;
  SYCLTargetSharedPtr sycl_target_stored;
  NDLocalArraySharedPtr<REAL, 2>
      pre_req_data; //!< Real-valued local matrix for storing
                    //!< any pre-requisite data relating to a
                    //!< derived reaction.
  Sym<REAL> weight_sym;
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
 * @tparam DataCalc typename for the DataCalculator object used to calculate
 * prerequisite data
 *
 * @param sycl_target Compute device used by the instance.
 * @param total_rate_dat Symbol index for a ParticleDat that's used to track
 * the cumulative weighted reaction rate modification imposed on all of the
 * particles in the ParticleSubGroup passed to run_rate_loop(...).
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
 * @param particle_spec_ ParticleSpec object containing particle properties to
 * use to construct sym_vectors.
 */
template <int num_products_per_parent, typename ReactionData,
          typename ReactionKernels, typename DataCalc = DataCalculator<>>
struct LinearReactionBase : public AbstractReaction {

  LinearReactionBase() = delete;

  LinearReactionBase(
      SYCLTargetSharedPtr sycl_target, const Sym<REAL> total_rate_dat,
      int in_state, std::array<int, num_products_per_parent> out_states,
      std::vector<ParticleProp<REAL>> real_descendant_particles_props,
      std::vector<ParticleProp<INT>> int_descendant_particles_props,
      ReactionData reaction_data, ReactionKernels reaction_kernels,
      const ParticleSpec &particle_spec_,
      Sym<REAL> weight_sym = Sym<REAL>("WEIGHT"))
      : AbstractReaction(sycl_target, total_rate_dat, weight_sym),
        in_state(in_state), out_states(out_states),
        real_descendant_particles_props(real_descendant_particles_props),
        int_descendant_particles_props(int_descendant_particles_props),
        reaction_data(reaction_data), reaction_kernels(reaction_kernels),
        particle_spec(particle_spec_),
        default_data_calculator(DataCalculator<>(particle_spec_)) {
    // These assertions are necessary since the typenames for ReactionData and
    // ReactionKernels could be any type and for run_rate_loop and
    // descendant_product_loop to operate correctly, ReactionData and
    // ReactionKernels have to be derived from ReactionKernelsBase and
    // AbstractReactionKernels respectively
    static_assert(std::is_base_of_v<ReactionDataBase, ReactionData>,
                  "Template parameter ReactionData is not derived from "
                  "ReactionDataBase...");
    static_assert(std::is_base_of_v<ReactionKernelsBase, ReactionKernels>,
                  "Template parameter ReactionKernels is not derived from "
                  "ReactionKernelsBase...");
    static_assert(std::is_base_of_v<AbstractDataCalculator, DataCalc>,
                  "Template parameter DataCalc is not derived from "
                  "AbstractDataCalculator...");
    auto reaction_data_buffer = this->reaction_data;
    auto reaction_kernel_buffer = this->reaction_kernels;

    run_rate_loop_int_syms = utils::build_sym_vector<INT>(
        this->particle_spec, reaction_data_buffer.get_required_int_props());

    run_rate_loop_real_syms = utils::build_sym_vector<REAL>(
        this->particle_spec, reaction_data_buffer.get_required_real_props());

    descendant_product_loop_int_syms = utils::build_sym_vector<INT>(
        this->particle_spec, reaction_kernel_buffer.get_required_int_props());

    descendant_product_loop_real_syms = utils::build_sym_vector<REAL>(
        this->particle_spec, reaction_kernel_buffer.get_required_real_props());

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

    auto empty_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        AbstractReaction::get_sycl_target(), 0, 0, 0);

    this->set_pre_req_data(empty_pre_req_data);
  }

  /**
   * @brief Constructor with explicit DataCalculator
   *
   * @tparam num_products_per_parent The number of products produced per parent
   * by the derived linear reaction.
   * @tparam ReactionData typename for reaction_data constructor argument
   * @tparam ReactionKernels template class for reaction_kernels constructor
   * argument
   * @tparam DataCalc typename for the DataCalculator object used to calculate
   * prerequisite data
   *
   * @param sycl_target Compute device used by the instance.
   * @param total_rate_dat Symbol index for a ParticleDat that's used to track
   * the cumulative weighted reaction rate modification imposed on all of the
   * particles in the ParticleSubGroup passed to run_rate_loop(...).
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
   * @param data_claculator DataCalculator object for filling in the
   * pre_req_data buffer
   * @param particle_spec_ ParticleSpec object containing particle properties to
   * use to construct sym_vectors.
   */
  LinearReactionBase(
      SYCLTargetSharedPtr sycl_target, const Sym<REAL> total_rate_dat,
      int in_state, std::array<int, num_products_per_parent> out_states,
      std::vector<ParticleProp<REAL>> real_descendant_particles_props,
      std::vector<ParticleProp<INT>> int_descendant_particles_props,
      ReactionData reaction_data, ReactionKernels reaction_kernels,
      const ParticleSpec &particle_spec_, DataCalc data_calculator_,
      Sym<REAL> weight_sym = Sym<REAL>("WEIGHT"))
      : LinearReactionBase(sycl_target, total_rate_dat, in_state, out_states,
                           real_descendant_particles_props,
                           int_descendant_particles_props, reaction_data,
                           reaction_kernels, particle_spec_, weight_sym) {
    this->data_calculator = data_calculator_;

    auto empty_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        AbstractReaction::get_sycl_target(), 0,
        this->data_calculator->get_data_size(), 0);

    this->set_pre_req_data(empty_pre_req_data);
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
                     INT cell_idx) override {
    auto reaction_data_buffer = this->reaction_data;
    auto reaction_data_on_device = reaction_data_buffer.get_on_device_obj();

    auto sycl_target_stored = this->get_sycl_target();
    this->cellwise_flush_buffer(particle_sub_group, cell_idx);
    this->cellwise_flush_pre_req_data(particle_sub_group, cell_idx);
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

    auto loop = particle_loop(
        "calc_rate_loop", particle_sub_group,
        [=](auto particle_index, auto req_int_props, auto req_real_props,
            auto tot_rate, auto buffer) {
          INT current_count = particle_index.get_loop_linear_index();
          REAL rate = reaction_data_on_device.calc_rate(
              particle_index, req_int_props, req_real_props);
          buffer[current_count] = rate;
          tot_rate[0] += rate;
        },
        Access::read(ParticleLoopIndex{}),
        Access::read(
            sym_vector<INT>(particle_sub_group, this->run_rate_loop_int_syms)),
        Access::read(sym_vector<REAL>(particle_sub_group,
                                      this->run_rate_loop_real_syms)),
        Access::write(this->get_total_reaction_rate()),
        Access::write(device_rate_buffer));

    loop->execute(cell_idx);

    if (this->data_calculator) {
      this->data_calculator->fill_buffer(this->get_pre_req_data(),
                                         particle_sub_group, cell_idx);
    } else {
      this->default_data_calculator.fill_buffer(this->get_pre_req_data(),
                                                particle_sub_group, cell_idx);
    }

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
                               ParticleGroupSharedPtr child_group) override {
    auto sycl_target_stored = this->get_sycl_target();
    auto device_rate_buffer = this->get_device_rate_buffer();

    auto reaction_kernel_buffer = this->reaction_kernels;
    auto reaction_kernel_on_device = reaction_kernel_buffer.get_on_device_obj();

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

    this->pre_calc_req_data(cell_idx);

    auto loop = particle_loop(
        "descendant_products_loop", particle_sub_group,
        [=](auto descendant_particle, auto particle_index, auto req_int_props,
            auto req_real_props, auto rate_buffer, auto pre_req_data,
            auto weight, auto total_reaction_rate) {
          INT current_count = particle_index.get_loop_linear_index();
          REAL rate = rate_buffer.at(current_count);

          REAL deltaweight = dt * rate;
          REAL total_deltaweight = dt * total_reaction_rate.at(0);
          REAL modified_weight = std::min(
              deltaweight, deltaweight * (weight.at(0) / total_deltaweight));

          REAL modified_dt = dt * modified_weight / deltaweight;
          for (int childx = 0; childx < num_products_per_parent; childx++) {

            descendant_particle.set_parent(particle_index, childx);
          }

          reaction_kernel_on_device.scattering_kernel(
              modified_weight, particle_index, descendant_particle,
              req_int_props, req_real_props, out_states_arr, pre_req_data,
              modified_dt);

          reaction_kernel_on_device.weight_kernel(
              modified_weight, particle_index, descendant_particle,
              req_int_props, req_real_props, out_states_arr, pre_req_data,
              modified_dt);

          reaction_kernel_on_device.transformation_kernel(
              modified_weight, particle_index, descendant_particle,
              req_int_props, req_real_props, out_states_arr, pre_req_data,
              modified_dt);

          reaction_kernel_on_device.feedback_kernel(
              modified_weight, particle_index, descendant_particle,
              req_int_props, req_real_props, out_states_arr, pre_req_data,
              modified_dt);
        },
        Access::write(this->descendant_particles),
        Access::read(ParticleLoopIndex{}),
        Access::write(sym_vector<INT>(particle_sub_group,
                                      this->descendant_product_loop_int_syms)),
        Access::write(sym_vector(particle_sub_group,
                                 this->descendant_product_loop_real_syms)),
        Access::read(device_rate_buffer),
        Access::read(this->get_pre_req_data()),
        Access::read(this->get_weight_sym()),
        Access::write(this->get_total_reaction_rate()));

    this->descendant_particles->reset(
        particle_sub_group->get_npart_cell(cell_idx));

    loop->execute(cell_idx);

    child_group->add_particles_local(this->descendant_particles,
                                     particle_sub_group->get_particle_group());

    return;
  }

  /**
   * @brief Creates an empty rate buffer of a specified size
   *
   * @param buffer_size Size of the empty buffer that needs to be created and
   * stored.
   */
  void flush_buffer(size_t buffer_size) {
    auto empty_device_rate_buffer = std::make_shared<LocalArray<REAL>>(
        AbstractReaction::get_sycl_target(), buffer_size, 0);
    AbstractReaction::set_device_rate_buffer(empty_device_rate_buffer);
  }

  /**
   * @brief Flushes the rate buffer cellwise, allocating extra memory if
   * necessary.
   *
   * @param particle_sub_group Particle subgroup used to infer the number of
   * particles in the cell
   * @param cell_idx Index of the cell for which the buffer flush is performed
   */
  void cellwise_flush_buffer(ParticleSubGroupSharedPtr particle_sub_group,
                             int cell_idx) {
    auto device_rate_buffer_size = this->get_device_rate_buffer_size();
    auto n_part_cell = particle_sub_group->get_npart_cell(cell_idx);
    if (device_rate_buffer_size < n_part_cell) {
      NESOASSERT(n_part_cell <= MAX_BUFFER_SIZE,
                 "Number of particles in cell exceeds the maximum reaction "
                 "buffer size");
      if ((n_part_cell * 2) < MAX_BUFFER_SIZE) {
        this->flush_buffer(n_part_cell * 2);
      } else {
        this->flush_buffer(MAX_BUFFER_SIZE);
      }
    } else if (n_part_cell < (device_rate_buffer_size / 4)) {
      this->flush_buffer((n_part_cell * 2));
    }
  }

  /**
   * @brief Flushes the stored pre_req_data by setting all values to 0.0.
   */
  void flush_pre_req_data() override { this->get_pre_req_data()->fill(0.0); }

  /**
   * @brief Creates an empty pre_req_data buffer of a specified size, keeping
   * the current number of columns
   *
   * @param buffer_size Number of the empty buffer rows that need to be created
   * and stored.
   */
  void flush_pre_req_data(size_t buffer_size) {
    auto shape = this->get_pre_req_data()->index.shape;
    auto empty_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        AbstractReaction::get_sycl_target(), buffer_size, shape[1], 0);
    AbstractReaction::set_pre_req_data(empty_pre_req_data);
  }

  /**
   * @brief Flushes the pre_req_data buffer cellwise, allocating extra memory if
   * necessary.
   *
   * @param particle_sub_group Particle subgroup used to infer the number of
   * particles in the cell
   * @param cell_idx Index of the cell for which the buffer flush is performed
   */
  void cellwise_flush_pre_req_data(ParticleSubGroupSharedPtr particle_sub_group,
                                   int cell_idx) {
    auto shape = this->get_pre_req_data()->index.shape;
    auto pre_req_buffer_size = shape[0];
    auto n_part_cell = particle_sub_group->get_npart_cell(cell_idx);
    if (pre_req_buffer_size < n_part_cell) {
      NESOASSERT(n_part_cell <= MAX_BUFFER_SIZE,
                 "Number of particles in cell exceeds the maximum reaction "
                 "buffer size");
      if ((n_part_cell * 2) < MAX_BUFFER_SIZE) {
        this->flush_pre_req_data(n_part_cell * 2);
      } else {
        this->flush_pre_req_data(MAX_BUFFER_SIZE);
      }
    } else if (n_part_cell < (pre_req_buffer_size / 4)) {
      this->flush_pre_req_data((n_part_cell * 2));
    }
  }
  /**
   * @brief Getter for in_states that define which species the reaction is to
   * be applied to.
   * @return std::vector<int> Integer vector of species IDs.
   */
  std::vector<int> get_in_states() override {
    return std::vector<int>{this->in_state};
  }

  /**
   * @brief Getter for out_states that define which species the reaction is to
   * produce.
   * @return std::vector<int> Integer vector of species IDs.
   */
  std::vector<int> get_out_states() override {
    return std::vector<int>(out_states.begin(), out_states.end());
  }

  /**
   * @brief Getter for real_descendant_particles_props that define which
   * REAL properties are modified for descendant particles.
   * @return std::vector<ParticleProp<REAL>> Vector of REAL ParticleProps.
   */
  std::vector<ParticleProp<REAL>> get_real_descendant_props() {
    return this->real_descendant_particles_props;
  }

  /**
   * @brief Getter for int_descendant_particles_props that define which
   * INT properties are modified for descendant particles.
   * @return std::vector<ParticleProp<INT>> Vector of INT ParticleProps.
   */
  std::vector<ParticleProp<INT>> get_int_descendant_props() {
    return this->int_descendant_particles_props;
  }

private:
  int in_state;
  std::array<int, num_products_per_parent> out_states;
  std::vector<ParticleProp<REAL>> real_descendant_particles_props;
  std::vector<ParticleProp<INT>> int_descendant_particles_props;
  ReactionData reaction_data;
  ReactionKernels reaction_kernels;
  std::shared_ptr<DescendantProducts> descendant_particles;

  std::vector<Sym<INT>> run_rate_loop_int_syms;
  std::vector<Sym<REAL>> run_rate_loop_real_syms;

  std::vector<Sym<INT>> descendant_product_loop_int_syms;
  std::vector<Sym<REAL>> descendant_product_loop_real_syms;

  ParticleSpec particle_spec;
  std::optional<DataCalc> data_calculator;
  DataCalculator<> default_data_calculator;
};
} // namespace Reactions
