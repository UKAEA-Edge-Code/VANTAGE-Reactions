#pragma once
#include "data_calculator.hpp"
#include "particle_properties_map.hpp"
#include "reaction_data.hpp"
#include "reaction_kernels.hpp"
#include <array>
#include <cstring>
#include <neso_particles.hpp>
#include <type_traits>

#include <vector>

using namespace NESO::Particles;

// TODO: Improve docs - implementation specific parameter descriptions -
// avoid!!!

namespace Reactions {

/**
 * @brief Abstract base class for reactions. All reactions operate on
 * particles in a ParticleSubGroup in a given cell and modify the
 * ParticleSubGroup and, depending on the reaction, produce and
 * process descendants.
 *
 * @param sycl_target Compute device used by the instance.
 * @param properties_map Optional property remapping. Used to get weight and
 * rate buffer syms.
 */
struct AbstractReaction {
  AbstractReaction() = default;

  AbstractReaction(
      SYCLTargetSharedPtr sycl_target,
      const std::map<int, std::string> &properties_map = default_map)
      : sycl_target_stored(sycl_target),
        total_reaction_rate(
            Sym<REAL>(properties_map.at(default_properties.tot_reaction_rate))),
        device_rate_buffer(
            std::make_shared<LocalArray<REAL>>(sycl_target, 0, 0.0)),
        pre_req_data(
            std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, 0, 0)),
        weight_sym(
            Sym<REAL>(properties_map.at(default_properties.weight))),
        max_buffer_size(16384 * 256) {
    this->pre_req_data->fill(0.0);
  }

public:
  /**
   * @brief Virtual functions to be overidden by an implementation in a derived
   * struct.
   */
  virtual void run_rate_loop(ParticleSubGroupSharedPtr particle_sub_group,
                             INT cell_idx_start, INT cell_idx_end) {}

  virtual void
  descendant_product_loop(ParticleSubGroupSharedPtr particle_sub_group,
                          INT cell_idx_start, INT cell_idx_end, double dt,
                          ParticleGroupSharedPtr child_group,
                          bool full_weight = false) {}

  virtual std::vector<int> get_in_states() { return std::vector<int>{0}; }

  virtual std::vector<int> get_out_states() { return std::vector<int>{0}; }

  virtual void flush_buffer(size_t buffer_size) {}
  virtual void flush_weight_buffer(size_t buffer_size) {}

  virtual void flush_pre_req_data() {}

  /**
   * @brief Set the maximum size for data buffers on this reaction
   *
   * @param max_size Maximum size (per dimension) of data buffers on this
   * reaction
   */
  void set_max_buffer_size(size_t max_size) {
    this->max_buffer_size = max_size;
  }

  const Sym<REAL> &get_weight_sym() const { return weight_sym; }

  void set_weight_sym(const Sym<REAL> &weight_sym) {
    this->weight_sym = weight_sym;
  }

protected:
  /**
   * @brief Setters and getters for private members.
   */

  const Sym<REAL> &get_total_reaction_rate() { return total_reaction_rate; }

  void set_total_reaction_rate(const Sym<REAL> &total_reaction_rate_) {
    this->total_reaction_rate = total_reaction_rate_;
  }

  const LocalArraySharedPtr<REAL> &get_device_rate_buffer() {
    return this->device_rate_buffer;
  }

  const LocalArraySharedPtr<REAL> &get_device_weight_buffer() {
    return this->device_weight_buffer;
  }
  void
  set_device_weight_buffer(LocalArraySharedPtr<REAL> &device_weight_buffer) {
    this->device_weight_buffer = device_weight_buffer;
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

  size_t get_max_buffer_size() { return this->max_buffer_size; }

private:
  Sym<REAL> total_reaction_rate;
  LocalArraySharedPtr<REAL> device_rate_buffer;
  LocalArraySharedPtr<REAL> device_weight_buffer;
  SYCLTargetSharedPtr sycl_target_stored;
  NDLocalArraySharedPtr<REAL, 2>
      pre_req_data; //!< Real-valued local matrix for storing
                    //!< any pre-requisite data relating to a
                    //!< derived reaction.
  Sym<REAL> weight_sym;
  size_t max_buffer_size; //!< max buffer size for data on the reactions object
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
 * prerequisite data (defaults to DataCalculator<>)
 *
 * @param sycl_target Compute device used by the instance.
 * @param in_state Integer specifying the ID of the species on
 * which the derived reaction is acting on.
 * @param out_states Array of integers specifying the species IDs of the
 * descendants produced by the derived reaction.
 * @param reaction_data ReactionData object to be used in run_rate_loop.
 * @param reaction_kernels ReactionKernels object to be used in
 * descendant_product_loop.
 * @param particle_spec_ ParticleSpec object containing particle properties to
 * use to construct sym_vectors.
 * @param data_calculator DataCalculator object for filling in the
 * pre_req_data buffer
 * @param properties_map Optional property remapping. Used to get weight and
 * rate buffer syms.
 */
template <int num_products_per_parent, typename ReactionData,
          typename ReactionKernels, typename DataCalc = DataCalculator<>>
struct LinearReactionBase : public AbstractReaction {

  LinearReactionBase() = delete;

  LinearReactionBase(
      SYCLTargetSharedPtr sycl_target, int in_state,
      std::array<int, num_products_per_parent> out_states,
      ReactionData reaction_data, ReactionKernels reaction_kernels,
      const ParticleSpec &particle_spec_, DataCalc data_calculator_,
      const std::map<int, std::string> &properties_map = default_map)
      : AbstractReaction(sycl_target, properties_map), in_state(in_state),
        out_states(out_states), reaction_data(reaction_data),
        reaction_kernels(reaction_kernels), particle_spec(particle_spec_),
        data_calculator(data_calculator_) {
    // These assertions are necessary since the typenames for ReactionData and
    // ReactionKernels could be any type and for run_rate_loop and
    // descendant_product_loop to operate correctly, ReactionData and
    // ReactionKernels have to be derived from ReactionKernelsBase and
    // AbstractReactionKernels respectively
    static_assert(std::is_base_of_v<
                      ReactionDataBase<reaction_data.get_dim(),
                                       typename ReactionData::RNG_KERNEL_TYPE>,
                      ReactionData>,
                  "Template parameter ReactionData is not derived from "
                  "ReactionDataBase...");
    static_assert(std::is_base_of_v<AbstractDataCalculator, DataCalc>,
                  "Template parameter DataCalc is not derived from "
                  "AbstractDataCalculator...");
    static_assert(std::is_base_of_v<ReactionKernelsBase, ReactionKernels>,
                  "Template parameter ReactionKernels is not derived from "
                  "ReactionKernelsBase...");
    NESOASSERT(this->data_calculator.get_data_size() ==
                   this->reaction_kernels.get_pre_ndims(),
               "The number of ReactionData-derived objects in DataCalculator "
               "does not match the required number of dimensions for the "
               "provided ReactionKernels object.");

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

    auto descendant_matrix_spec =
        reaction_kernel_buffer.get_descendant_matrix_spec();

    this->descendant_particles = std::make_shared<DescendantProducts>(
        this->get_sycl_target(), descendant_matrix_spec,
        num_products_per_parent);

    auto empty_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        AbstractReaction::get_sycl_target(), 0,
        this->data_calculator.get_data_size());
    empty_pre_req_data->fill(0);

    this->set_pre_req_data(empty_pre_req_data);
  }

  /**
   * @brief Constructor with no explicit DataCalculator
   *
   * @tparam num_products_per_parent The number of products produced per parent
   * by the derived linear reaction.
   * @tparam ReactionData typename for reaction_data constructor argument
   * @tparam ReactionKernels template class for reaction_kernels constructor
   * argument
   *
   * @param sycl_target Compute device used by the instance.
   * @param in_state Integer specifying the ID of the species on
   * which the derived reaction is acting on.
   * @param out_states Array of integers specifying the species IDs of the
   * descendants produced by the derived reaction.
   * @param reaction_data ReactionData object to be used in run_rate_loop.
   * @param reaction_kernels ReactionKernels object to be used in
   * descendant_product_loop.
   * @param particle_spec_ ParticleSpec object containing particle properties to
   * use to construct sym_vectors.
   * @param properties_map Optional property remapping. Used to get weight and
   * rate buffer syms.
   */
  LinearReactionBase(
      SYCLTargetSharedPtr sycl_target, int in_state,
      std::array<int, num_products_per_parent> out_states,
      ReactionData reaction_data, ReactionKernels reaction_kernels,
      const ParticleSpec &particle_spec_,
      const std::map<int, std::string> &properties_map = default_map)
      : LinearReactionBase(sycl_target, in_state, out_states, reaction_data,
                           reaction_kernels, particle_spec_, DataCalc(),
                           properties_map) {}

  /**
   * @brief Calculates the reaction rates for all particles in the given
   * particle sub group and cell cell block. Stores the total rate for all
   * particles within a property assigned to each particle (all particles know
   * the total reaction rate) and stores the rate for each particle within a
   * buffer.
   * @param particle_sub_group A ParticleSubGroupSharedPtr that contains
   * particles with the relevant species ID out of the full ParticleGroup
   * @param cell_idx_start The id of the first cell over which to run the
   * principle ParticleLoop to calculate reaction rates.
   * @param cell_idx_end The cell id up to which to run the rate loop over
   */
  void run_rate_loop(ParticleSubGroupSharedPtr particle_sub_group,
                     INT cell_idx_start, INT cell_idx_end) override {

    auto reaction_data_buffer = this->reaction_data;
    auto reaction_data_on_device = reaction_data_buffer.get_on_device_obj();

    auto sycl_target_stored = this->get_sycl_target();
    this->blockwise_flush_buffer(particle_sub_group, cell_idx_start,
                                 cell_idx_end);
    auto device_rate_buffer = this->get_device_rate_buffer();
    auto device_weight_buffer = this->get_device_weight_buffer();

    // The ->get_particle_group() is temporary since ParticleSubGroup doesn't
    // have a sycl_target member
    NESOASSERT(particle_sub_group->get_particle_group()->sycl_target ==
                   sycl_target_stored,
               "sycl_target assigned to particle_group is not the same as "
               "the sycl_target passed to Reaction object...");

    constexpr auto data_dim = reaction_data_buffer.get_dim();

    auto loop = particle_loop(
        "calc_data_loop", particle_sub_group,
        [=](auto particle_index, auto req_int_props, auto req_real_props,
            auto tot_rate, auto buffer, auto weight, auto weight_buffer,
            auto kernel) {
          INT current_count = particle_index.get_loop_linear_index();
          std::array<REAL, data_dim> rate = reaction_data_on_device.calc_data(
              particle_index, req_int_props, req_real_props, kernel);
          buffer[current_count] = rate[0];
          weight_buffer[current_count] =
              weight[0]; // store the particle weight before the application of
                         // any kernels in case we need to know the total weight
                         // of the particle before any reactions are applied
          tot_rate[0] += rate[0];
        },
        Access::read(ParticleLoopIndex{}),
        Access::write(
            sym_vector<INT>(particle_sub_group, this->run_rate_loop_int_syms)),
        Access::read(sym_vector<REAL>(particle_sub_group,
                                      this->run_rate_loop_real_syms)),
        Access::write(this->get_total_reaction_rate()),
        Access::write(device_rate_buffer), Access::read(this->get_weight_sym()),
        Access::write(device_weight_buffer),
        Access::read(this->reaction_data.get_rng_kernel()));

    loop->execute(cell_idx_start, cell_idx_end);

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
   * @param cell_idx_start The id of the first cell over which to run the
   * principle ParticleLoop to determine the effect of reactions.
   * @param cell_idx_end The cell id up to which to run the product loop over
   * @param dt The current time step size.
   * @param child_group ParticleGroupSharedPtr that contains a particle group
   * into which descendants are placed after generation.
   * @param full_weight If true, will consume the full weight of the particles,
   * regardless of timestep
   */
  void descendant_product_loop(ParticleSubGroupSharedPtr particle_sub_group,
                               INT cell_idx_start, INT cell_idx_end, double dt,
                               ParticleGroupSharedPtr child_group,
                               bool full_weight = false) override {
    auto sycl_target_stored = this->get_sycl_target();
    auto device_rate_buffer = this->get_device_rate_buffer();
    auto device_weight_buffer = this->get_device_weight_buffer();

    auto reaction_kernel_buffer = this->reaction_kernels;
    auto reaction_kernel_on_device = reaction_kernel_buffer.get_on_device_obj();

    std::array<int, num_products_per_parent> out_states_arr = this->out_states;
    // The ->get_particle_group() is temporary since ParticleSubGroup doesn't
    // have a sycl_target member
    NESOASSERT(particle_sub_group->get_particle_group()->sycl_target ==
                   sycl_target_stored,
               "sycl_target assigned to particle_group is not the same as "
               "the sycl_target passed to Reaction object...");
    this->blockwise_flush_pre_req_data(particle_sub_group, cell_idx_start,
                                       cell_idx_end);

    this->data_calculator.fill_buffer(this->get_pre_req_data(),
                                      particle_sub_group, cell_idx_start,
                                      cell_idx_end);
    auto loop = particle_loop(
        "descendant_products_loop", particle_sub_group,
        [=](auto descendant_particle, auto particle_index, auto req_int_props,
            auto req_real_props, auto rate_buffer, auto pre_req_data,
            auto weight_buffer, auto total_reaction_rate) {
          INT current_count = particle_index.get_loop_linear_index();
          REAL rate = rate_buffer.at(current_count);

          // If the entire weight is used need to know the weight before any
          // reactions are applied
          REAL used_dt = full_weight ? weight_buffer.at(current_count) /
                                           total_reaction_rate.at(0)
                                     : dt;
          REAL deltaweight =
              used_dt * rate; // weight participating in this reaction
          REAL total_deltaweight =
              used_dt *
              total_reaction_rate.at(0); // Total weight participating in all
                                         // reactions (before rescaling_

          // Use either the above calculated delta weight or a fraction of the
          // initial weight based on how much of the total reaction rate this
          // reaction is responsible
          REAL modified_weight = Kernel::min(
              deltaweight, deltaweight * (weight_buffer.at(current_count) /
                                          total_deltaweight));

          REAL modified_dt = used_dt * modified_weight / deltaweight;
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
        Access::read(device_weight_buffer),
        Access::write(this->get_total_reaction_rate()));

    INT npart_block = 0;
    for (auto cx = cell_idx_start; cx < cell_idx_end; cx++) {
      npart_block += particle_sub_group->get_npart_cell(cx);
    }

    this->descendant_particles->reset(npart_block);

    loop->execute(cell_idx_start, cell_idx_end);

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
  void flush_buffer(size_t buffer_size) override {
    auto empty_device_rate_buffer = std::make_shared<LocalArray<REAL>>(
        AbstractReaction::get_sycl_target(), buffer_size, 0);
    AbstractReaction::set_device_rate_buffer(empty_device_rate_buffer);
  }

  /**
   * @brief Creates an empty weight buffer of a specified size
   *
   * @param buffer_size Size of the empty buffer that needs to be created and
   * stored.
   */
  void flush_weight_buffer(size_t buffer_size) override {
    auto empty_device_buffer = std::make_shared<LocalArray<REAL>>(
        AbstractReaction::get_sycl_target(), buffer_size, 0);
    AbstractReaction::set_device_weight_buffer(empty_device_buffer);
  }
  /**
   * @brief Flushes the rate and weight buffers blockwise, allocating extra
   * memory if necessary.
   *
   * @param particle_sub_group Particle subgroup used to infer the number of
   * particles in the cell
   * @param cell_idx_start Index of the first cell for which the buffer flush is
   * performed
   * @param cell_idx_end Loop end index - cell up to which the buffer is flushed
   */
  void blockwise_flush_buffer(ParticleSubGroupSharedPtr particle_sub_group,
                              int cell_idx_start, int cell_idx_end) {
    auto device_rate_buffer_size = this->get_device_rate_buffer_size();
    INT npart_block = 0;
    for (auto cx = cell_idx_start; cx < cell_idx_end; cx++) {
      npart_block += particle_sub_group->get_npart_cell(cx);
    }

    if (device_rate_buffer_size < npart_block) {
      NESOASSERT(npart_block <= this->get_max_buffer_size(),
                 "Number of particles in cell exceeds the maximum reaction "
                 "buffer size");
      if ((npart_block * 2) < this->get_max_buffer_size()) {
        this->flush_buffer(npart_block * 2);
        this->flush_weight_buffer(npart_block * 2);
      } else {
        this->flush_buffer(this->get_max_buffer_size());
        this->flush_weight_buffer(this->get_max_buffer_size());
      }
    } else if (npart_block < (device_rate_buffer_size / 4)) {
      this->flush_buffer((npart_block * 2));
      this->flush_weight_buffer((npart_block * 2));
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
        AbstractReaction::get_sycl_target(), buffer_size, shape[1]);
    empty_pre_req_data->fill(0);
    AbstractReaction::set_pre_req_data(empty_pre_req_data);
  }

  /**
   * @brief Flushes the pre_req_data buffer blockwise, allocating extra memory
   * if necessary.
   *
   * @param particle_sub_group Particle subgroup used to infer the number of
   * particles in the cell
   * @param cell_idx_start Index of the first cell for which the buffer flush is
   * performed
   * @param cell_idx_end Loop end index - cell up to which the buffer is flushed
   */
  void
  blockwise_flush_pre_req_data(ParticleSubGroupSharedPtr particle_sub_group,
                               int cell_idx_start, int cell_idx_end) {
    auto shape = this->get_pre_req_data()->index.shape;
    auto pre_req_buffer_size = shape[0];
    INT npart_block = 0;
    for (auto cx = cell_idx_start; cx < cell_idx_end; cx++) {
      npart_block += particle_sub_group->get_npart_cell(cx);
    }
    if (pre_req_buffer_size < npart_block) {
      NESOASSERT(npart_block <= this->get_max_buffer_size(),
                 "Number of particles in cell exceeds the maximum reaction "
                 "buffer size");
      if ((npart_block * 2) < this->get_max_buffer_size()) {
        this->flush_pre_req_data(npart_block * 2);
      } else {
        this->flush_pre_req_data(this->get_max_buffer_size());
      }
    } else if (npart_block < (pre_req_buffer_size / 4)) {
      this->flush_pre_req_data((npart_block * 2));
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

private:
  int in_state;
  std::array<int, num_products_per_parent> out_states;
  ReactionData reaction_data;
  ReactionKernels reaction_kernels;
  std::shared_ptr<DescendantProducts> descendant_particles;

  std::vector<Sym<INT>> run_rate_loop_int_syms;
  std::vector<Sym<REAL>> run_rate_loop_real_syms;

  std::vector<Sym<INT>> descendant_product_loop_int_syms;
  std::vector<Sym<REAL>> descendant_product_loop_real_syms;

  ParticleSpec particle_spec;
  DataCalc data_calculator;
};
} // namespace Reactions
