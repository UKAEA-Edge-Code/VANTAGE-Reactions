#pragma once
#include "containers/descendant_products.hpp"
#include "containers/local_array.hpp"
#include "containers/sym_vector.hpp"
#include "loop/access_descriptors.hpp"
#include "particle_group.hpp"
#include "particle_spec.hpp"
#include "particle_sub_group.hpp"
#include "typedefs.hpp"
#include <array>
#include <exception>
#include <memory>
#include <neso_particles.hpp>
#include <vector>

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
 * the cumulative reaction rate modification imposed on all of the particles
 * in the ParticleSubGroup passed to run_rate_loop(...).
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
                   const Sym<REAL> total_rate_dat,
                   const std::vector<Sym<REAL>> required_dats_real_read,
                   std::vector<Sym<REAL>> required_dats_real_write,
                   const std::vector<Sym<INT>> required_dats_int_read,
                   std::vector<Sym<INT>> required_dats_int_write)
      : sycl_target_stored(sycl_target), total_reaction_rate(total_rate_dat),
        read_required_particle_dats_real(required_dats_real_read),
        write_required_particle_dats_real(required_dats_real_write),
        read_required_particle_dats_int(required_dats_int_read),
        write_required_particle_dats_int(required_dats_int_write),
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

  virtual void flush_buffer() {}

protected:
  /**
   * @brief Setters and getters for private members.
   */
  const std::vector<Sym<REAL>> &get_read_req_dats_real() {
    return read_required_particle_dats_real;
  }

  const std::vector<Sym<INT>> &get_read_req_dats_int() {
    return read_required_particle_dats_int;
  }

  std::vector<Sym<REAL>> &get_write_req_dats_real() {
    return write_required_particle_dats_real;
  }

  std::vector<Sym<INT>> &get_write_req_dats_int() {
    return write_required_particle_dats_int;
  }

  const std::vector<REAL> &get_rate_buffer() { return rate_buffer; }

  // TODO: Consider removing (not needed in public interface)
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

private:
  std::vector<Sym<REAL>> read_required_particle_dats_real;
  std::vector<Sym<REAL>> write_required_particle_dats_real;
  std::vector<Sym<INT>> read_required_particle_dats_int;
  std::vector<Sym<INT>> write_required_particle_dats_int;
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
 * @brief SYCL CRTP base linear reaction type. Specifically meant for
 * reactions that only involve a single particle at the start of the reaction
 *
 * @tparam LinearReactionDerived SYCL CRTP template argument
 * @tparam num_products_per_parent The number of products produced per parent
 * by the derived linear reaction.
 * @param sycl_target Compute device used by the instance.
 * @param total_rate_dat Symbol index for a ParticleDat that's used to track
 * the cumulative reaction rate modification imposed on all of the particles
 * in the ParticleSubGroup passed to run_rate_loop(...).
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
 * @param in_states Vector of integers specifying the IDs of the species on
 * which the derived reaction is acting on.
 */
template <typename LinearReactionDerived, INT num_products_per_parent>

struct LinearReactionBase : public AbstractReaction {

  LinearReactionBase() = default;

  LinearReactionBase(SYCLTargetSharedPtr sycl_target,
                     const Sym<REAL> total_rate_dat,
                     const std::vector<Sym<REAL>> required_dats_real_read,
                     std::vector<Sym<REAL>> required_dats_real_write,
                     const std::vector<Sym<INT>> required_dats_int_read,
                     std::vector<Sym<INT>> required_dats_int_write,
                     int in_state,
                     std::array<int, num_products_per_parent> out_states)
      : AbstractReaction(sycl_target, total_rate_dat, required_dats_real_read,
                         required_dats_real_write, required_dats_int_read,
                         required_dats_int_write),
        in_state(in_state), out_states(out_states) {}

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
    const auto &underlying = static_cast<LinearReactionDerived &>(*this);
    auto reaction_data_buffer = underlying.get_reaction_data();

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

    auto loop = particle_loop(
        "calc_rate_loop", particle_sub_group,
        [=](auto particle_index, auto req_reals, auto req_ints, auto tot_rate,
            auto buffer) {
          INT current_count = particle_index.get_loop_linear_index();
          REAL rate = reaction_data_buffer.calc_rate(particle_index, req_reals);
          buffer[current_count] = rate;
          tot_rate[0] += rate;
        },
        Access::read(ParticleLoopIndex{}),
        // The ->get_particle_group() is temporary until sym_vector accepts
        // ParticleSubGroup as an argument
        Access::read(sym_vector<REAL>(particle_sub_group->get_particle_group(),
                                      this->get_read_req_dats_real())),
        Access::read(sym_vector<INT>(particle_sub_group->get_particle_group(),
                                     this->get_read_req_dats_int())),
        Access::write(this->get_total_reaction_rate()),
        Access::write(device_rate_buffer));

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

    // Product matrix spec for descendant particles that specifies which
    // properties of the descendant particles are to be modified in this
    // reaction upon creation of the descendant particles
    auto descendant_particles_spec = product_matrix_spec(
        ParticleSpec(ParticleProp(Sym<REAL>("V"), 2),
                     ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
                     ParticleProp(Sym<INT>("INTERNAL_STATE"), 1)));

    auto descendant_particles = std::make_shared<DescendantProducts>(
        sycl_target_stored, descendant_particles_spec, num_products_per_parent);

    auto loop = particle_loop(
        "descendant_products_loop", particle_sub_group,
        [=](auto descendant_particle, auto particle_index, auto read_req_reals,
            auto read_req_ints, auto write_req_reals, auto write_req_ints,
            auto rate_buffer, auto pre_req_data, auto weight,
            auto total_reaction_rate) {
          INT current_count = particle_index.get_loop_linear_index();
          REAL rate = rate_buffer.at(current_count);

          REAL deltaweight = dt * rate * weight.at(0);
          REAL total_deltaweight =
              dt * total_reaction_rate.at(0) * weight.at(0);
          REAL modified_weight = std::min(
              deltaweight, deltaweight * (weight.at(0) / total_deltaweight));

          for (int childx = 0; childx < num_products_per_parent; childx++) {

            descendant_particle.set_parent(particle_index, childx);
          }

          scattering_kernel(modified_weight, particle_index,
                            descendant_particle, read_req_ints, read_req_reals,
                            write_req_ints, write_req_reals, out_states_arr,
                            pre_req_data, dt);

          weight_kernel(modified_weight, particle_index, descendant_particle,
                        read_req_ints, read_req_reals, write_req_ints,
                        write_req_reals, out_states_arr, pre_req_data, dt);

          transformation_kernel(modified_weight, particle_index,
                                descendant_particle, read_req_ints,
                                read_req_reals, write_req_ints, write_req_reals,
                                out_states_arr, pre_req_data, dt);

          // // move into scattering kernel
          // for (int dimx = 0; dimx < 2; dimx++) {
          //   descendant_particle.at_real(particle_index, childx, 0, dimx) =
          //       read_req_reals.at(0, particle_index, dimx);
          // }

          // //
          // descendant_particle.at_real(particle_index, childx, 1, 0) =
          //     read_req_reals.at(1, particle_index, 0) * (1 + (modified_weight
          //     / num_products_per_parent));

          // descendant_particle.at_int(particle_index, childx, 0,
          //                            0) = out_states_arr[childx];

          feedback_kernel(modified_weight, particle_index, descendant_particle,
                          read_req_ints, read_req_reals, write_req_ints,
                          write_req_reals, out_states_arr, pre_req_data, dt);
        },
        Access::write(descendant_particles), Access::read(ParticleLoopIndex{}),
        // The ->get_particle_group() is temporary until sym_vector accepts
        // ParticleSubGroup as an argument
        Access::read(sym_vector<REAL>(particle_sub_group->get_particle_group(),
                                      this->get_read_req_dats_real())),
        Access::read(sym_vector<INT>(particle_sub_group->get_particle_group(),
                                     this->get_read_req_dats_int())),
        Access::write(sym_vector<REAL>(particle_sub_group->get_particle_group(),
                                       this->get_write_req_dats_real())),
        Access::write(sym_vector<INT>(particle_sub_group->get_particle_group(),
                                      this->get_write_req_dats_int())),
        Access::read(device_rate_buffer),
        Access::read(this->get_pre_req_data()),
        Access::read(this->get_weight_sym()),
        Access::read(this->get_total_reaction_rate()));

    descendant_particles->reset(particle_sub_group->get_npart_cell(cell_idx));

    loop->execute(cell_idx);

    child_group->add_particles_local(descendant_particles,
                                     particle_sub_group->get_particle_group());

    return;
  }

  /**
   * @brief SYCL CRTP base scattering kernel for calculating and applying
   * reaction-derived velocity modifications of the particles.
   * @return std::vector<REAL>
   */
  void
  scattering_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                    Access::DescendantProducts::Write &descendant_products,
                    Access::SymVector::Read<INT> &read_req_ints,
                    Access::SymVector::Read<REAL> &read_req_reals,
                    Access::SymVector::Write<INT> &write_req_ints,
                    Access::SymVector::Write<REAL> &write_req_reals,
                    const std::array<int, num_products_per_parent> &out_states,
                    Access::LocalArray::Read<REAL> &pre_req_data,
                    double dt) const {
    const auto &underlying = static_cast<const LinearReactionDerived &>(*this);

    return underlying.template scattering_kernel(
        modified_weight, index, descendant_products, read_req_ints,
        read_req_reals, write_req_ints, write_req_reals, out_states,
        pre_req_data, dt);
  }

  /**
   * @brief SYCL CRTP base feedback kernel for calculating and applying
   * background field modifications from the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param write_req_ints Symbol indices for integer-valued
   * ParticleDats that need to be modified
   * @param write_req_reals Symbol indices for real-valued
   * ParticleDats that need to be modified
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  void
  feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                  Access::DescendantProducts::Write &descendant_products,
                  Access::SymVector::Read<INT> &read_req_ints,
                  Access::SymVector::Read<REAL> &read_req_reals,
                  Access::SymVector::Write<INT> &write_req_ints,
                  Access::SymVector::Write<REAL> &write_req_reals,
                  const std::array<int, num_products_per_parent> &out_states,
                  Access::LocalArray::Read<REAL> &pre_req_data,
                  double dt) const {
    const auto &underlying = static_cast<const LinearReactionDerived &>(*this);

    return underlying.template feedback_kernel(
        modified_weight, index, descendant_products, read_req_ints,
        read_req_reals, write_req_ints, write_req_reals, out_states,
        pre_req_data, dt);
  }

  /**
   * @brief SYCL CRTP base transformation kernel for calculating and applying
   * reaction-derived ID modifications of the particles.
   */
  void transformation_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Read<INT> &read_req_ints,
      Access::SymVector::Read<REAL> &read_req_reals,
      Access::SymVector::Write<INT> &write_req_ints,
      Access::SymVector::Write<REAL> &write_req_reals,
      const std::array<int, num_products_per_parent> &out_states,
      Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {
    const auto &underlying = static_cast<const LinearReactionDerived &>(*this);

    return underlying.template transformation_kernel(
        modified_weight, index, descendant_products, read_req_ints,
        read_req_reals, write_req_ints, write_req_reals, out_states,
        pre_req_data, dt);
  }

  /**
   * @brief SYCL CRTP base weight kernel for calculating and applying
   * reaction-derived weight modifications of the particles.
   */
  void weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                     Access::DescendantProducts::Write &descendant_products,
                     Access::SymVector::Read<INT> &read_req_ints,
                     Access::SymVector::Read<REAL> &read_req_reals,
                     Access::SymVector::Write<INT> &write_req_ints,
                     Access::SymVector::Write<REAL> &write_req_reals,
                     const std::array<int, num_products_per_parent> &out_states,
                     Access::LocalArray::Read<REAL> &pre_req_data,
                     double dt) const {
    const auto &underlying = static_cast<const LinearReactionDerived &>(*this);

    return underlying.template weight_kernel(
        modified_weight, index, descendant_products, read_req_ints,
        read_req_reals, write_req_ints, write_req_reals, out_states,
        pre_req_data, dt);
  }

  /**
   * @brief SYCL CRTP base apply kernel which combines the calls to
   * scattering_kernel, transformation_kernel and weight_kernel
   */
  // void apply_kernel() const {
  //   const auto &underlying = static_cast<const LinearReactionDerived
  //   &>(*this);

  //   return underlying.template apply_kernel();
  // }

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
  std::vector<int> get_in_states() { return std::vector<int>{in_state}; }

  /**
   * @brief Getter for out_states that define which species the reaction is to
   * produce.
   * @return std::vector<int> Integer vector of species IDs.
   */
  std::vector<int> get_out_states() {
    return std::vector<int>(out_states.begin(), out_states.end());
  }

private:
  int in_state;
  std::array<int, num_products_per_parent> out_states;
};
} // namespace Reactions