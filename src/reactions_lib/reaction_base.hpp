#include "containers/local_array.hpp"
#include "containers/sym_vector.hpp"
#include "particle_group.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

template <typename LinearReactionDerived, typename ReactionDataDerived, INT in_state_id>

struct LinearReactionBase {

  LinearReactionBase() = default;

  LinearReactionBase(
    const Sym<REAL> total_rate_dat,
    const std::vector<Sym<REAL>> required_dats_real_read,
    std::vector<Sym<REAL>> required_dats_real_write,
    const std::vector<Sym<INT>> required_dats_int_read,
    std::vector<Sym<INT>> required_dats_int_write,
    ReactionDataDerived reaction_data_
  ): 
  total_reaction_rate(total_rate_dat),
  read_required_particle_dats_real(required_dats_real_read),
  write_required_particle_dats_real(required_dats_real_write),
  read_required_particle_dats_int(required_dats_int_read),
  write_required_particle_dats_int(required_dats_int_write),
  reaction_data(reaction_data_)
  {}

  void run_rate_loop(ParticleGroupSharedPtr particle_group, INT cell_idx) {
    const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

    return underlying.template run_rate_loop(particle_group, cell_idx);
  }

  std::vector<REAL> scattering_kernel() const {
    const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

    return underlying.template scattering_kernel();
  }

  // void feedback_kernel(REAL& weight_fraction,Access::LoopIndex::Read index,Access::SymVector::Read<REAL> vars,Access::SymVector::Write<REAL> write_vars) {
  //   const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

  //   return underlying.template feedback_kernel(reactionData, weight_fraction);
  // }

  // void feedback_kernel(ioniseData& reactionData, REAL& weight_fraction) {
  //   const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

  //   return underlying.template feedback_kernel(reactionData, weight_fraction);
  // }

  // void transformation_kernel() {    
  //   for (auto& out_test_state : this->out_test_states) {
  //     this->post_collision_internal_states.push_back(0);
  //   }
  // }

  // void weight_kernel() {
  //   for (auto& out_test_state : this->out_test_states) {
  //     this->post_collision_weights.push_back(0.0);
  //   }
  // }

  void apply_kernel() const {
    const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

    return underlying.template apply_kernel();
  }

  void flush_buffer() {
    std::fill(this->rate_buffer.begin(), this->rate_buffer.end(), 0.0);
  }

  void flush_buffer(size_t buffer_size) {
    std::vector<REAL> empty_rate_buffer(buffer_size);
    set_rate_buffer(empty_rate_buffer);
    this->flush_buffer();
  }

  protected:
    const std::vector<Sym<REAL>>& get_read_req_dats_real() {
      return read_required_particle_dats_real;
    }

    const std::vector<Sym<INT>>& get_read_req_dats_int() {
      return read_required_particle_dats_int;
    }

    std::vector<Sym<REAL>>& get_write_req_dats_real() {
      return write_required_particle_dats_real;
    }

    std::vector<Sym<INT>>& get_write_req_dats_int() {
      return write_required_particle_dats_int;
    }

    const std::vector<REAL>& get_rate_buffer() {
      return rate_buffer;
    }
    
    void set_rate_buffer(const std::vector<REAL>& rate_buffer_) {
      rate_buffer = rate_buffer_;
    }

    const Sym<REAL>& get_total_reaction_rate() {
      return total_reaction_rate;
    }

    void set_total_reaction_rate(const Sym<REAL>& total_reaction_rate_) {
      total_reaction_rate = total_reaction_rate_;
    }

    const ReactionDataDerived& get_reaction_data() {
      return reaction_data;
    }

    void set_reaction_data(const ReactionDataDerived& reaction_data_) {
      reaction_data = reaction_data_;
    }

  private:
    // std::vector<int> out_test_states;
    std::vector<Sym<REAL>> read_required_particle_dats_real;
    std::vector<Sym<REAL>> write_required_particle_dats_real;
    std::vector<Sym<INT>> read_required_particle_dats_int;
    std::vector<Sym<INT>> write_required_particle_dats_int;
    std::vector<REAL> rate_buffer;
    Sym<REAL> total_reaction_rate;
    ReactionDataDerived reaction_data;
};