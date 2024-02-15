#include "containers/local_array.hpp"
#include "containers/sym_vector.hpp"
#include "containers/descendant_products.hpp"
#include "loop/access_descriptors.hpp"
#include "particle_group.hpp"
#include "particle_spec.hpp"
#include "particle_sub_group.hpp"
#include "typedefs.hpp"
#include <exception>
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

struct AbstractReaction {
  AbstractReaction() = default;

  AbstractReaction(
    SYCLTargetSharedPtr sycl_target,
    const Sym<REAL> total_rate_dat,
    const std::vector<Sym<REAL>> required_dats_real_read,
    std::vector<Sym<REAL>> required_dats_real_write,
    const std::vector<Sym<INT>> required_dats_int_read,
    std::vector<Sym<INT>> required_dats_int_write
  ) :
  sycl_target_stored(sycl_target),
  total_reaction_rate(total_rate_dat),
  read_required_particle_dats_real(required_dats_real_read),
  write_required_particle_dats_real(required_dats_real_write),
  read_required_particle_dats_int(required_dats_int_read),
  write_required_particle_dats_int(required_dats_int_write),
  device_rate_buffer(LocalArray<REAL>(sycl_target, 0, 0.0)),
  pre_req_data(LocalArray<REAL>(sycl_target, 0, 0.0))
  {}

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
    
    //TODO: Consider removing (not needed in public interface)
    void set_rate_buffer(const std::vector<REAL>& rate_buffer_) {
      rate_buffer = rate_buffer_;
    }

    const Sym<REAL>& get_total_reaction_rate() {
      return total_reaction_rate;
    }

    void set_total_reaction_rate(const Sym<REAL>& total_reaction_rate_) {
      total_reaction_rate = total_reaction_rate_;
    }

    const LocalArray<REAL>& get_device_rate_buffer() {
      return device_rate_buffer;
    }

    const SYCLTargetSharedPtr& get_sycl_target() {
      return sycl_target_stored;
    }

    void flush_buffer() {
    std::fill(this->rate_buffer.begin(), this->rate_buffer.end(), 0.0);
    }

    void flush_buffer(size_t buffer_size) {
      std::vector<REAL> empty_rate_buffer(buffer_size);
      set_rate_buffer(empty_rate_buffer);
      this->device_rate_buffer = LocalArray<REAL>(
              this->sycl_target_stored,
              buffer_size,
              0
          );
      this->flush_buffer();
    }

    const LocalArray<REAL>& get_pre_req_data() const {
      return pre_req_data;
    }

    const Sym<REAL>& get_weight_sym() const {
      return weight_sym;
    }

  private:
    std::vector<Sym<REAL>> read_required_particle_dats_real;
    std::vector<Sym<REAL>> write_required_particle_dats_real;
    std::vector<Sym<INT>> read_required_particle_dats_int;
    std::vector<Sym<INT>> write_required_particle_dats_int;
    std::vector<REAL> rate_buffer;
    Sym<REAL> total_reaction_rate;
    LocalArray<REAL> device_rate_buffer;
    SYCLTargetSharedPtr sycl_target_stored;
    LocalArray<REAL> pre_req_data;
    Sym<REAL> weight_sym = Sym<REAL>("COMPUTATIONAL_WEIGHT");
};


template <typename LinearReactionDerived, INT num_products_per_parent>

struct LinearReactionBase: public AbstractReaction {

  friend struct AbstractReaction;

  LinearReactionBase() = default;

  LinearReactionBase(
    SYCLTargetSharedPtr sycl_target,
    const Sym<REAL> total_rate_dat,
    const std::vector<Sym<REAL>> required_dats_real_read,
    std::vector<Sym<REAL>> required_dats_real_write,
    const std::vector<Sym<INT>> required_dats_int_read,
    std::vector<Sym<INT>> required_dats_int_write
  ): 
  AbstractReaction(
    sycl_target,
    total_rate_dat,
    required_dats_real_read,
    required_dats_real_write,
    required_dats_int_read,
    required_dats_int_write
  )
  {}

  void run_rate_loop(ParticleSubGroupSharedPtr particle_sub_group, INT cell_idx) {

        const auto& underlying = static_cast<LinearReactionDerived&>(*this);
        auto reaction_data_buffer = underlying.get_reaction_data();

        auto sycl_target_stored = this->get_sycl_target();
        auto device_rate_buffer = this->get_device_rate_buffer();

        try {
          // The ->get_particle_group() is temporary since ParticleSubGroup doesn't have a sycl_target member
          if (particle_sub_group->get_particle_group()->sycl_target != sycl_target_stored) {
            throw;
          }
        }
        catch (...) {
          std::cout << "sycl_target assigned to particle_group is not the same as the sycl_target passed to Reaction object..." << std::endl;
        }
        
        auto loop = particle_loop(
            "calc_rate_loop",
            particle_sub_group,
            [=](auto particle_index, auto req_reals, auto req_ints, auto tot_rate, auto buffer){
                INT current_count = particle_index.get_loop_linear_index();
                REAL rate = reaction_data_buffer.calc_rate(particle_index, req_reals);
                buffer[current_count] = rate;
                tot_rate[0] += rate;
            },
            Access::read(ParticleLoopIndex{}),
            // The ->get_particle_group() is temporary until sym_vector accepts ParticleSubGroup as an argument
            Access::read(sym_vector<REAL>(particle_sub_group->get_particle_group(), this->get_read_req_dats_real())),
            Access::read(sym_vector<INT>(particle_sub_group->get_particle_group(), this->get_read_req_dats_int())),
            Access::write(this->get_total_reaction_rate()),
            Access::write(device_rate_buffer)
        );

        loop->execute(cell_idx);

        this->set_rate_buffer(device_rate_buffer.get());

        return;
    }

  void descendant_product_loop(ParticleSubGroupSharedPtr particle_sub_group, INT cell_idx, double dt) {
    auto sycl_target_stored = this->get_sycl_target();
    auto device_rate_buffer = this->get_device_rate_buffer();

    try {
      // The ->get_particle_group() is temporary since ParticleSubGroup doesn't have a sycl_target member
      if (particle_sub_group->get_particle_group()->sycl_target != sycl_target_stored) {
        throw;
      }
    }
    catch (...) {
      std::cout << "sycl_target assigned to particle_group is not the same as the sycl_target passed to Reaction object..." << std::endl;
    }

    auto descendant_particles_spec = product_matrix_spec(
      ParticleSpec(
        ParticleProp(Sym<REAL>("V"), 2),
        ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
        ParticleProp(Sym<INT>("INTERNAL_STATE"), 1)
      )
    );

    auto descendant_particles = std::make_shared<DescendantProducts>(
      sycl_target_stored,
      descendant_particles_spec,
      num_products_per_parent
    );

    auto loop = particle_loop(
      "descendant_products_loop",
      particle_sub_group,
      [=](
        auto descendant_particle,
        auto particle_index,
        auto read_req_reals,
        auto read_req_ints,
        auto write_req_reals,
        auto write_req_ints,
        auto rate_buffer,
        auto pre_req_data,
        auto weight,
        auto total_reaction_rate
      ) {
        INT current_count = particle_index.get_loop_linear_index();
        REAL rate = rate_buffer.at(current_count);

        REAL deltaweight = -dt * rate * weight.at(0);
        REAL total_deltaweight = -dt * total_reaction_rate.at(0) * weight.at(0);
        REAL modified_weight = std::min(deltaweight, deltaweight * (weight.at(0) / total_deltaweight));

        for (int childx=0 ; childx < num_products_per_parent; childx++) {

          descendant_particle.set_parent(particle_index, childx);

          // move into scattering kernel
          for (int dimx = 0 ; dimx < 2 ; dimx++) {
            descendant_particle.at_real(particle_index, childx, 0, dimx) = -1 * read_req_reals.at(0, particle_index, dimx);
          }

          //
          descendant_particle.at_real(particle_index, childx, 1, 0) = read_req_reals.at(1, particle_index, 0);

          descendant_particle.at_int(particle_index, childx, 0, 0) = 1;
        }
        // call feedback_kernel
        feedback_kernel(
          modified_weight,
          particle_index,
          write_req_ints,
          write_req_reals,
          pre_req_data,
          dt
        );
      },
      Access::write(descendant_particles),
      Access::read(ParticleLoopIndex{}),
      // The ->get_particle_group() is temporary until sym_vector accepts ParticleSubGroup as an argument
      Access::read(sym_vector<REAL>(particle_sub_group->get_particle_group(), this->get_read_req_dats_real())),
      Access::read(sym_vector<INT>(particle_sub_group->get_particle_group(), this->get_read_req_dats_int())),
      Access::write(sym_vector<REAL>(particle_sub_group->get_particle_group(), this->get_write_req_dats_real())),
      Access::write(sym_vector<INT>(particle_sub_group->get_particle_group(), this->get_write_req_dats_int())),
      Access::read(device_rate_buffer),
      Access::read(this->get_pre_req_data()),
      Access::read(this->get_weight_sym()),
      Access::read(this->get_total_reaction_rate())
    );

    descendant_particles->reset(particle_sub_group->get_npart_local());

    loop->execute(cell_idx);

    particle_sub_group->get_particle_group()->add_particles_local(descendant_particles);

    return;
  }

  std::vector<REAL> scattering_kernel() const {
    const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

    return underlying.template scattering_kernel();
  }

  // Handle weight modifications of parents as well
  void feedback_kernel(
    REAL& modified_weight,
    Access::LoopIndex::Read& index,
    Access::SymVector::Write<INT>& write_req_ints,
    Access::SymVector::Write<REAL>& write_req_reals,
    Access::LocalArray::Read<REAL>& pre_req_data,
    double dt 
  ) const {
    const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

    return underlying.template feedback_kernel(
      modified_weight,
      index,
      write_req_ints,
      write_req_reals,
      pre_req_data,
      dt
    );
  }

  void transformation_kernel() {
    const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

    return underlying.template transformation_kernel();
  }

  void weight_kernel() {
    const auto& underlying = static_cast<const LinearReactionDerived&>(*this);

    return underlying.template weight_kernel();
  }

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

  void flush_buffer(size_t buffer_size) {
    return AbstractReaction::flush_buffer(buffer_size);
  }

};