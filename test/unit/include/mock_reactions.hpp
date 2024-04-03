#include "particle_spec.hpp"
#include "particle_sub_group.hpp"
#include "typedefs.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;

struct TestReactionData : public ReactionDataBase<TestReactionData> {
    TestReactionData() = default;

    TestReactionData(REAL rate_) : rate(rate_){};

    REAL calc_rate(Access::LoopIndex::Read &index,
                   Access::SymVector::Read<REAL> &vars) const {

      return this->rate;
    }

  private:
    REAL rate;
  };

template <INT num_products_per_parent>
struct TestReactionKernels
      : public ReactionKernelsBase<TestReactionKernels<num_products_per_parent>,
                                  num_products_per_parent> {
    TestReactionKernels() = default;

    void scattering_kernel(
        REAL &modified_weight, Access::LoopIndex::Read &index,
        Access::DescendantProducts::Write &descendant_products,
        Access::SymVector::Read<INT> &read_req_ints,
        Access::SymVector::Read<REAL> &read_req_reals,
        Access::SymVector::Write<INT> &write_req_ints,
        Access::SymVector::Write<REAL> &write_req_reals,
        const std::array<int, num_products_per_parent> &out_states,
        Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {
      for (int childx = 0; childx < num_products_per_parent; childx++) {
        for (int dimx = 0; dimx < 2; dimx++) {
          descendant_products.at_real(index, childx, 0, dimx) =
              read_req_reals.at(0, index, dimx);
        }
      }
    }

    void
    weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                  Access::DescendantProducts::Write &descendant_products,
                  Access::SymVector::Read<INT> &read_req_ints,
                  Access::SymVector::Read<REAL> &read_req_reals,
                  Access::SymVector::Write<INT> &write_req_ints,
                  Access::SymVector::Write<REAL> &write_req_reals,
                  const std::array<int, num_products_per_parent> &out_states,
                  Access::LocalArray::Read<REAL> &pre_req_data,
                  double dt) const {
      for (int childx = 0; childx < num_products_per_parent; childx++) {
        descendant_products.at_real(index, childx, 1, 0) =
            (modified_weight / num_products_per_parent);
      }
    }

    void transformation_kernel(
        REAL &modified_weight, Access::LoopIndex::Read &index,
        Access::DescendantProducts::Write &descendant_products,
        Access::SymVector::Read<INT> &read_req_ints,
        Access::SymVector::Read<REAL> &read_req_reals,
        Access::SymVector::Write<INT> &write_req_ints,
        Access::SymVector::Write<REAL> &write_req_reals,
        const std::array<int, num_products_per_parent> &out_states,
        Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {
      for (int childx = 0; childx < num_products_per_parent; childx++) {
        descendant_products.at_int(index, childx, 0, 0) = out_states[childx];
      }
    }

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
      write_req_reals.at(0, index, 0) -= modified_weight;
    }

    private:
    int test_int;
  };


template <INT num_products_per_parent>
struct TestReaction
    : public LinearReactionBase<TestReaction<num_products_per_parent>,
                                num_products_per_parent, TestReactionData, TestReactionKernels> {

  TestReaction() = default;

  TestReaction(SYCLTargetSharedPtr sycl_target_, Sym<REAL> total_reaction_rate_,
               REAL rate_, int in_states_,
               const std::array<int, num_products_per_parent> out_states_)
      : LinearReactionBase<TestReaction<num_products_per_parent>,
                           num_products_per_parent, TestReactionData, TestReactionKernels>(
            sycl_target_, total_reaction_rate_,
            std::vector<Sym<REAL>>{Sym<REAL>("V"),
                                   Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<REAL>>{Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<INT>>{Sym<INT>("INTERNAL_STATE")},
            std::vector<Sym<INT>>(), in_states_, out_states_,
            std::vector<ParticleProp<REAL>>{
                ParticleProp(Sym<REAL>("V"), 2),
                ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1)},
            std::vector<ParticleProp<INT>>{
                ParticleProp(Sym<INT>("INTERNAL_STATE"), 1)},
            TestReactionData(rate_),
            TestReactionKernels<num_products_per_parent>()
            )
        {}
};

struct IoniseReactionData : public ReactionDataBase<IoniseReactionData> {
    IoniseReactionData() = default;

    REAL calc_rate(Access::LoopIndex::Read &index,
                   Access::SymVector::Read<REAL> &vars) const {

      return 1.0;
    }
  };

template <INT num_products_per_parent>
struct IoniseReactionKernels
      : public ReactionKernelsBase<IoniseReactionKernels<num_products_per_parent>, num_products_per_parent> {
    IoniseReactionKernels() = default;

    void
    scattering_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                      Access::DescendantProducts::Write &descendant_products,
                      Access::SymVector::Read<INT> &read_req_ints,
                      Access::SymVector::Read<REAL> &read_req_reals,
                      Access::SymVector::Write<INT> &write_req_ints,
                      Access::SymVector::Write<REAL> &write_req_reals,
                      const std::array<int, 0> &out_states,
                      Access::LocalArray::Read<REAL> &pre_req_data,
                      double dt) const {}

    void weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::DescendantProducts::Write &descendant_products,
                       Access::SymVector::Read<INT> &read_req_ints,
                       Access::SymVector::Read<REAL> &read_req_reals,
                       Access::SymVector::Write<INT> &write_req_ints,
                       Access::SymVector::Write<REAL> &write_req_reals,
                       const std::array<int, 0> &out_states,
                       Access::LocalArray::Read<REAL> &pre_req_data,
                       double dt) const {}

    void transformation_kernel(
        REAL &modified_weight, Access::LoopIndex::Read &index,
        Access::DescendantProducts::Write &descendant_products,
        Access::SymVector::Read<INT> &read_req_ints,
        Access::SymVector::Read<REAL> &read_req_reals,
        Access::SymVector::Write<INT> &write_req_ints,
        Access::SymVector::Write<REAL> &write_req_reals,
        const std::array<int, 0> &out_states,
        Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {}

    void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                         Access::DescendantProducts::Write &descendant_products,
                         Access::SymVector::Read<INT> &read_req_ints,
                         Access::SymVector::Read<REAL> &read_req_reals,
                         Access::SymVector::Write<INT> &write_req_ints,
                         Access::SymVector::Write<REAL> &write_req_reals,
                         const std::array<int, 0> &out_states,
                         Access::LocalArray::Read<REAL> &pre_req_data,
                         double dt) const {
      auto k_V_0 = write_req_reals.at(0, index, 0);
      auto k_V_1 = write_req_reals.at(0, index, 1);
      const REAL vsquared = (k_V_0 * k_V_0) + (k_V_1 * k_V_1);

      REAL k_n_scale = 1.0; // / test_reaction_data.get_n_to_SI();
      REAL inv_k_dt = 1.0 / dt;

      auto nE = write_req_reals.at(2, index, 0);

      // Set SOURCE_DENSITY
      write_req_reals.at(5, index, 0) =
          -nE * modified_weight * k_n_scale * inv_k_dt;

      // Get SOURCE_DENSITY for SOURCE_MOMENTUM calc
      auto k_SD = write_req_reals.at(5, index, 0);
      write_req_reals.at(4, index, 0) = k_SD * k_V_0;
      write_req_reals.at(4, index, 1) = k_SD * k_V_1;

      // Set SOURCE_ENERGY
      write_req_reals.at(3, index, 0) = k_SD * vsquared * 0.5;

      write_req_reals.at(6, index, 0) -= modified_weight;
    }
  };

struct IoniseReaction : public LinearReactionBase<IoniseReaction, 0, IoniseReactionData, IoniseReactionKernels> {

  IoniseReaction() = default;

  IoniseReaction(SYCLTargetSharedPtr sycl_target_,
                 Sym<REAL> total_reaction_rate_, int in_states_)
      : LinearReactionBase<IoniseReaction, 0, IoniseReactionData, IoniseReactionKernels>(
            sycl_target_, total_reaction_rate_,
            std::vector<Sym<REAL>>{
                Sym<REAL>("V"), Sym<REAL>("ELECTRON_TEMPERATURE"),
                Sym<REAL>("ELECTRON_DENSITY"), Sym<REAL>("SOURCE_ENERGY"),
                Sym<REAL>("SOURCE_MOMENTUM"), Sym<REAL>("SOURCE_DENSITY"),
                Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<REAL>>{
                Sym<REAL>("V"), Sym<REAL>("ELECTRON_TEMPERATURE"),
                Sym<REAL>("ELECTRON_DENSITY"), Sym<REAL>("SOURCE_ENERGY"),
                Sym<REAL>("SOURCE_MOMENTUM"), Sym<REAL>("SOURCE_DENSITY"),
                Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<INT>>(), std::vector<Sym<INT>>(), in_states_,
            std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
            std::vector<ParticleProp<INT>>{},
            IoniseReactionData(),
            IoniseReactionKernels<0>()
            )
        {}

};

struct TestReactionVarData : public ReactionDataBase<TestReactionVarData> {
    TestReactionVarData() = default;

    REAL calc_rate(Access::LoopIndex::Read &index,
                   Access::SymVector::Read<REAL> &vars) const {

      return vars.at(0, index, 0);
    }
  };

template <INT num_products_per_parent>
struct TestReactionVarKernels
      : public ReactionKernelsBase<TestReactionVarKernels<num_products_per_parent>, num_products_per_parent> {
    TestReactionVarKernels() = default;

    void
    scattering_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                      Access::DescendantProducts::Write &descendant_products,
                      Access::SymVector::Read<INT> &read_req_ints,
                      Access::SymVector::Read<REAL> &read_req_reals,
                      Access::SymVector::Write<INT> &write_req_ints,
                      Access::SymVector::Write<REAL> &write_req_reals,
                      const std::array<int, 0> &out_states,
                      Access::LocalArray::Read<REAL> &pre_req_data,
                      double dt) const {}

    void weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::DescendantProducts::Write &descendant_products,
                       Access::SymVector::Read<INT> &read_req_ints,
                       Access::SymVector::Read<REAL> &read_req_reals,
                       Access::SymVector::Write<INT> &write_req_ints,
                       Access::SymVector::Write<REAL> &write_req_reals,
                       const std::array<int, 0> &out_states,
                       Access::LocalArray::Read<REAL> &pre_req_data,
                       double dt) const {}

    void transformation_kernel(
        REAL &modified_weight, Access::LoopIndex::Read &index,
        Access::DescendantProducts::Write &descendant_products,
        Access::SymVector::Read<INT> &read_req_ints,
        Access::SymVector::Read<REAL> &read_req_reals,
        Access::SymVector::Write<INT> &write_req_ints,
        Access::SymVector::Write<REAL> &write_req_reals,
        const std::array<int, 0> &out_states,
        Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {}

    void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                         Access::DescendantProducts::Write &descendant_products,
                         Access::SymVector::Read<INT> &read_req_ints,
                         Access::SymVector::Read<REAL> &read_req_reals,
                         Access::SymVector::Write<INT> &write_req_ints,
                         Access::SymVector::Write<REAL> &write_req_reals,
                         const std::array<int, 0> &out_states,
                         Access::LocalArray::Read<REAL> &pre_req_data,
                         double dt) const {
      auto k_W = write_req_reals.at(0, index, 0);
      write_req_reals.at(0, index, 0) += (k_W * modified_weight);
    }
  };

struct TestReactionVarRate : public LinearReactionBase<TestReactionVarRate, 0, TestReactionVarData, TestReactionVarKernels> {

  TestReactionVarRate() = default;

  TestReactionVarRate(SYCLTargetSharedPtr sycl_target_,
                      Sym<REAL> total_reaction_rate_, Sym<REAL> read_var,
                      int in_states_)
      : LinearReactionBase<TestReactionVarRate, 0, TestReactionVarData, TestReactionVarKernels>(
            sycl_target_, total_reaction_rate_,
            std::vector<Sym<REAL>>{read_var, Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<REAL>>{Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<INT>>(), std::vector<Sym<INT>>(), in_states_,
            std::array<int, 0>{}, std::vector<ParticleProp<REAL>>{},
            std::vector<ParticleProp<INT>>{},
            TestReactionVarData(),
            TestReactionVarKernels<0>())
      {}

};

inline auto create_test_particle_group(int N_total)
    -> shared_ptr<ParticleGroup> {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 2;
  dims[1] = 2;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;

  const int global_cell_count =
      dims[0] * dims[1] * std::pow(std::pow(2, subdivision_order), ndim);
  const int npart_per_cell =
      std::round((double)N_total / (double)global_cell_count);

  auto mesh =
      std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                                       subdivision_order, stencil_width);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<REAL>("TOT_REACTION_RATE"), 1),
                             ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
                             ParticleProp(Sym<INT>("INTERNAL_STATE"), 1),
                             ParticleProp(Sym<REAL>("ELECTRON_TEMPERATURE"), 1),
                             ParticleProp(Sym<REAL>("ELECTRON_DENSITY"), 1),
                             ParticleProp(Sym<REAL>("SOURCE_ENERGY"), 1),
                             ParticleProp(Sym<REAL>("SOURCE_MOMENTUM"), ndim),
                             ParticleProp(Sym<REAL>("SOURCE_DENSITY"), 1)};
  auto particle_group =
      std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

  const int cell_count = domain->mesh->get_cell_count();
  const int N = npart_per_cell * cell_count;

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells,
                                 rng_pos);

  auto velocities =
      NESO::Particles::normal_distribution(N, ndim, 0.0, 0.5, rng_vel);
  // std::uniform_int_distribution<int> uniform_dist(
  //     0, size - 1);
  ParticleSet initial_distribution(N, particle_group->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("V")][px][dimx] =
          velocities.at(dimx).at(px);
      initial_distribution[Sym<REAL>("SOURCE_MOMENTUM")][px][dimx] = 0.0;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    initial_distribution[Sym<REAL>("TOT_REACTION_RATE")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("COMPUTATIONAL_WEIGHT")][px][0] = 1.0;
    initial_distribution[Sym<INT>("INTERNAL_STATE")][px][0] = 0;
    initial_distribution[Sym<REAL>("ELECTRON_TEMPERATURE")][px][0] = 2.0;
    initial_distribution[Sym<REAL>("ELECTRON_DENSITY")][px][0] = 3.0e18;
    initial_distribution[Sym<REAL>("SOURCE_ENERGY")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("SOURCE_DENSITY")][px][0] = 0.0;
  }
  particle_group->add_particles_local(initial_distribution);

  auto pbc = std::make_shared<CartesianPeriodic>(sycl_target, mesh,
                                                 particle_group->position_dat);
  auto ccb = std::make_shared<CartesianCellBin>(sycl_target, mesh,
                                                particle_group->position_dat,
                                                particle_group->cell_id_dat);

  pbc->execute();
  particle_group->hybrid_move();
  ccb->execute();
  particle_group->cell_move();

  MPI_Barrier(sycl_target->comm_pair.comm_parent);

  return particle_group;
}
