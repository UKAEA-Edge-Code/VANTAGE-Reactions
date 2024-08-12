#pragma once
#include "mock_reactions.hpp"
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <cmath>
#include <data_calculator.hpp>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/compute_target.hpp>
#include <neso_particles/containers/descendant_products.hpp>
#include <neso_particles/containers/product_matrix.hpp>
#include <neso_particles/containers/sym_vector.hpp>
#include <neso_particles/particle_spec.hpp>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>
#include <reaction_kernels.hpp>
#include <string>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;
using namespace ParticlePropertiesIndices;

struct TestReactionDataOnDevice : public ReactionDataBaseOnDevice<> {
  TestReactionDataOnDevice(REAL rate_) : rate(rate_){};

  std::array<REAL,1> calc_data(
      Access::LoopIndex::Read &index,
      Access::SymVector::Read<INT> &req_int_props,
      Access::SymVector::Read<REAL> &req_real_props,
      const typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType &kernel) const {

    return std::array<REAL,1>{this->rate};
  }

private:
  REAL rate;
};

struct TestReactionData : public ReactionDataBase<> {

  TestReactionData(REAL rate_)
      : rate(rate_),
        test_reaction_data_on_device(TestReactionDataOnDevice(rate_)) {}

private:
  TestReactionDataOnDevice test_reaction_data_on_device;

  REAL rate;

public:
  TestReactionDataOnDevice get_on_device_obj() {
    return this->test_reaction_data_on_device;
  }
};

namespace TEST_REACTION_KERNEL {
const auto props = ParticlePropertiesIndices::default_properties;
const std::vector<int> required_simple_real_props = {props.velocity,
                                                     props.weight};

const std::vector<int> required_descendant_simple_int_props = {
    props.internal_state};
const std::vector<int> required_descendant_simple_real_props = {props.velocity,
                                                                props.weight};
} // namespace TEST_REACTION_KERNEL

template <INT num_products_per_parent>
struct TestReactionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<num_products_per_parent> {
  TestReactionKernelsOnDevice() = default;

  void
  scattering_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                    Access::DescendantProducts::Write &descendant_products,
                    Access::SymVector::Write<INT> &req_int_props,
                    Access::SymVector::Write<REAL> &req_real_props,
                    const std::array<int, num_products_per_parent> &out_states,
                    Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                    double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      for (int dimx = 0; dimx < 2; dimx++) {
        descendant_products.at_real(index, childx, descendant_velocity_ind,
                                    dimx) =
            req_real_props.at(velocity_ind, index, dimx);
      }
    }
  }

  void weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                     Access::DescendantProducts::Write &descendant_products,
                     Access::SymVector::Write<INT> &req_int_props,
                     Access::SymVector::Write<REAL> &req_real_props,
                     const std::array<int, num_products_per_parent> &out_states,
                     Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                     double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      descendant_products.at_real(index, childx, descendant_weight_ind, 0) =
          (modified_weight / num_products_per_parent);
    }
  }

  void transformation_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, num_products_per_parent> &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      descendant_products.at_int(index, childx, descendant_internal_state_ind,
                                 0) = out_states[childx];
    }
  }

  void
  feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                  Access::DescendantProducts::Write &descendant_products,
                  Access::SymVector::Write<INT> &req_int_props,
                  Access::SymVector::Write<REAL> &req_real_props,
                  const std::array<int, num_products_per_parent> &out_states,
                  Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                  double dt) const {
    req_real_props.at(weight_ind, index, 0) -= modified_weight;
  }

public:
  int velocity_ind, weight_ind, descendant_velocity_ind, descendant_weight_ind,
      descendant_internal_state_ind;
};

template <INT num_products_per_parent>
struct TestReactionKernels : public ReactionKernelsBase {
  TestReactionKernels()
      : ReactionKernelsBase(Properties<REAL>(
            TEST_REACTION_KERNEL::required_simple_real_props)) {

    this->set_required_descendant_int_props(Properties<INT>(
        TEST_REACTION_KERNEL::required_descendant_simple_int_props));

    this->set_required_descendant_real_props(Properties<REAL>(
        TEST_REACTION_KERNEL::required_descendant_simple_real_props));

    auto props = TEST_REACTION_KERNEL::props;

    this->test_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity);
    this->test_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight);

    this->test_reaction_kernels_on_device.descendant_internal_state_ind =
        this->required_descendant_int_props.simple_prop_index(
            props.internal_state);
    this->test_reaction_kernels_on_device.descendant_velocity_ind =
        this->required_descendant_real_props.simple_prop_index(props.velocity);
    this->test_reaction_kernels_on_device.descendant_weight_ind =
        this->required_descendant_real_props.simple_prop_index(props.weight);

    const auto descendant_internal_state_prop =
        ParticleProp<INT>(Sym<INT>(ParticlePropertiesIndices::default_map.at(
                              props.internal_state)),
                          1);
    const auto descendant_velocity_prop = ParticleProp<REAL>(
        Sym<REAL>(ParticlePropertiesIndices::default_map.at(props.velocity)),
        2);
    const auto descendant_weight_prop = ParticleProp<REAL>(
        Sym<REAL>(ParticlePropertiesIndices::default_map.at(props.weight)), 1);

    auto descendant_particles_spec = ParticleSpec();
    descendant_particles_spec.push(descendant_internal_state_prop);
    descendant_particles_spec.push(descendant_velocity_prop);
    descendant_particles_spec.push(descendant_weight_prop);

    auto matrix_spec = product_matrix_spec(descendant_particles_spec);

    this->set_descendant_matrix_spec(matrix_spec);
  }

private:
  TestReactionKernelsOnDevice<num_products_per_parent>
      test_reaction_kernels_on_device;

public:
  TestReactionKernelsOnDevice<num_products_per_parent> get_on_device_obj() {
    return this->test_reaction_kernels_on_device;
  }
};

template <INT num_products_per_parent>
struct TestReaction
    : public LinearReactionBase<num_products_per_parent, TestReactionData,
                                TestReactionKernels<num_products_per_parent>> {

  TestReaction() = default;

  TestReaction(SYCLTargetSharedPtr sycl_target_, Sym<REAL> total_reaction_rate_,
               REAL rate_, int in_states_,
               const std::array<int, num_products_per_parent> out_states_,
               const ParticleSpec &particle_spec)
      : LinearReactionBase<num_products_per_parent, TestReactionData,
                           TestReactionKernels<num_products_per_parent>>(
            sycl_target_, total_reaction_rate_, in_states_, out_states_,
            TestReactionData(rate_),
            TestReactionKernels<num_products_per_parent>(), particle_spec) {}
};

namespace TEST_REACTION_VAR_DATA {
const auto props = ParticlePropertiesIndices::default_properties;
const std::vector<int> required_simple_real_props = {props.position};
} // namespace TEST_REACTION_VAR_DATA

struct TestReactionVarDataOnDevice : public ReactionDataBaseOnDevice<> {
  TestReactionVarDataOnDevice() = default;

  std::array<REAL,1>calc_data(
      Access::LoopIndex::Read &index,
      Access::SymVector::Read<INT> req_int_props,
      Access::SymVector::Read<REAL> req_real_props,
      const typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType &kernel) const {

    return std::array<REAL,1>{req_real_props.at(position_ind, index, 0)};
  }

public:
  int position_ind;
};

struct TestReactionVarData : public ReactionDataBase<> {
  TestReactionVarData()
      : ReactionDataBase(
            Properties<REAL>(TEST_REACTION_VAR_DATA::required_simple_real_props,
                             std::vector<Species>{}, std::vector<int>{})) {
    auto props = TEST_REACTION_VAR_DATA::props;

    this->test_reaction_var_data_on_device.position_ind =
        this->required_real_props.simple_prop_index(props.position);
  };

private:
  TestReactionVarDataOnDevice test_reaction_var_data_on_device;

public:
  TestReactionVarDataOnDevice get_on_device_obj() {
    return this->test_reaction_var_data_on_device;
  }
};

namespace TEST_REACTION_VAR_KERNEL {
constexpr int num_products_per_parent = 0;

const auto props = ParticlePropertiesIndices::default_properties;

const std::vector<int> required_simple_real_props = {props.weight};
} // namespace TEST_REACTION_VAR_KERNEL

struct TestReactionVarKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<
          TEST_REACTION_VAR_KERNEL::num_products_per_parent> {
  void feedback_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, TEST_REACTION_VAR_KERNEL::num_products_per_parent>
          &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    req_real_props.at(weight_ind, index, 0) -= modified_weight;
  }

public:
  int weight_ind;
};

struct TestReactionVarKernels : public ReactionKernelsBase {
  TestReactionVarKernels()
      : ReactionKernelsBase(Properties<REAL>(
            TEST_REACTION_VAR_KERNEL::required_simple_real_props,
            std::vector<Species>{}, std::vector<int>{})) {

    auto props = TEST_REACTION_VAR_KERNEL::props;

    this->test_reaction_var_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight);
  };

private:
  TestReactionVarKernelsOnDevice test_reaction_var_kernels_on_device;

public:
  TestReactionVarKernelsOnDevice get_on_device_obj() {
    return this->test_reaction_var_kernels_on_device;
  }
};

struct TestReactionVarRate : public LinearReactionBase<0, TestReactionVarData,
                                                       TestReactionVarKernels> {

  TestReactionVarRate(SYCLTargetSharedPtr sycl_target_,
                      Sym<REAL> total_reaction_rate_, int in_states_,
                      const ParticleSpec &particle_spec)
      : LinearReactionBase<0, TestReactionVarData, TestReactionVarKernels>(
            sycl_target_, total_reaction_rate_, in_states_,
            std::array<int, 0>{}, TestReactionVarData(),
            TestReactionVarKernels(), particle_spec) {}
};

namespace TEST_REACTION_KERNEL_DATA_CALC {
const auto props = ParticlePropertiesIndices::default_properties;
const std::vector<int> required_simple_real_props = {props.velocity,
                                                     props.weight};
const std::vector<int> required_species_real_props = {props.source_density,
                                                      props.source_energy};
} // namespace TEST_REACTION_KERNEL_DATA_CALC

template <INT num_products_per_parent>
struct TestReactionKernelsDataCalcOnDevice
    : public ReactionKernelsBaseOnDevice<num_products_per_parent> {
  TestReactionKernelsDataCalcOnDevice() = default;

  void
  scattering_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                    Access::DescendantProducts::Write &descendant_products,
                    Access::SymVector::Write<INT> &req_int_props,
                    Access::SymVector::Write<REAL> &req_real_props,
                    const std::array<int, num_products_per_parent> &out_states,
                    Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                    double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      for (int dimx = 0; dimx < 2; dimx++) {
        descendant_products.at_real(index, childx, 0, dimx) =
            req_real_props.at(velocity_ind, index, dimx);
      }
    }
  }

  void weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                     Access::DescendantProducts::Write &descendant_products,
                     Access::SymVector::Write<INT> &req_int_props,
                     Access::SymVector::Write<REAL> &req_real_props,
                     const std::array<int, num_products_per_parent> &out_states,
                     Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                     double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      descendant_products.at_real(index, childx, 1, 0) =
          (modified_weight / num_products_per_parent);
    }
  }

  void transformation_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, num_products_per_parent> &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      descendant_products.at_int(index, childx, 0, 0) = out_states[childx];
    }
  }

  void
  feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                  Access::DescendantProducts::Write &descendant_products,
                  Access::SymVector::Write<INT> &req_int_props,
                  Access::SymVector::Write<REAL> &req_real_props,
                  const std::array<int, num_products_per_parent> &out_states,
                  Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                  double dt) const {
    req_real_props.at(weight_ind, index, 0) -= modified_weight;
    req_real_props.at(source_ind, index, 0) +=
        pre_req_data.at(index.get_loop_linear_index(), 0);
    req_real_props.at(energy_source_ind, index, 0) +=
        pre_req_data.at(index.get_loop_linear_index(), 1);
  }

public:
  int velocity_ind, weight_ind, source_ind, energy_source_ind;
};

template <INT num_products_per_parent>
struct TestReactionDataCalcKernels : public ReactionKernelsBase {
  TestReactionDataCalcKernels()
      : ReactionKernelsBase(
            Properties<REAL>(
                TEST_REACTION_KERNEL_DATA_CALC::required_simple_real_props,
                std::vector<Species>{Species("ELECTRON")},
                TEST_REACTION_KERNEL_DATA_CALC::required_species_real_props),
            2) {

    auto props = TEST_REACTION_KERNEL::props;

    this->test_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity);
    this->test_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight);
    this->test_reaction_kernels_on_device.source_ind =
        this->required_real_props.species_prop_index("ELECTRON",
                                                     props.source_density);
    this->test_reaction_kernels_on_device.energy_source_ind =
        this->required_real_props.species_prop_index("ELECTRON",
                                                     props.source_energy);
  }

  // public:
  // std::vector<std::string> get_required_real_props() {
  //   std::vector<std::string> simple_props;
  //   try {
  //     simple_props = this->required_real_props.simple_prop_names();
  //   } catch (std::logic_error) {
  //     simple_props = {};
  //   }
  //   std::vector<std::string> species_props;
  //   try {
  //     species_props = this->required_real_props.species_prop_names();
  //   } catch (std::logic_error) {
  //     species_props = {};
  //   }
  //   simple_props.insert(simple_props.end(), species_props.begin(),
  //                       species_props.end());
  //   return simple_props;
  // }

private:
  TestReactionKernelsDataCalcOnDevice<num_products_per_parent>
      test_reaction_kernels_on_device;

  // Properties<REAL> required_real_props;

public:
  TestReactionKernelsDataCalcOnDevice<num_products_per_parent>
  get_on_device_obj() {
    return this->test_reaction_kernels_on_device;
  }
};

inline auto create_test_particle_group(int N_total)
    -> std::shared_ptr<ParticleGroup> {

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

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("POSITION"), ndim, true),
      ParticleProp(Sym<REAL>("VELOCITY"), ndim),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ID"), 1),
      ParticleProp(Sym<REAL>("TOT_REACTION_RATE"), 1),
      ParticleProp(Sym<REAL>("WEIGHT"), 1),
      ParticleProp(Sym<INT>("INTERNAL_STATE"), 1),
      ParticleProp(Sym<REAL>("ELECTRON_TEMPERATURE"), 1),
      ParticleProp(Sym<REAL>("ELECTRON_DENSITY"), 1),
      ParticleProp(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), 1),
      ParticleProp(Sym<REAL>("ELECTRON_SOURCE_MOMENTUM"), ndim),
      ParticleProp(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), 1),
      ParticleProp(Sym<REAL>("ION_SOURCE_DENSITY"), 1),
      ParticleProp(Sym<REAL>("ION_SOURCE_MOMENTUM"), ndim),
      ParticleProp(Sym<REAL>("ION_SOURCE_ENERGY"), 1),
      ParticleProp(Sym<REAL>("ION2_SOURCE_DENSITY"), 1),
      ParticleProp(Sym<REAL>("ION2_SOURCE_MOMENTUM"), ndim),
      ParticleProp(Sym<REAL>("ION2_SOURCE_ENERGY"), 1),
      ParticleProp(Sym<REAL>("FLUID_DENSITY"), 1),
      ParticleProp(Sym<REAL>("FLUID_TEMPERATURE"), 1)};
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
      initial_distribution[Sym<REAL>("POSITION")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("VELOCITY")][px][dimx] =
          velocities.at(dimx).at(px);
      initial_distribution[Sym<REAL>("ELECTRON_SOURCE_MOMENTUM")][px][dimx] =
          0.0;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    initial_distribution[Sym<REAL>("TOT_REACTION_RATE")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("WEIGHT")][px][0] = 1.0;
    initial_distribution[Sym<INT>("INTERNAL_STATE")][px][0] = 0;
    initial_distribution[Sym<REAL>("ELECTRON_TEMPERATURE")][px][0] = 2.0;
    initial_distribution[Sym<REAL>("ELECTRON_DENSITY")][px][0] = 3.0e18;
    initial_distribution[Sym<REAL>("ELECTRON_SOURCE_ENERGY")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("ELECTRON_SOURCE_DENSITY")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("FLUID_DENSITY")][px][0] = 3.0e18;
    initial_distribution[Sym<REAL>("FLUID_TEMPERATURE")][px][0] = 2.0;
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
