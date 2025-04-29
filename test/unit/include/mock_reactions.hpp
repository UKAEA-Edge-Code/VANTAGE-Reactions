#pragma once
#include <neso_particles.hpp>
#include <reactions.hpp>

using namespace NESO::Particles;
using namespace Reactions;

struct TestReactionDataOnDevice : public ReactionDataBaseOnDevice<> {
  TestReactionDataOnDevice(REAL rate_) : rate(rate_){};

  std::array<REAL, 1>
  calc_data(Access::LoopIndex::Read &index,
            Access::SymVector::Write<INT> &req_int_props,
            Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<REAL, 1>{this->rate};
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
const auto props = default_properties;
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
  TestReactionKernels(std::map<int, std::string> properties_map_ = default_map)
      : ReactionKernelsBase(
            Properties<REAL>(TEST_REACTION_KERNEL::required_simple_real_props),
            0, properties_map_) {
    auto props = TEST_REACTION_KERNEL::props;

    this->test_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity,
                                                    this->properties_map);
    this->test_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);

    this->set_required_descendant_int_props(Properties<INT>(
        TEST_REACTION_KERNEL::required_descendant_simple_int_props));

    this->set_required_descendant_real_props(Properties<REAL>(
        TEST_REACTION_KERNEL::required_descendant_simple_real_props));

    this->test_reaction_kernels_on_device.descendant_internal_state_ind =
        this->required_descendant_int_props.simple_prop_index(
            props.internal_state, this->properties_map);
    this->test_reaction_kernels_on_device.descendant_velocity_ind =
        this->required_descendant_real_props.simple_prop_index(
            props.velocity, this->properties_map);
    this->test_reaction_kernels_on_device.descendant_weight_ind =
        this->required_descendant_real_props.simple_prop_index(
            props.weight, this->properties_map);

    this->set_descendant_matrix_spec<2, num_products_per_parent>();
  }

private:
  TestReactionKernelsOnDevice<num_products_per_parent>
      test_reaction_kernels_on_device;

public:
  TestReactionKernelsOnDevice<num_products_per_parent> get_on_device_obj() {
    return this->test_reaction_kernels_on_device;
  }
};

namespace TEST_REACTION_KERNEL_DATA_CALC {
const auto props = default_properties;
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

template <INT num_products_per_parent>
struct TestReaction
    : public LinearReactionBase<num_products_per_parent, TestReactionData,
                                TestReactionKernels<num_products_per_parent>> {

  TestReaction() = default;

  TestReaction(SYCLTargetSharedPtr sycl_target_, REAL rate_, int in_states_,
               const std::array<int, num_products_per_parent> out_states_,
               const ParticleSpec &particle_spec)
      : LinearReactionBase<num_products_per_parent, TestReactionData,
                           TestReactionKernels<num_products_per_parent>>(
            sycl_target_, in_states_, out_states_, TestReactionData(rate_),
            TestReactionKernels<num_products_per_parent>(), particle_spec) {}
};

namespace TEST_REACTION_VAR_DATA {
const auto props = default_properties;
const std::vector<int> required_simple_real_props = {props.position};
} // namespace TEST_REACTION_VAR_DATA

struct TestReactionVarDataOnDevice : public ReactionDataBaseOnDevice<> {
  TestReactionVarDataOnDevice() = default;

  std::array<REAL, 1>
  calc_data(Access::LoopIndex::Read &index,
            Access::SymVector::Write<INT> req_int_props,
            Access::SymVector::Read<REAL> req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<REAL, 1>{req_real_props.at(position_ind, index, 0)};
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

const auto props = default_properties;

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

  TestReactionVarRate(SYCLTargetSharedPtr sycl_target_, int in_states_,
                      const ParticleSpec &particle_spec)
      : LinearReactionBase<0, TestReactionVarData, TestReactionVarKernels>(
            sycl_target_, in_states_, std::array<int, 0>{},
            TestReactionVarData(), TestReactionVarKernels(), particle_spec) {}
};
