#ifndef MOCK_REACTIONS_H
#define MOCK_REACTIONS_H
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
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
    props.velocity, props.weight
  };

  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
    props.internal_state
  };

  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
    props.velocity, props.weight
  };

  TestReactionKernels(std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(
            Properties<REAL>(required_simple_real_props),
            0, properties_map) {

    this->test_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity,
                                                    this->properties_map);
    this->test_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);

    this->set_required_descendant_int_props(Properties<INT>(
        required_descendant_simple_int_props));

    this->set_required_descendant_real_props(Properties<REAL>(
        required_descendant_simple_real_props));

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
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
    props.velocity, props.weight
  };

  constexpr static std::array<int, 2> required_species_real_props = {
    props.source_density, props.source_energy
  };

  TestReactionDataCalcKernels()
      : ReactionKernelsBase(
            Properties<REAL>(
                required_simple_real_props,
                std::vector<Species>{Species("ELECTRON")},
                required_species_real_props),
            2) {

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
               const std::array<int, num_products_per_parent> out_states_)
      : LinearReactionBase<num_products_per_parent, TestReactionData,
                           TestReactionKernels<num_products_per_parent>>(
            sycl_target_, in_states_, out_states_, TestReactionData(rate_),
            TestReactionKernels<num_products_per_parent>()) {}
};

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
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {props.position};

  TestReactionVarData()
      : ReactionDataBase(
            Properties<REAL>(required_simple_real_props,
                             std::vector<Species>{}, std::array<int, 0>{})) {

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

struct TestReactionVarKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<0> {
  void feedback_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, 0> &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    req_real_props.at(weight_ind, index, 0) -= modified_weight;
  }

public:
  int weight_ind;
};

struct TestReactionVarKernels : public ReactionKernelsBase {
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {props.weight};

  TestReactionVarKernels()
      : ReactionKernelsBase(Properties<REAL>(
            required_simple_real_props,
            std::vector<Species>{}, std::array<int, 0>{})) {

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

  TestReactionVarRate(SYCLTargetSharedPtr sycl_target_, int in_states_)
      : LinearReactionBase<0, TestReactionVarData, TestReactionVarKernels>(
            sycl_target_, in_states_, std::array<int, 0>{},
            TestReactionVarData(), TestReactionVarKernels()) {}
};

struct TestEphemeralVarDataOnDevice : public ReactionDataBaseOnDevice<> {
  TestEphemeralVarDataOnDevice() = default;

  std::array<REAL, 1>
  calc_data(Access::LoopIndex::Read &index,
            Access::SymVector::Write<INT> req_int_props,
            Access::SymVector::Read<REAL> req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<REAL, 1>{
        req_real_props.at_ephemeral(point_ind, index, 0) *
        req_real_props.at_ephemeral(normal_ind, index, 0)};
  }

public:
  int normal_ind, point_ind;
};
// TODO: Add corresponding kernel for ephemeral dat test
struct TestEphemeralVarData : public ReactionDataBase<> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};

  constexpr static std::array<int, 2> required_simple_real_props_ephemeral = {
      props.boundary_intersection_point, props.boundary_intersection_normal};

  TestEphemeralVarData(std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(Properties<INT>(),
                         Properties<REAL>(required_simple_real_props),
                         Properties<INT>(),
                         Properties<REAL>(required_simple_real_props_ephemeral),
                         properties_map) {

    this->test_reaction_var_data_on_device.point_ind =
        this->required_real_props.simple_prop_index(
            props.boundary_intersection_point);

    this->test_reaction_var_data_on_device.normal_ind =
        this->required_real_props.simple_prop_index(
            props.boundary_intersection_normal);
  }

private:
  TestEphemeralVarDataOnDevice test_reaction_var_data_on_device;

public:
  TestEphemeralVarDataOnDevice get_on_device_obj() {
    return this->test_reaction_var_data_on_device;
  }
};
#endif