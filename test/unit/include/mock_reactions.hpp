#ifndef REACTIONS_MOCK_REACTIONS_H
#define REACTIONS_MOCK_REACTIONS_H

#include "test_common.hpp"

using namespace VANTAGE::Reactions;

struct TestReactionDataOnDevice : public ReactionDataBaseOnDevice<> {

  TestReactionDataOnDevice() = default;
  TestReactionDataOnDevice(REAL rate) : rate(rate) {};

  std::array<REAL, 1>
  calc_data(const SingleReactionDataAccessors &accessors,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<REAL, 1>{this->rate};
  }

private:
  REAL rate;
};

struct TestReactionData : public ReactionDataBase<TestReactionDataOnDevice> {

  TestReactionData(REAL rate) : rate(rate) {
    this->on_device_obj = TestReactionDataOnDevice(rate);
  }

  void index_on_device_object() {}

private:
  REAL rate;
};

template <INT num_products_per_parent>
struct TestReactionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<num_products_per_parent> {
  TestReactionKernelsOnDevice() = default;

  void
  scattering_kernel(REAL &modified_weight,
                    SingleReactionKernelAccessors &accessors,
                    const std::array<int, num_products_per_parent> &out_states,
                    double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      for (int dimx = 0; dimx < 2; dimx++) {
        accessors.descendant_products.at_real(
            accessors.index, childx, this->descendant_velocity_ind, dimx) =
            accessors.req_real_props.at(this->velocity_ind, accessors.index,
                                        dimx);
      }
    }
  }

  void weight_kernel(REAL &modified_weight,
                     SingleReactionKernelAccessors &accessors,
                     const std::array<int, num_products_per_parent> &out_states,
                     double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      accessors.descendant_products.at_real(accessors.index, childx,
                                            this->descendant_weight_ind, 0) =
          (modified_weight / num_products_per_parent);
    }
  }

  void transformation_kernel(
      REAL &modified_weight, SingleReactionKernelAccessors &accessors,
      const std::array<int, num_products_per_parent> &out_states,
      double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      accessors.descendant_products.at_int(accessors.index, childx,
                                           this->descendant_internal_state_ind,
                                           0) = out_states[childx];
    }
  }

  void
  feedback_kernel(REAL &modified_weight,
                  SingleReactionKernelAccessors &accessors,
                  const std::array<int, num_products_per_parent> &out_states,
                  double dt) const {
    accessors.req_real_props.at(this->weight_ind, accessors.index, 0) -=
        modified_weight;
  }

public:
  int velocity_ind, weight_ind, descendant_velocity_ind, descendant_weight_ind,
      descendant_internal_state_ind;
};

template <INT num_products_per_parent>
struct TestReactionKernels
    : public ReactionKernelsBase<
          TestReactionKernelsOnDevice<num_products_per_parent>> {
  using Base =
      ReactionKernelsBase<TestReactionKernelsOnDevice<num_products_per_parent>>;
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.velocity, props.weight};

  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};

  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};

  TestReactionKernels(const PropertiesMap properties_map = PropertiesMap())
      : Base(Properties<REAL>(required_simple_real_props),
             (num_products_per_parent > 0)
                 ? Properties<INT>(required_descendant_simple_int_props)
                 : Properties<INT>(),
             (num_products_per_parent > 0)
                 ? Properties<REAL>(required_descendant_simple_real_props)
                 : Properties<REAL>(),
             0, 2, properties_map) {
    this->on_device_obj =
        TestReactionKernelsOnDevice<num_products_per_parent>();
    this->index_on_device_object();
  }

  void index_on_device_object() override {
    this->on_device_obj->velocity_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    this->on_device_obj->weight_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.weight));
    if constexpr (num_products_per_parent > 0) {
      this->on_device_obj->descendant_velocity_ind =
          this->argument_pack.required_descendant_real_props.find_index(
              this->get_properties_map().at(props.velocity));
      this->on_device_obj->descendant_weight_ind =
          this->argument_pack.required_descendant_real_props.find_index(
              this->get_properties_map().at(props.weight));
      this->on_device_obj->descendant_internal_state_ind =
          this->argument_pack.required_descendant_int_props.find_index(
              this->get_properties_map().at(props.internal_state));
    }
  }
};

template <INT num_products_per_parent>
struct TestReactionKernelsDataCalcOnDevice
    : public ReactionKernelsBaseOnDevice<num_products_per_parent> {
  TestReactionKernelsDataCalcOnDevice() = default;

  void
  scattering_kernel(REAL &modified_weight,
                    SingleReactionKernelAccessors &accessors,
                    const std::array<int, num_products_per_parent> &out_states,
                    double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      for (int dimx = 0; dimx < 2; dimx++) {
        accessors.descendant_products.at_real(
            accessors.index, childx, this->descendant_velocity_ind, dimx) =
            accessors.req_real_props.at(this->velocity_ind, accessors.index,
                                        dimx);
      }
    }
  }

  void weight_kernel(REAL &modified_weight,
                     SingleReactionKernelAccessors &accessors,
                     const std::array<int, num_products_per_parent> &out_states,
                     double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      accessors.descendant_products.at_real(accessors.index, childx,
                                            this->descendant_weight_ind, 0) =
          (modified_weight / num_products_per_parent);
    }
  }

  void transformation_kernel(
      REAL &modified_weight, SingleReactionKernelAccessors &accessors,
      const std::array<int, num_products_per_parent> &out_states,
      double dt) const {
    for (int childx = 0; childx < num_products_per_parent; childx++) {
      accessors.descendant_products.at_int(accessors.index, childx,
                                           this->descendant_internal_state_ind,
                                           0) = out_states[childx];
    }
  }

  void
  feedback_kernel(REAL &modified_weight,
                  SingleReactionKernelAccessors &accessors,
                  const std::array<int, num_products_per_parent> &out_states,
                  double dt) const {
    accessors.req_real_props.at(this->weight_ind, accessors.index, 0) -=
        modified_weight;
    accessors.req_real_props.at(this->source_ind, accessors.index, 0) +=
        accessors.pre_req_data.at(accessors.index.get_loop_linear_index(), 0);
    accessors.req_real_props.at(this->energy_source_ind, accessors.index, 0) +=
        accessors.pre_req_data.at(accessors.index.get_loop_linear_index(), 1);
  }

public:
  int velocity_ind, weight_ind, source_ind, energy_source_ind;
  int descendant_velocity_ind, descendant_weight_ind,
      descendant_internal_state_ind;
};

template <INT num_products_per_parent>
struct TestReactionDataCalcKernels
    : public ReactionKernelsBase<
          TestReactionKernelsDataCalcOnDevice<num_products_per_parent>> {
  using Base = ReactionKernelsBase<
      TestReactionKernelsDataCalcOnDevice<num_products_per_parent>>;
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.velocity, props.weight};

  constexpr static std::array<int, 2> required_species_real_props = {
      props.source_density, props.source_energy};

  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};

  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};

  TestReactionDataCalcKernels()
      : Base(Properties<REAL>(required_simple_real_props,
                              std::vector<Species>{Species("ELECTRON")},
                              required_species_real_props),
             (num_products_per_parent > 0)
                 ? Properties<INT>(required_descendant_simple_int_props)
                 : Properties<INT>(),
             (num_products_per_parent > 0)
                 ? Properties<REAL>(required_descendant_simple_real_props)
                 : Properties<REAL>(),
             2, 2) {
    this->on_device_obj =
        TestReactionKernelsDataCalcOnDevice<num_products_per_parent>();
    this->index_on_device_object();
  }

  void index_on_device_object() override {
    const Species electron("ELECTRON");
    this->on_device_obj->velocity_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    this->on_device_obj->weight_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.weight));
    this->on_device_obj->source_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            electron, this->get_properties_map().at(props.source_density)));
    this->on_device_obj->energy_source_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            electron, this->get_properties_map().at(props.source_energy)));
    if constexpr (num_products_per_parent > 0) {
      this->on_device_obj->descendant_velocity_ind =
          this->argument_pack.required_descendant_real_props.find_index(
              this->get_properties_map().at(props.velocity));
      this->on_device_obj->descendant_weight_ind =
          this->argument_pack.required_descendant_real_props.find_index(
              this->get_properties_map().at(props.weight));
      this->on_device_obj->descendant_internal_state_ind =
          this->argument_pack.required_descendant_int_props.find_index(
              this->get_properties_map().at(props.internal_state));
    }
  }
};

template <INT num_products_per_parent>
struct TestReaction
    : public LinearReactionBase<num_products_per_parent, TestReactionData,
                                TestReactionKernels<num_products_per_parent>> {

  TestReaction() = default;

  TestReaction(NP::SYCLTargetSharedPtr sycl_target, REAL rate, int in_states,
               const std::array<int, num_products_per_parent> out_states)
      : LinearReactionBase<num_products_per_parent, TestReactionData,
                           TestReactionKernels<num_products_per_parent>>(
            sycl_target, in_states, out_states, TestReactionData(rate),
            TestReactionKernels<num_products_per_parent>()) {}
};

struct TestReactionVarDataOnDevice : public ReactionDataBaseOnDevice<> {
  TestReactionVarDataOnDevice() = default;

  std::array<REAL, 1>
  calc_data(const SingleReactionDataAccessors &accessors,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<REAL, 1>{
        accessors.req_real_props.at(position_ind, accessors.index, 0)};
  }

public:
  int position_ind;
};

struct TestReactionVarData
    : public ReactionDataBase<TestReactionVarDataOnDevice> {
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.position};

  TestReactionVarData()
      : ReactionDataBase<TestReactionVarDataOnDevice>(
            Properties<REAL>(required_simple_real_props, std::vector<Species>{},
                             std::array<int, 0>{})) {
    this->on_device_obj = TestReactionVarDataOnDevice();

    this->index_on_device_object();
  };

  void index_on_device_object() {

    this->on_device_obj->position_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.position));
  };
};

struct TestReactionVarKernelsOnDevice : public ReactionKernelsBaseOnDevice<0> {
  void feedback_kernel(REAL &modified_weight,
                       SingleReactionKernelAccessors &accessors,
                       const std::array<int, 0> &out_states, double dt) const {
    accessors.req_real_props.at(this->weight_ind, accessors.index, 0) -=
        modified_weight;
  }

public:
  int weight_ind;
};

struct TestReactionVarKernels
    : public ReactionKernelsBase<TestReactionVarKernelsOnDevice> {
  using Base = ReactionKernelsBase<TestReactionVarKernelsOnDevice>;
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};

  TestReactionVarKernels()
      : Base(Properties<REAL>(required_simple_real_props)) {
    this->on_device_obj = TestReactionVarKernelsOnDevice();
    this->index_on_device_object();
  }

  void index_on_device_object() override {
    this->on_device_obj->weight_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.weight));
  }
};

struct TestReactionVarRate : public LinearReactionBase<0, TestReactionVarData,
                                                       TestReactionVarKernels> {

  TestReactionVarRate(NP::SYCLTargetSharedPtr sycl_target_, int in_states_)
      : LinearReactionBase<0, TestReactionVarData, TestReactionVarKernels>(
            sycl_target_, in_states_, std::array<int, 0>{},
            TestReactionVarData(), TestReactionVarKernels()) {}
};

struct TestEphemeralVarDataOnDevice : public ReactionDataBaseOnDevice<> {
  TestEphemeralVarDataOnDevice() = default;

  std::array<REAL, 1>
  calc_data(const SingleReactionDataAccessors &accessors,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    return std::array<REAL, 1>{
        accessors.req_real_props.at_ephemeral(point_ind, accessors.index, 0) *
        accessors.req_real_props.at_ephemeral(normal_ind, accessors.index, 0)};
  }

public:
  int normal_ind, point_ind;
};
struct TestEphemeralVarData
    : public ReactionDataBase<TestEphemeralVarDataOnDevice> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};

  constexpr static std::array<int, 2> required_simple_real_props_ephemeral = {
      props.boundary_intersection_point, props.boundary_intersection_normal};

  TestEphemeralVarData(const PropertiesMap properties_map = PropertiesMap())
      : ReactionDataBase<TestEphemeralVarDataOnDevice>(
            Properties<INT>(), Properties<REAL>(required_simple_real_props),
            Properties<INT>(),
            Properties<REAL>(required_simple_real_props_ephemeral),
            properties_map) {

    this->on_device_obj = TestEphemeralVarDataOnDevice();
    this->index_on_device_object();
  }

  void index_on_device_object() {

    this->on_device_obj->point_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.boundary_intersection_point));

    this->on_device_obj->normal_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.boundary_intersection_normal));
  };
};
#endif
