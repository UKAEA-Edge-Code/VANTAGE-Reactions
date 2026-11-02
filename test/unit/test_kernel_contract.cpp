#include "include/test_common.hpp"
#include "reactions_lib/reaction_kernels.hpp"
#include <cstdlib>
#include <stdexcept>

using namespace VANTAGE::Reactions;

namespace {
struct SingleDevice : ReactionKernelsBaseOnDevice<1> {
  SingleDevice() = default;
  SingleDevice(REAL mass) : mass(mass) {}
  int weight, descendant_weight;
  REAL mass;
};
struct ZeroProductDevice : ReactionKernelsBaseOnDevice<0> {
  int weight;
};
struct WrongAccess : SingleDevice {
  void feedback_kernel(REAL &, int &, const std::array<int, 1> &, double) const;
};
struct WrongReturn : SingleDevice {
  int weight_kernel(REAL &, SingleReactionKernelAccessors &,
                    const std::array<int, 1> &, double) const;
};
struct WrongConst : SingleDevice {
  void scattering_kernel(REAL &, SingleReactionKernelAccessors &,
                         const std::array<int, 1> &, double);
};
struct NegativeCountDevice {
  static constexpr int NUM_PRODUCTS = -1;
};

struct SingleHost : ReactionKernelsBase<SingleDevice> {
  SingleHost(Properties<REAL> fields, PropertiesMap map, INT pre_req_ndims = 1)
      : ReactionKernelsBase(fields, {}, fields, pre_req_ndims, 3,
                            std::move(map)) {
    this->on_device_obj = SingleDevice(7.0);
    this->index_on_device_object();
  }

  void index_on_device_object() override {
    const auto &name = this->get_properties_map().at(default_properties.weight);
    this->on_device_obj->weight =
        this->argument_pack.required_real_props.find_index(name);
    this->on_device_obj->descendant_weight =
        this->argument_pack.required_descendant_real_props.find_index(name);
  }
};

struct UninitializedHost : ReactionKernelsBase<ZeroProductDevice> {
  UninitializedHost(INT pre_req_ndims = 0)
      : ReactionKernelsBase(Properties<REAL>({default_properties.weight}),
                            pre_req_ndims) {}
};
} // namespace

TEST(KernelContract, ExactSignaturesIncludingInheritedOperations) {
  static_assert(SingleHost::validate_on_device_type());
  static_assert(is_single_reaction_kernel<SingleHost>::value);
  static_assert(!is_single_reaction_kernel<int>::value);
  static_assert(!is_reaction_kernel_device_v<WrongAccess,
                                             SingleReactionKernelArgumentPack>);
  static_assert(!is_reaction_kernel_device_v<WrongReturn,
                                             SingleReactionKernelArgumentPack>);
  static_assert(!is_reaction_kernel_device_v<WrongConst,
                                             SingleReactionKernelArgumentPack>);
  static_assert(!is_reaction_kernel_device_v<NegativeCountDevice,
                                             SingleReactionKernelArgumentPack>);
}

TEST(KernelContract, ConstructionIndexesMappedNamesWithoutLosingMass) {
  auto map = PropertiesMap();
  map.at(default_properties.weight) = "A_WEIGHT";
  map.at(default_properties.velocity) = "Z_VELOCITY";
  Properties<REAL> fields(
      {default_properties.velocity, default_properties.weight});
  SingleHost host(fields, map);
  const auto &device = host.get_on_device_obj();
  EXPECT_EQ(device.weight, 0);
  EXPECT_EQ(device.mass, 7.0);
  EXPECT_EQ(host.get_descendant_matrix_spec()->get_sym_index(
                NP::Sym<REAL>("A_WEIGHT")),
            device.descendant_weight);
  auto schema = host.get_descendant_matrix_spec();
  schema->set_default_value(NP::Sym<REAL>("A_WEIGHT"), 0, 5.0);
  EXPECT_TRUE(host.get_descendant_matrix_spec()->default_values_real.empty());
}

TEST(KernelContract, InvalidPreReqNdimsAndUninitializedDeviceAssert) {
  UninitializedHost host;
  EXPECT_EQ(host.get_num_products(), 0);
  EXPECT_EQ(host.get_descendant_matrix_spec()->num_properties_int, 0);
  EXPECT_EQ(host.get_descendant_matrix_spec()->num_properties_real, 0);
  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW((void)host.get_on_device_obj(), std::logic_error);
    EXPECT_THROW((UninitializedHost(-1)), std::logic_error);
  }
}
