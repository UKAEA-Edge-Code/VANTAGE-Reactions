#include "include/test_common.hpp"
#include "reactions_lib/reaction_kernels_abstract.hpp"
#include <limits>
#include <stdexcept>

using namespace VANTAGE::Reactions;

namespace {
struct SingleDevice : SingleReactionKernelOnDevice<1> {
  int weight, descendant_weight;
  REAL mass;
};
struct PairDevice : PairReactionKernelOnDevice<2> {
  int weight_a, weight_b, descendant_weight_a, descendant_weight_b;
};
struct WrongAccess : SingleDevice {
  void feedback_kernel(REAL &, PairReactionKernelAccessors &,
                       const std::array<int, 1> &, double) const;
};
struct WrongReturn : SingleDevice {
  int weight_kernel(REAL &, SingleReactionKernelAccessors &,
                    const std::array<int, 1> &, double) const;
};
struct WrongConst : SingleDevice {
  void scattering_kernel(REAL &, SingleReactionKernelAccessors &,
                         const std::array<int, 1> &, double);
};
struct WrongParent : PairDevice {
  void parent_kernel(PairReactionKernelAccessors &,
                     const std::array<int, 1> &) const;
};

struct SingleHost : SingleReactionKernelBase<SingleDevice> {
  SingleHost(SingleReactionKernelArgumentPack pack, PropertiesMap map)
      : SingleReactionKernelBase(std::move(pack), 1, 3, std::move(map)) {
    this->on_device_obj.emplace();
    this->on_device_obj->mass = 7.0;
    this->index_on_device_object();
  }

protected:
  void index_on_device_object() override {
    auto pack = this->get_arg_pack();
    const auto &name = this->get_properties_map().at(default_properties.weight);
    this->on_device_obj->weight =
        pack.inputs.required_real_props.find_index(name);
    this->on_device_obj->descendant_weight =
        pack.descendants.required_real_props.find_index(name);
  }
};
struct PairHost : PairReactionKernelBase<PairDevice> {
  PairHost(PairReactionKernelArgumentPack pack, INT a = 1, INT b = 1)
      : PairReactionKernelBase(std::move(pack), 2, 3, a, b) {
    this->on_device_obj.emplace();
    this->index_on_device_object();
  }

protected:
  void index_on_device_object() override {
    auto pack = this->get_arg_pack();
    this->on_device_obj->weight_a =
        pack.inputs_a.required_real_props.find_index("WEIGHT");
    this->on_device_obj->weight_b =
        pack.inputs_b.required_real_props.find_index("WEIGHT");
    this->on_device_obj->descendant_weight_a =
        pack.descendants_a.required_real_props.find_index("WEIGHT");
    this->on_device_obj->descendant_weight_b =
        pack.descendants_b.required_real_props.find_index("WEIGHT");
  }
};
} // namespace

TEST(KernelContract, ExactSignaturesIncludingInheritedOperations) {
  static_assert(SingleHost::validate_on_device_type());
  static_assert(PairHost::validate_on_device_type());
  static_assert(!is_reaction_kernel_device_v<WrongAccess,
                                             SingleReactionKernelArgumentPack>);
  static_assert(!is_reaction_kernel_device_v<WrongReturn,
                                             SingleReactionKernelArgumentPack>);
  static_assert(!is_reaction_kernel_device_v<WrongConst,
                                             SingleReactionKernelArgumentPack>);
  static_assert(!is_reaction_kernel_device_v<WrongParent,
                                             PairReactionKernelArgumentPack>);
  static_assert(!is_reaction_kernel_device_v<SingleDevice,
                                             PairReactionKernelArgumentPack>);
}

TEST(KernelContract, ConstructionIndexesMappedNamesWithoutLosingMass) {
  auto map = PropertiesMap();
  map.at(default_properties.weight) = "A_WEIGHT";
  map.at(default_properties.velocity) = "Z_VELOCITY";
  Properties<REAL> fields(
      {default_properties.velocity, default_properties.weight});
  SingleHost host({KernelInputDescription({}, fields, {}, {}, map),
                   KernelPropertyDescription({}, fields, map)},
                  map);
  const auto &device = host.get_on_device_obj();
  EXPECT_EQ(device.weight, 0);
  EXPECT_EQ(device.mass, 7.0);
  EXPECT_EQ(host.get_descendant_matrix_spec()->get_sym_index(
                NP::Sym<REAL>("A_WEIGHT")),
            device.descendant_weight);
}

TEST(KernelContract, PairIndicesAndSchemasRemainIndependent) {
  KernelPropertyDescription a({},
                              Properties<REAL>({default_properties.weight}));
  KernelPropertyDescription b({},
                              Properties<REAL>({default_properties.velocity,
                                                default_properties.weight}));
  PairHost host({a, b, a, b});
  const auto &device = host.get_on_device_obj();
  EXPECT_EQ(device.weight_a, 0);
  EXPECT_EQ(device.weight_b, 1);
  EXPECT_EQ(host.get_descendant_matrix_spec_a()->get_sym_index(
                NP::Sym<REAL>("WEIGHT")),
            device.descendant_weight_a);
  EXPECT_EQ(host.get_descendant_matrix_spec_b()->get_sym_index(
                NP::Sym<REAL>("WEIGHT")),
            device.descendant_weight_b);
  auto schema = host.get_descendant_matrix_spec_b();
  schema->set_default_value(NP::Sym<REAL>("WEIGHT"), 0, 5.0);
  EXPECT_TRUE(host.get_descendant_matrix_spec_b()->default_values_real.empty());
}

TEST(KernelContract, InvalidPairCountsThrowBeforeIndexingOrSchemaPreparation) {
  for (const auto &counts :
       {std::pair<INT, INT>{-1, 3},
        {2, -1},
        {1, 2},
        {std::numeric_limits<INT>::max(), std::numeric_limits<INT>::max()}}) {
    EXPECT_THROW(PairHost({}, counts.first, counts.second),
                 std::invalid_argument);
  }
}
