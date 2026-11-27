#include "include/mock_particle_group.hpp"
#include "include/test_common.hpp"
#include "reactions_lib/pair_reaction_kernels.hpp"
#include "reactions_lib/reaction_kernels.hpp"
#include <cstdlib>
#include <limits>
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
struct PairDevice : PairReactionKernelsBaseOnDevice<2> {
  int weight_a, weight_b, descendant_weight_a, descendant_weight_b;
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
struct WrongParent : PairDevice {
  void parent_kernel(PairReactionKernelAccessors &,
                     const std::array<int, 1> &) const;
};
struct MissingParent
    : AbstractReactionKernelsOnDevice<PairReactionKernelAccessors, 2> {};
struct NonConstParent : PairDevice {
  void parent_kernel(PairReactionKernelAccessors &, const std::array<int, 2> &);
};
struct NegativeCountDevice {
  static constexpr int NUM_PRODUCTS = -1;
};

struct SingleHost : ReactionKernelsBase<SingleDevice> {
  SingleHost(Properties<INT> req_int, Properties<REAL> req_real,
             Properties<INT> eph_int, Properties<REAL> eph_real,
             Properties<INT> desc_int, Properties<REAL> desc_real,
             PropertiesMap map, INT pre_req_ndims = 1, int ndim_velocity = 3)
      : ReactionKernelsBase(req_int, req_real, eph_int, eph_real, desc_int,
                            desc_real, pre_req_ndims, ndim_velocity, map) {
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

struct PairHost : PairReactionKernelsBase<PairDevice> {
  PairHost(Properties<REAL> req_real_a = {}, Properties<REAL> req_real_b = {},
           Properties<REAL> desc_real_a = {}, Properties<REAL> desc_real_b = {},
           INT a = 1, INT b = 1, PropertiesMap map = PropertiesMap(),
           int ndim_velocity = 3)
      : PairReactionKernelsBase(
            {}, std::move(req_real_a), {}, std::move(req_real_b), {},
            std::move(desc_real_a), {}, std::move(desc_real_b), 2,
            ndim_velocity, a, b, std::move(map)) {
    this->on_device_obj = PairDevice{};
    this->index_on_device_object();
  }

  void index_on_device_object() override {
    const auto &name = this->get_properties_map().at(default_properties.weight);
    this->on_device_obj->weight_a =
        this->argument_pack.required_real_props_a.find_index(name);
    this->on_device_obj->weight_b =
        this->argument_pack.required_real_props_b.find_index(name);
    this->on_device_obj->descendant_weight_a =
        this->argument_pack.required_descendant_real_props_a.find_index(name);
    this->on_device_obj->descendant_weight_b =
        this->argument_pack.required_descendant_real_props_b.find_index(name);
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
  static_assert(PairHost::validate_on_device_type());
  static_assert(is_single_reaction_kernel<SingleHost>::value);
  static_assert(!is_single_reaction_kernel<PairHost>::value);
  static_assert(!is_single_reaction_kernel<int>::value);
  static_assert(is_pair_reaction_kernel<PairHost>::value);
  static_assert(!is_pair_reaction_kernel<SingleHost>::value);
  static_assert(!is_pair_reaction_kernel<int>::value);
  static_assert(is_single_reaction_kernel_device_v<SingleDevice>);
  static_assert(is_pair_reaction_kernel_device_v<PairDevice>);
  static_assert(!is_single_reaction_kernel_device_v<WrongAccess>);
  static_assert(!is_single_reaction_kernel_device_v<WrongReturn>);
  static_assert(!is_single_reaction_kernel_device_v<WrongConst>);
  static_assert(!is_single_reaction_kernel_device_v<PairDevice>);
  static_assert(!is_pair_reaction_kernel_device_v<WrongParent>);
  static_assert(!is_pair_reaction_kernel_device_v<MissingParent>);
  static_assert(!is_pair_reaction_kernel_device_v<NonConstParent>);
  static_assert(!is_pair_reaction_kernel_device_v<SingleDevice>);
  static_assert(!is_single_reaction_kernel_device_v<NegativeCountDevice>);
  static_assert(!is_pair_reaction_kernel_device_v<NegativeCountDevice>);
}

TEST(KernelContract, PropertiesConstructorsEnforceMapAndCounts) {
  using Single = ReactionKernelsBase<SingleDevice>;
  using Pair = PairReactionKernelsBase<PairDevice>;
  using Map = PropertiesMap;
  static_assert(
      !std::is_constructible_v<Single, SingleReactionKernelArgumentPack>);
  static_assert(
      !std::is_constructible_v<Single, SingleReactionKernelArgumentPack, INT,
                               int, Map>);
  static_assert(!std::is_constructible_v<Pair, PairReactionKernelArgumentPack>);
  static_assert(!std::is_constructible_v<Pair, PairReactionKernelArgumentPack,
                                         INT, int, INT, INT, Map>);

  PairReactionKernelsBase<PairReactionKernelsBaseOnDevice<0>> zero;
  EXPECT_EQ(zero.get_num_products_a(), 0);
  EXPECT_EQ(zero.get_num_products_b(), 0);
  EXPECT_EQ(zero.get_descendant_matrix_spec_a()->num_properties_real, 0);
  EXPECT_EQ(zero.get_descendant_matrix_spec_b()->num_properties_real, 0);

  auto map = PropertiesMap();
  map.at(default_properties.velocity) = "CUSTOM_VELOCITY";
  Properties<REAL> velocity({default_properties.velocity});
  for (const int ndim : {2, 3}) {
    Single single({}, {}, {}, {}, {}, velocity, 0, ndim, map);
    Pair pair(Properties<REAL>{}, {}, velocity, 0, ndim, 1, 1, map);
    for (const auto &schema : {single.get_descendant_matrix_spec(),
                               pair.get_descendant_matrix_spec_a(),
                               pair.get_descendant_matrix_spec_b()}) {
      EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("CUSTOM_VELOCITY")),
                ndim);
      EXPECT_EQ(schema->get_sym_index(NP::Sym<REAL>("VELOCITY")), -1);
    }
  }
}

TEST(KernelContract, SinglePackEphemeralMergeAndSchema) {
  const auto props = default_properties;
  const Species ion("ION");
  auto map = PropertiesMap();
  map.at(props.weight) = "A_WEIGHT";
  map.at(props.velocity) = "Z_VELOCITY";
  map.at(props.source_momentum) = "MOMENTUM";

  SingleHost host(Properties<INT>({props.id}), Properties<REAL>({props.weight}),
                  Properties<INT>({props.id, props.panic}),
                  Properties<REAL>({ion}, {props.source_momentum}),
                  Properties<INT>({props.internal_state}),
                  Properties<REAL>({props.velocity, props.weight}), map, 1, 3);

  const auto &pack = host.get_arg_pack();
  EXPECT_EQ(pack.required_int_props.to_string_vector(),
            (std::vector<std::string>{"ID", "REACTIONS_PANIC_FLAG"}));
  EXPECT_EQ(pack.required_real_props.to_string_vector(),
            (std::vector<std::string>{"A_WEIGHT",
                                      species_property(ion, "MOMENTUM")}));

  const auto &device = host.get_on_device_obj();
  EXPECT_EQ(device.weight, 0);
  EXPECT_EQ(device.mass, 7.0);

  auto schema = host.get_descendant_matrix_spec();
  EXPECT_EQ(schema->get_sym_index(NP::Sym<REAL>("A_WEIGHT")),
            device.descendant_weight);
  EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("A_WEIGHT")), 1);
  EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("Z_VELOCITY")), 3);
  EXPECT_EQ(schema->get_num_components(NP::Sym<INT>("INTERNAL_STATE")), 1);
  schema->set_default_value(NP::Sym<REAL>("A_WEIGHT"), 0, 5.0);
  EXPECT_TRUE(host.get_descendant_matrix_spec()->default_values_real.empty());
}

TEST(KernelContract, PairIndicesAndSchemasRemainIndependent) {
  const auto props = default_properties;
  PairHost host(Properties<REAL>({props.weight}),
                Properties<REAL>({props.velocity, props.weight}),
                Properties<REAL>({props.weight}),
                Properties<REAL>({props.velocity, props.weight}));
  const auto &device = host.get_on_device_obj();
  EXPECT_EQ(device.weight_a, 0);
  EXPECT_EQ(device.weight_b, 1);
  EXPECT_EQ(host.get_descendant_matrix_spec_a()->get_sym_index(
                NP::Sym<REAL>("WEIGHT")),
            device.descendant_weight_a);
  EXPECT_EQ(host.get_descendant_matrix_spec_b()->get_sym_index(
                NP::Sym<REAL>("WEIGHT")),
            device.descendant_weight_b);
  EXPECT_EQ(host.get_descendant_matrix_spec_b()->get_num_components(
                NP::Sym<REAL>("VELOCITY")),
            3);
  auto schema = host.get_descendant_matrix_spec_b();
  schema->set_default_value(NP::Sym<REAL>("WEIGHT"), 0, 5.0);
  EXPECT_TRUE(host.get_descendant_matrix_spec_b()->default_values_real.empty());
}

TEST(KernelContract, InvalidInputsAndUninitializedDeviceAssert) {
  UninitializedHost host;
  EXPECT_EQ(host.get_num_products(), 0);
  EXPECT_EQ(host.get_descendant_matrix_spec()->num_properties_int, 0);
  EXPECT_EQ(host.get_descendant_matrix_spec()->num_properties_real, 0);
  ArgumentNameSet<REAL> scalar(Properties<REAL>({default_properties.weight}));
  EXPECT_NO_THROW(build_kernel_product_schema({}, scalar, 0));

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW((void)host.get_on_device_obj(), std::logic_error);
    EXPECT_THROW((UninitializedHost(-1)), std::logic_error);
    ArgumentNameSet<REAL> vel(Properties<REAL>({default_properties.velocity}));
    EXPECT_THROW(build_kernel_product_schema({}, vel, 0), std::logic_error);
    for (const auto &counts :
         {std::pair<INT, INT>{-1, 3},
          {2, -1},
          {1, 2},
          {std::numeric_limits<INT>::max(), std::numeric_limits<INT>::max()}}) {
      EXPECT_THROW((PairHost({}, {}, {}, {}, counts.first, counts.second)),
                   std::logic_error);
    }
  }
}

TEST(KernelContract, RemappedLinearAndPairKernelExecution) {
  // Reverse velocity/weight ordering relative to the default names.
  auto map = PropertiesMap();
  map.at(default_properties.weight) = "A_WEIGHT";
  map.at(default_properties.velocity) = "Z_VELOCITY";

  auto [A, B] = create_test_particle_groups_pairs(32);
  for (auto &g : {A, B}) {
    g->add_particle_dat(NP::Sym<REAL>("A_WEIGHT"), 1);
    g->add_particle_dat(NP::Sym<REAL>("Z_VELOCITY"), 2);
  }
  particle_loop(
      "init_A", A,
      [](auto w, auto v) {
        w.at(0) = 2.0;
        v.at(0) = 3.0;
        v.at(1) = -4.0;
      },
      NP::Access::write(NP::Sym<REAL>("A_WEIGHT")),
      NP::Access::write(NP::Sym<REAL>("Z_VELOCITY")))
      ->execute();
  particle_loop(
      "init_B", B,
      [](auto w, auto v) {
        w.at(0) = 3.0;
        v.at(0) = 1.0;
        v.at(1) = 2.0;
      },
      NP::Access::write(NP::Sym<REAL>("A_WEIGHT")),
      NP::Access::write(NP::Sym<REAL>("Z_VELOCITY")))
      ->execute();

  const int cells = A->domain->mesh->get_cell_count();
  auto subgroup = NP::particle_sub_group(A);
  const Species target("ION", 2.0, 0.0, 5), projectile("ION2", 3.0, 0.0, 0);
  auto linear_calc = DataCalculator(FixedRateData(-1.0), FixedRateData(2.0));
  LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                     decltype(linear_calc)>
      linear_reaction(A->sycl_target, 0, {5}, FixedRateData(1.0),
                      CXReactionKernels<2>(target, projectile, map),
                      linear_calc, map);
  auto linear_children = std::make_shared<NP::ParticleGroup>(
      A->domain, A->get_particle_spec(), A->sycl_target);
  linear_reaction.calculate_rates(subgroup, 0, cells);
  linear_reaction.apply(subgroup, 0, cells, 0.25, linear_children);

  auto cellwise_pair_list =
      std::make_shared<NP::CellwisePairListSimple>(A->sycl_target, cells);
  std::vector<int> c, i, j;
  for (int cell = 0; cell < cells; ++cell) {
    for (int px = 0; px < A->get_npart_cell(cell); ++px) {
      c.push_back(cell);
      i.push_back(px);
      j.push_back(px);
    }
  }
  cellwise_pair_list->push_back(c, i, j);
  auto pair_list =
      NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList>(
          A, B, cellwise_pair_list);

  auto cs_data = CSPairData<2>(ConstantRateCrossSection(0.5), map);
  auto pair_calc =
      PairDataCalculator(CSPairData<2>(ConstantRateCrossSection(-7.0), map),
                         CSPairData<2>(ConstantRateCrossSection(8.0), map),
                         CSPairData<2>(ConstantRateCrossSection(9.0), map),
                         CSPairData<2>(ConstantRateCrossSection(-10.0), map));
  PairScatteringKernels<2> pair_kernels(map);
  EXPECT_EQ(pair_kernels.get_on_device_obj().descendant_weight_ind_a, 0);
  EXPECT_EQ(pair_kernels.get_on_device_obj().descendant_velocity_ind_a, 1);

  SWPMReaction<2, decltype(cs_data), PairScatteringKernels<2>,
               decltype(pair_calc)>
      pair_reaction(A->sycl_target, {0, 0}, {11, 22}, cs_data, pair_kernels,
                    pair_calc, map);
  particle_loop(
      "reset_rate_and_set_wc", A,
      [](auto tr, auto wc) {
        tr.at(0) = 0.0;
        wc.at(0) = 0.5;
      },
      NP::Access::write(NP::Sym<REAL>("TOT_REACTION_RATE")),
      NP::Access::write(NP::Sym<REAL>("WEIGHT_CHANGE")))
      ->execute();
  pair_reaction.calculate_rates(pair_list, 0, cells);
  auto pair_children = std::make_shared<NP::ParticleGroup>(
      A->domain, A->get_particle_spec(), A->sycl_target);
  pair_reaction.apply(pair_list, 0, cells, 1.0, pair_children);

  for (int cell = 0; cell < cells; ++cell) {
    auto wa = A->get_cell(NP::Sym<REAL>("A_WEIGHT"), cell);
    auto wb = B->get_cell(NP::Sym<REAL>("A_WEIGHT"), cell);
    auto lw = linear_children->get_cell(NP::Sym<REAL>("A_WEIGHT"), cell);
    auto lv = linear_children->get_cell(NP::Sym<REAL>("Z_VELOCITY"), cell);
    for (int row = 0; row < wa->nrow; ++row) {
      EXPECT_DOUBLE_EQ(wa->at(row, 0), 1.25);
      EXPECT_DOUBLE_EQ(wb->at(row, 0), 2.5);
      EXPECT_DOUBLE_EQ(lw->at(row, 0), 0.25);
      EXPECT_DOUBLE_EQ(lv->at(row, 0), -1.0);
      EXPECT_DOUBLE_EQ(lv->at(row, 1), 2.0);
    }
    auto pw = pair_children->get_cell(NP::Sym<REAL>("A_WEIGHT"), cell);
    auto pv = pair_children->get_cell(NP::Sym<REAL>("Z_VELOCITY"), cell);
    auto ps = pair_children->get_cell(NP::Sym<INT>("INTERNAL_STATE"), cell);
    ASSERT_EQ(pw->nrow, 2 * wa->nrow);
    for (int row = 0; row < pw->nrow; ++row) {
      EXPECT_DOUBLE_EQ(pw->at(row, 0), 0.5);
      if (ps->at(row, 0) == 11) {
        EXPECT_DOUBLE_EQ(pv->at(row, 0), -7.0);
        EXPECT_DOUBLE_EQ(pv->at(row, 1), 8.0);
      } else {
        ASSERT_EQ(ps->at(row, 0), 22);
        EXPECT_DOUBLE_EQ(pv->at(row, 0), 9.0);
        EXPECT_DOUBLE_EQ(pv->at(row, 1), -10.0);
      }
    }
  }
  A->sycl_target->free();
  A->domain->mesh->free();
}
