#include "../include/mock_particle_group.hpp"
#include "../include/test_common.hpp"

using namespace VANTAGE::Reactions;

namespace {
using TestRNG = DEFAULT_RNG_KERNEL::KernelType;

template <typename ACCESSORS, typename RESULT = std::array<REAL, 1>,
          typename RNG = TestRNG>
struct SignatureDevice : AbstractReactionDataOnDevice<ACCESSORS> {
  RESULT calc_data(const ACCESSORS &, RNG &) const;
};

using SingleDevice = SignatureDevice<SingleReactionDataAccessors>;
using PairDevice = SignatureDevice<PairReactionDataAccessors>;
using WrongReturnDevice = SignatureDevice<SingleReactionDataAccessors, REAL>;
using WrongSizeDevice =
    SignatureDevice<SingleReactionDataAccessors, std::array<REAL, 2>>;
using WrongRNGDevice =
    SignatureDevice<SingleReactionDataAccessors, std::array<REAL, 1>, int>;

struct InputDevice : ReactionDataBaseOnDevice<1, DEFAULT_RNG_KERNEL, 1> {
  std::array<REAL, 1> calc_data(const std::array<REAL, 1> &,
                                const SingleReactionDataAccessors &,
                                TestRNG &) const;
};

struct WrongInputReturnDevice : InputDevice {
  REAL calc_data(const std::array<REAL, 1> &,
                 const SingleReactionDataAccessors &, TestRNG &) const;
};

struct MutableDevice : ReactionDataBaseOnDevice<> {
  std::array<REAL, 1> calc_data(const SingleReactionDataAccessors &, TestRNG &);
};

using SingleData = ReactionDataBase<SingleDevice>;
using PairData = PairReactionDataBase<PairDevice>;
using InputData = ReactionDataBase<InputDevice, 1, DEFAULT_RNG_KERNEL, 1>;
using MultiData = ReactionDataBase<ReactionDataBaseOnDevice<2>, 2>;
} // namespace

TEST(ReactionDataContract, DeviceSignatures) {
  static_assert(SingleData::validate_on_device_type());
  static_assert(PairData::validate_on_device_type());
  static_assert(InputData::validate_on_device_type());
  static_assert(is_abstract_calc_data_callable_v<
                SingleDevice, const SingleReactionDataAccessors &, TestRNG &>);
  static_assert(!is_abstract_calc_data_callable_v<
                PairDevice, const SingleReactionDataAccessors &, TestRNG &>);
  static_assert(
      !is_abstract_calc_data_callable_v<
          WrongRNGDevice, const SingleReactionDataAccessors &, TestRNG &>);
  static_assert(!is_abstract_calc_data_callable_v<
                MutableDevice, const SingleReactionDataAccessors &, TestRNG &>);
  static_assert(!is_abstract_calc_data_callable_v<
                InputDevice, const SingleReactionDataAccessors &, TestRNG &>);
  static_assert(!check_abstract_calc_data_return_type<
                WrongReturnDevice, std::array<REAL, 1>,
                const SingleReactionDataAccessors &, TestRNG &>());
  static_assert(!check_abstract_calc_data_return_type<
                WrongSizeDevice, std::array<REAL, 1>,
                const SingleReactionDataAccessors &, TestRNG &>());
  static_assert(!check_abstract_calc_data_return_type<
                WrongInputReturnDevice, std::array<REAL, 1>,
                const std::array<REAL, 1> &,
                const SingleReactionDataAccessors &, TestRNG &>());
}

TEST(ReactionDataContract, LoopCompatibility) {
  static_assert(
      is_loop_reaction_data<SingleData, SingleReactionDataArgumentPack>::value);
  static_assert(
      is_loop_reaction_data<PairData, PairReactionDataArgumentPack>::value);
  static_assert(
      !is_loop_reaction_data<PairData, SingleReactionDataArgumentPack>::value);
  static_assert(
      !is_loop_reaction_data<SingleData, PairReactionDataArgumentPack>::value);
  static_assert(
      !is_loop_reaction_data<int, SingleReactionDataArgumentPack>::value);
  static_assert(
      !is_loop_reaction_data<InputData, SingleReactionDataArgumentPack>::value);
  static_assert(!is_loop_reaction_data<ReactionDataBase<WrongReturnDevice>,
                                    SingleReactionDataArgumentPack>::value);
  static_assert(!is_loop_reaction_data<ReactionDataBase<WrongSizeDevice>,
                                    SingleReactionDataArgumentPack>::value);
  static_assert(is_loop_reaction_data<ConcatenatorData<SingleData, SingleData>,
                                   SingleReactionDataArgumentPack>::value);
  static_assert(is_loop_reaction_data<ConcatenatorData<PairData, PairData>,
                                   PairReactionDataArgumentPack>::value);
  // Calculator prerequisites are arrays, not scalar reaction rates.
  auto single = DataCalculator(MultiData{});
  auto pair = PairDataCalculator(PairData{});
  EXPECT_EQ(single.get_data_size(), 2);
  EXPECT_EQ(pair.get_data_size(), 1);
  EXPECT_EQ(DataCalculator<>().get_data_size(), 0);
  EXPECT_EQ(PairDataCalculator<>().get_data_size(), 0);
}

TEST(ReactionDataArgumentPack, IntegerOnlyConstruction) {
  ReactionDataBase<FixedRateDataOnDevice> data(
      Properties<INT>(std::array<int, 1>{default_properties.panic}));

  auto pack = data.get_arg_pack();
  EXPECT_EQ(pack.required_int_props.to_string_vector(),
            (std::vector<std::string>{"REACTIONS_PANIC_FLAG"}));
  EXPECT_TRUE(pack.required_real_props.to_string_vector().empty());
}

TEST(ReactionDataArgumentPack, RegularAndEphemeralRequirementsAreDeduplicated) {
  const auto int_props =
      Properties<INT>(std::array<int, 1>{default_properties.panic});
  const auto real_props =
      Properties<REAL>(std::array<int, 1>{default_properties.weight});
  const auto ephemeral_int_props = Properties<INT>(std::array<int, 2>{
      default_properties.panic, default_properties.internal_state});
  const auto ephemeral_real_props = Properties<REAL>(std::array<int, 2>{
      default_properties.weight, default_properties.velocity});

  // Exercise the compiled pack constructor and its host-base forwarding path.
  auto pack = SingleReactionDataArgumentPack(
      int_props, real_props, ephemeral_int_props, ephemeral_real_props);
  ReactionDataBase<FixedRateDataOnDevice> data(
      int_props, real_props, ephemeral_int_props, ephemeral_real_props);
  auto forwarded_pack = data.get_arg_pack();

  const auto expected_int =
      std::vector<std::string>{"INTERNAL_STATE", "REACTIONS_PANIC_FLAG"};
  const auto expected_real = std::vector<std::string>{"VELOCITY", "WEIGHT"};
  EXPECT_EQ(pack.required_int_props.to_string_vector(), expected_int);
  EXPECT_EQ(pack.required_real_props.to_string_vector(), expected_real);
  EXPECT_EQ(forwarded_pack.required_int_props.to_string_vector(), expected_int);
  EXPECT_EQ(forwarded_pack.required_real_props.to_string_vector(),
            expected_real);
}

TEST(ReactionDataArgumentPack, NestedCompositionReindexesChildren) {
  auto particle_group = create_test_particle_group(100);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  NP::particle_loop(
      "set_nested_composition_properties", particle_group,
      [=](auto weight, auto position) {
        weight[0] = 3.0;
        position[0] = 7.0;
      },
      NP::Access::write(NP::Sym<REAL>("WEIGHT")),
      NP::Access::write(NP::Sym<REAL>("POSITION")))
      ->execute();

  auto coefficient = FixedCoefficientData(2.0);
  auto weight = ExtractorData<1>(NP::Sym<REAL>("WEIGHT"));
  auto inner = ConcatenatorData(coefficient, weight);
  auto inner_pack = inner.get_arg_pack();
  EXPECT_EQ(inner_pack.required_real_props.find_index("WEIGHT"), 0);

  // POSITION sorts before WEIGHT, shifting both nested children's indices.
  auto outer =
      ConcatenatorData(inner, ExtractorData<1>(NP::Sym<REAL>("POSITION")));
  auto outer_pack = outer.get_arg_pack();
  EXPECT_EQ(outer_pack.required_real_props.find_index("WEIGHT"), 1);
  auto calculator = DataCalculator(outer);
  EXPECT_EQ(calculator.get_data_size(), 3);

  const int cell_count = particle_group->domain->mesh->get_cell_count();
  for (int cell = 0; cell < cell_count; cell++) {
    const auto npart = particle_sub_group->get_npart_cell(cell);
    auto buffer = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, npart, calculator.get_data_size());
    buffer->fill(0.0);
    calculator.fill_buffer(buffer, particle_sub_group, cell, cell + 1);
    const auto values = buffer->get();

    for (int particle = 0; particle < npart; particle++) {
      EXPECT_DOUBLE_EQ(values.at(3 * particle), 6.0);
      EXPECT_DOUBLE_EQ(values.at(3 * particle + 1), 3.0);
      EXPECT_DOUBLE_EQ(values.at(3 * particle + 2), 7.0);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
