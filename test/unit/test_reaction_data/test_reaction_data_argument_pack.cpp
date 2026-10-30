#include "../include/mock_particle_group.hpp"
#include "../include/test_common.hpp"

using namespace VANTAGE::Reactions;

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
