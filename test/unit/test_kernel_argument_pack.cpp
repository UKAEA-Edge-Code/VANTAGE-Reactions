#include "include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(KernelArgumentPack, EphemeralNamesShareCanonicalSlots) {
  const auto props = default_properties;
  auto map = PropertiesMap();
  map.at(props.id) = "SHARED_INT";
  map.at(props.internal_state) = "SHARED_INT";
  map.at(props.weight) = "SHARED_REAL";
  map.at(props.velocity) = "SHARED_REAL";
  KernelInputDescription inputs(
      Properties<INT>({props.id, props.cell_id}),
      Properties<REAL>({props.weight, props.position}),
      Properties<INT>({props.internal_state}),
      Properties<REAL>({props.velocity}), map);
  EXPECT_EQ(inputs.required_int_props.to_string_vector(),
            (std::vector<std::string>{"CELL_ID", "SHARED_INT"}));
  EXPECT_EQ(inputs.required_real_props.to_string_vector(),
            (std::vector<std::string>{"POSITION", "SHARED_REAL"}));
  EXPECT_EQ(inputs.required_int_props_ephemeral.to_string_vector(),
            (std::vector<std::string>{"SHARED_INT"}));
  EXPECT_EQ(inputs.required_real_props_ephemeral.to_string_vector(),
            (std::vector<std::string>{"SHARED_REAL"}));
  EXPECT_EQ(inputs.required_int_props.find_index("SHARED_INT"), 1);
  EXPECT_EQ(inputs.required_real_props.find_index("SHARED_REAL"), 1);
}

TEST(KernelArgumentPack, EphemeralOnlySpeciesInputs) {
  const auto props = default_properties;
  const Species ion("ion");
  auto map = PropertiesMap();
  map.at(props.source_momentum) = "MOMENTUM";
  KernelInputDescription inputs(
      {}, {}, Properties<INT>({props.panic}),
      Properties<REAL>({ion}, {props.source_momentum}), map);
  EXPECT_EQ(inputs.required_int_props.to_string_vector(),
            inputs.required_int_props_ephemeral.to_string_vector());
  EXPECT_EQ(inputs.required_real_props.to_string_vector(),
            inputs.required_real_props_ephemeral.to_string_vector());
  EXPECT_EQ(
      inputs.required_real_props.find_index(species_property(ion, "MOMENTUM")),
      0);
}

TEST(KernelArgumentPack, IndependentPairLayouts) {
  const auto props = default_properties;
  PairReactionKernelArgumentPack pair{
      KernelPropertyDescription({}, Properties<REAL>({props.weight})),
      KernelPropertyDescription(
          {}, Properties<REAL>({props.velocity, props.weight})),
      KernelPropertyDescription(Properties<INT>({props.internal_state}), {}),
      KernelPropertyDescription(
          Properties<INT>({props.id, props.internal_state}), {})};
  EXPECT_EQ(pair.inputs_a.required_real_props.find_index("WEIGHT"), 0);
  EXPECT_EQ(pair.inputs_b.required_real_props.find_index("WEIGHT"), 1);
  EXPECT_EQ(pair.descendants_a.required_int_props.find_index("INTERNAL_STATE"),
            0);
  EXPECT_EQ(pair.descendants_b.required_int_props.find_index("INTERNAL_STATE"),
            1);
}
