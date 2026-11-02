#include "include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(KernelArgumentPack, EphemeralNamesShareCanonicalSlots) {
  const auto props = default_properties;
  auto map = PropertiesMap();
  map.at(props.id) = "SHARED_INT";
  map.at(props.internal_state) = "SHARED_INT";
  map.at(props.weight) = "SHARED_REAL";
  map.at(props.velocity) = "SHARED_REAL";
  SingleReactionKernelArgumentPack pack(
      Properties<INT>({props.id, props.cell_id}),
      Properties<REAL>({props.weight, props.position}),
      Properties<INT>({props.internal_state}),
      Properties<REAL>({props.velocity}), Properties<INT>({props.id}),
      Properties<REAL>({props.weight}), map);
  EXPECT_EQ(pack.required_int_props.to_string_vector(),
            (std::vector<std::string>{"CELL_ID", "SHARED_INT"}));
  EXPECT_EQ(pack.required_real_props.to_string_vector(),
            (std::vector<std::string>{"POSITION", "SHARED_REAL"}));
  EXPECT_EQ(pack.required_descendant_int_props.to_string_vector(),
            (std::vector<std::string>{"SHARED_INT"}));
  EXPECT_EQ(pack.required_descendant_real_props.to_string_vector(),
            (std::vector<std::string>{"SHARED_REAL"}));
  EXPECT_EQ(pack.required_int_props.find_index("SHARED_INT"), 1);
  EXPECT_EQ(pack.required_real_props.find_index("SHARED_REAL"), 1);
}

TEST(KernelArgumentPack, EphemeralOnlySpeciesInputs) {
  const auto props = default_properties;
  const Species ion("ion");
  auto map = PropertiesMap();
  map.at(props.source_momentum) = "MOMENTUM";
  SingleReactionKernelArgumentPack pack(
      {}, {}, Properties<INT>({props.panic}),
      Properties<REAL>({ion}, {props.source_momentum}), map);
  EXPECT_EQ(pack.required_int_props.to_string_vector(),
            (std::vector<std::string>{"REACTIONS_PANIC_FLAG"}));
  EXPECT_EQ(pack.required_real_props.to_string_vector(),
            (std::vector<std::string>{species_property(ion, "MOMENTUM")}));
  EXPECT_EQ(
      pack.required_real_props.find_index(species_property(ion, "MOMENTUM")),
      0);
  EXPECT_TRUE(pack.required_descendant_int_props.to_string_vector().empty());
  EXPECT_TRUE(pack.required_descendant_real_props.to_string_vector().empty());
}
