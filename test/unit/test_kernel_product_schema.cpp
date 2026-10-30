#include "include/test_common.hpp"
#include <stdexcept>

using namespace VANTAGE::Reactions;

TEST(KernelProductSchema, EmptySchemaIsInitialized) {
  // No product count is inferred from an empty explicit-write description.
  auto schema = build_kernel_product_schema({}, 3);
  EXPECT_EQ(schema->num_properties_int, 0);
  EXPECT_EQ(schema->num_properties_real, 0);
}

TEST(KernelProductSchema, LegacyDimensionsAndCanonicalIndices) {
  const auto props = default_properties;
  KernelPropertyDescription descendants(
      Properties<INT>({props.internal_state, props.id, props.id}),
      Properties<REAL>({props.weight, props.velocity, props.source_momentum,
                        props.position, props.weight}));
  for (const int ndim : {2, 3}) {
    auto schema = build_kernel_product_schema(descendants, ndim);
    EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("VELOCITY")), ndim);
    // This pass deliberately preserves scalar dimensions for other fields.
    EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("POSITION")), 1);
    EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("SOURCE_MOMENTUM")), 1);
    for (const auto &name : descendants.required_int_props.to_string_vector()) {
      EXPECT_EQ(schema->get_sym_index(NP::Sym<INT>(name)),
                descendants.required_int_props.find_index(name));
    }
    for (const auto &name :
         descendants.required_real_props.to_string_vector()) {
      EXPECT_EQ(schema->get_sym_index(NP::Sym<REAL>(name)),
                descendants.required_real_props.find_index(name));
    }
  }
}

TEST(KernelProductSchema, CustomMapAndSpeciesQualifiedFields) {
  const auto props = default_properties;
  const Species electron("electron");
  const Species ion("ion");
  auto map = PropertiesMap();
  map.at(props.velocity) = "Z_CUSTOM_VELOCITY";
  map.at(props.internal_state) = "CUSTOM_STATE";
  map.at(props.source_momentum) = "CUSTOM_MOMENTUM";
  KernelPropertyDescription descendants(
      Properties<INT>({props.internal_state}),
      Properties<REAL>({props.velocity, props.weight}, {ion, electron},
                       {props.source_momentum}),
      map);
  auto schema = build_kernel_product_schema(descendants, 2, map);
  EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("Z_CUSTOM_VELOCITY")), 2);
  EXPECT_EQ(schema->get_num_components(NP::Sym<INT>("CUSTOM_STATE")), 1);
  EXPECT_EQ(schema->get_sym_index(NP::Sym<REAL>("VELOCITY")), -1);
  for (const auto &species : {electron, ion}) {
    const auto name = species_property(species, "CUSTOM_MOMENTUM");
    EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>(name)), 1);
  }
}

TEST(KernelProductSchema, IndependentChannelsAndFreshSchemas) {
  const auto props = default_properties;
  PairReactionKernelArgumentPack pack;
  pack.descendants_a = KernelPropertyDescription(
      {}, Properties<REAL>({props.velocity, props.weight}));
  pack.descendants_b = KernelPropertyDescription(
      {}, Properties<REAL>({props.position, props.velocity, props.weight}));
  auto schema_a = build_kernel_product_schema(pack.descendants_a, 3);
  auto schema_b = build_kernel_product_schema(pack.descendants_b, 3);
  EXPECT_EQ(schema_a->get_sym_index(NP::Sym<REAL>("VELOCITY")), 0);
  EXPECT_EQ(schema_b->get_sym_index(NP::Sym<REAL>("VELOCITY")), 1);
  auto second_a = build_kernel_product_schema(pack.descendants_a, 2);
  EXPECT_NE(schema_a, second_a);
  EXPECT_EQ(schema_a->get_num_components(NP::Sym<REAL>("VELOCITY")), 3);
  EXPECT_EQ(second_a->get_num_components(NP::Sym<REAL>("VELOCITY")), 2);
  second_a->set_default_value(NP::Sym<REAL>("WEIGHT"), 0, 7.0);
  EXPECT_TRUE(schema_a->default_values_real.empty());
  EXPECT_TRUE(schema_b->default_values_real.empty());
}

TEST(KernelProductSchema, InvalidVelocityDimensionsThrowStandardExceptions) {
  KernelPropertyDescription descendants(
      {}, Properties<REAL>({default_properties.velocity}));
  EXPECT_THROW(build_kernel_product_schema(descendants, 0),
               std::invalid_argument);
  EXPECT_THROW(build_kernel_product_schema(descendants, -1),
               std::invalid_argument);
  // An unused velocity dimension does not constrain empty/scalar-only schemas.
  EXPECT_NO_THROW(build_kernel_product_schema({}, 0));
  KernelPropertyDescription scalar(
      {}, Properties<REAL>({default_properties.weight}));
  EXPECT_NO_THROW(build_kernel_product_schema(scalar, 0));
}

TEST(KernelProductSchema, ResolvedNamesCanBeDeclaredDirectly) {
  KernelPropertyDescription descendants;
  descendants.required_int_props.add("CUSTOM_INT");
  descendants.required_real_props.add("CUSTOM_REAL");
  auto schema = build_kernel_product_schema(descendants, 3);
  EXPECT_EQ(schema->get_sym_index(NP::Sym<INT>("CUSTOM_INT")), 0);
  EXPECT_EQ(schema->get_sym_index(NP::Sym<REAL>("CUSTOM_REAL")), 0);
  EXPECT_EQ(schema->get_num_components(NP::Sym<INT>("CUSTOM_INT")), 1);
  EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("CUSTOM_REAL")), 1);
}
