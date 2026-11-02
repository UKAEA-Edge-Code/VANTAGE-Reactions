#include "include/test_common.hpp"
#include <cstdlib>
#include <stdexcept>

using namespace VANTAGE::Reactions;

TEST(KernelProductSchema, EmptySchemaIsInitialized) {
  auto schema = build_kernel_product_schema({}, {}, 3);
  EXPECT_EQ(schema->num_properties_int, 0);
  EXPECT_EQ(schema->num_properties_real, 0);
}

TEST(KernelProductSchema, LegacyDimensionsAndCanonicalIndices) {
  const auto props = default_properties;
  ArgumentNameSet<INT> descendant_int(
      Properties<INT>({props.internal_state, props.id, props.id}));
  ArgumentNameSet<REAL> descendant_real(
      Properties<REAL>({props.weight, props.velocity, props.source_momentum,
                        props.position, props.weight}));
  for (const int ndim : {2, 3}) {
    auto schema =
        build_kernel_product_schema(descendant_int, descendant_real, ndim);
    EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("VELOCITY")), ndim);
    EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("POSITION")), 1);
    EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("SOURCE_MOMENTUM")), 1);
    for (const auto &name : descendant_int.to_string_vector()) {
      EXPECT_EQ(schema->get_sym_index(NP::Sym<INT>(name)),
                descendant_int.find_index(name));
    }
    for (const auto &name : descendant_real.to_string_vector()) {
      EXPECT_EQ(schema->get_sym_index(NP::Sym<REAL>(name)),
                descendant_real.find_index(name));
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
  ArgumentNameSet<INT> descendant_int(Properties<INT>({props.internal_state}),
                                      map);
  ArgumentNameSet<REAL> descendant_real(
      Properties<REAL>({props.velocity, props.weight}, {ion, electron},
                       {props.source_momentum}),
      map);
  auto schema =
      build_kernel_product_schema(descendant_int, descendant_real, 2, map);
  EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("Z_CUSTOM_VELOCITY")), 2);
  EXPECT_EQ(schema->get_num_components(NP::Sym<INT>("CUSTOM_STATE")), 1);
  EXPECT_EQ(schema->get_sym_index(NP::Sym<REAL>("VELOCITY")), -1);
  for (const auto &species : {electron, ion}) {
    const auto name = species_property(species, "CUSTOM_MOMENTUM");
    EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>(name)), 1);
  }
}

TEST(KernelProductSchema, InvalidVelocityDimensionsAssert) {
  ArgumentNameSet<REAL> descendant_real(
      Properties<REAL>({default_properties.velocity}));
  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(build_kernel_product_schema({}, descendant_real, 0),
                 std::logic_error);
    EXPECT_THROW(build_kernel_product_schema({}, descendant_real, -1),
                 std::logic_error);
  }
  // An unused velocity dimension does not constrain empty/scalar-only schemas.
  EXPECT_NO_THROW(build_kernel_product_schema({}, {}, 0));
  ArgumentNameSet<REAL> scalar(Properties<REAL>({default_properties.weight}));
  EXPECT_NO_THROW(build_kernel_product_schema({}, scalar, 0));
}

TEST(KernelProductSchema, ResolvedNamesCanBeDeclaredDirectly) {
  ArgumentNameSet<INT> descendant_int;
  ArgumentNameSet<REAL> descendant_real;
  descendant_int.add("CUSTOM_INT");
  descendant_real.add("CUSTOM_REAL");
  auto schema = build_kernel_product_schema(descendant_int, descendant_real, 3);
  EXPECT_EQ(schema->get_sym_index(NP::Sym<INT>("CUSTOM_INT")), 0);
  EXPECT_EQ(schema->get_sym_index(NP::Sym<REAL>("CUSTOM_REAL")), 0);
  EXPECT_EQ(schema->get_num_components(NP::Sym<INT>("CUSTOM_INT")), 1);
  EXPECT_EQ(schema->get_num_components(NP::Sym<REAL>("CUSTOM_REAL")), 1);
}
