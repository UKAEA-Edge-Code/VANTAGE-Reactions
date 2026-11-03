#include "../include/reactions_lib/pair_reaction_kernels.hpp"

namespace VANTAGE::Reactions {

PairReactionKernelArgumentPack::PairReactionKernelArgumentPack(
    Properties<INT> required_int_props_a,
    Properties<REAL> required_real_props_a,
    Properties<INT> required_int_props_b,
    Properties<REAL> required_real_props_b,
    Properties<INT> required_descendant_int_props_a,
    Properties<REAL> required_descendant_real_props_a,
    Properties<INT> required_descendant_int_props_b,
    Properties<REAL> required_descendant_real_props_b,
    const PropertiesMap &properties_map)
    : required_int_props_a(std::move(required_int_props_a), properties_map),
      required_real_props_a(std::move(required_real_props_a), properties_map),
      required_int_props_b(std::move(required_int_props_b), properties_map),
      required_real_props_b(std::move(required_real_props_b), properties_map),
      required_descendant_int_props_a(
          std::move(required_descendant_int_props_a), properties_map),
      required_descendant_real_props_a(
          std::move(required_descendant_real_props_a), properties_map),
      required_descendant_int_props_b(
          std::move(required_descendant_int_props_b), properties_map),
      required_descendant_real_props_b(
          std::move(required_descendant_real_props_b), properties_map) {}

PairReactionKernelArgumentPack::PairReactionKernelArgumentPack(
    Properties<INT> required_int_props_a,
    Properties<REAL> required_real_props_a,
    Properties<INT> required_int_props_b,
    Properties<REAL> required_real_props_b, const PropertiesMap &properties_map)
    : PairReactionKernelArgumentPack(
          std::move(required_int_props_a), std::move(required_real_props_a),
          std::move(required_int_props_b), std::move(required_real_props_b),
          Properties<INT>(), Properties<REAL>(), Properties<INT>(),
          Properties<REAL>(), properties_map) {}

namespace detail {

void validate_kernel_product_counts(INT extent, INT count_a, INT count_b) {
  NESOASSERT(extent >= 0 && count_a >= 0 && count_b >= 0 && count_a <= extent &&
                 count_b == extent - count_a,
             "Kernel A/B product counts must be nonnegative and sum to the "
             "device output-state extent.");
}

} // namespace detail

} // namespace VANTAGE::Reactions
