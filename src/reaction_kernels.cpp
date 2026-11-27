#include "../include/reactions_lib/reaction_kernels.hpp"

namespace VANTAGE::Reactions {

SingleReactionKernelArgumentPack::SingleReactionKernelArgumentPack(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    Properties<INT> required_int_props_ephemeral,
    Properties<REAL> required_real_props_ephemeral,
    Properties<INT> required_descendant_int_props,
    Properties<REAL> required_descendant_real_props,
    const PropertiesMap &properties_map)
    : required_int_props(
          ArgumentNameSet(std::move(required_int_props), properties_map)
              .merge_with(ArgumentNameSet(
                  std::move(required_int_props_ephemeral), properties_map))),
      required_real_props(
          ArgumentNameSet(std::move(required_real_props), properties_map)
              .merge_with(ArgumentNameSet(
                  std::move(required_real_props_ephemeral), properties_map))),
      required_descendant_int_props(std::move(required_descendant_int_props),
                                    properties_map),
      required_descendant_real_props(std::move(required_descendant_real_props),
                                     properties_map) {}

SingleReactionKernelArgumentPack::SingleReactionKernelArgumentPack(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    Properties<INT> required_int_props_ephemeral,
    Properties<REAL> required_real_props_ephemeral,
    const PropertiesMap &properties_map)
    : SingleReactionKernelArgumentPack(
          std::move(required_int_props), std::move(required_real_props),
          std::move(required_int_props_ephemeral),
          std::move(required_real_props_ephemeral), Properties<INT>(),
          Properties<REAL>(), properties_map) {}

} // namespace VANTAGE::Reactions
