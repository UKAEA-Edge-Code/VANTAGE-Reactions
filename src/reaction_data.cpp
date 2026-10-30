#include "../include/reactions_lib/reaction_data.hpp"

namespace VANTAGE::Reactions {

SingleReactionDataArgumentPack::SingleReactionDataArgumentPack(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    Properties<INT> required_int_props_ephemeral,
    Properties<REAL> required_real_props_ephemeral,
    const std::map<int, std::string> &properties_map)
    : required_int_props(
          ArgumentNameSet(required_int_props, properties_map)
              .merge_with(ArgumentNameSet(required_int_props_ephemeral,
                                          properties_map))),
      required_real_props(
          ArgumentNameSet(required_real_props, properties_map)
              .merge_with(ArgumentNameSet(required_real_props_ephemeral,
                                          properties_map))) {}

} // namespace VANTAGE::Reactions
