#include "../include/reactions_lib/reaction_data_abstract.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

ReactionDataStorage::ReactionDataStorage(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    Properties<INT> required_int_props_ephemeral,
    Properties<REAL> required_real_props_ephemeral,
    std::map<int, std::string> properties_map)
    : required_int_props(
          ArgumentNameSet(required_int_props, properties_map)
              .merge_with(ArgumentNameSet(required_int_props_ephemeral,
                                          properties_map))),
      required_real_props(
          ArgumentNameSet(required_real_props, properties_map)
              .merge_with(ArgumentNameSet(required_real_props_ephemeral,
                                          properties_map))),
      properties_map(properties_map) {}

ReactionDataStorage::ReactionDataStorage(
    std::map<int, std::string> properties_map)
    : ReactionDataStorage(Properties<INT>(), Properties<REAL>(),
                          Properties<INT>(), Properties<REAL>(),
                          properties_map) {}

ReactionDataStorage::ReactionDataStorage(
    Properties<INT> required_int_props,
    std::map<int, std::string> properties_map)
    : ReactionDataStorage(required_int_props, Properties<REAL>(),
                          Properties<INT>(), Properties<REAL>(),
                          properties_map) {}

ReactionDataStorage::ReactionDataStorage(
    Properties<REAL> required_real_props,
    std::map<int, std::string> properties_map)
    : ReactionDataStorage(Properties<INT>(), required_real_props,
                          Properties<INT>(), Properties<REAL>(),
                          properties_map) {}

ReactionDataStorage::ReactionDataStorage(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    std::map<int, std::string> properties_map)
    : ReactionDataStorage(required_int_props, required_real_props,
                          Properties<INT>(), Properties<REAL>(),
                          properties_map) {}

ReactionDataStorage::~ReactionDataStorage() = default;

ArgumentNameSet<INT> ReactionDataStorage::get_required_int_props() const {
  return this->required_int_props;
}

void ReactionDataStorage::set_required_int_props(
    const ArgumentNameSet<INT> &props) {
  this->required_int_props = props;
}

std::vector<NP::Sym<INT>> ReactionDataStorage::get_required_int_sym_vector() {
  return this->required_int_props.to_sym_vector();
}

ArgumentNameSet<REAL> ReactionDataStorage::get_required_real_props() const {
  return this->required_real_props;
}

void ReactionDataStorage::set_required_real_props(
    const ArgumentNameSet<REAL> &props) {
  this->required_real_props = props;
}

std::vector<NP::Sym<REAL>> ReactionDataStorage::get_required_real_sym_vector() {
  return this->required_real_props.to_sym_vector();
}

const std::map<int, std::string> &
ReactionDataStorage::get_properties_map() const {
  return this->properties_map;
}

}; // namespace VANTAGE::Reactions
