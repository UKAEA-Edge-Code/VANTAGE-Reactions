
#include "../include/reactions_lib/reaction_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

ReactionDataBaseImpl::ReactionDataBaseImpl(
    Properties<NP::INT> required_int_props,
    Properties<NP::REAL> required_real_props,
    Properties<NP::INT> required_int_props_ephemeral,
    Properties<NP::REAL> required_real_props_ephemeral,
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

ReactionDataBaseImpl::ReactionDataBaseImpl(
    std::map<int, std::string> properties_map)
    : ReactionDataBaseImpl(Properties<NP::INT>(), Properties<NP::REAL>(),
                           Properties<NP::INT>(), Properties<NP::REAL>(),
                           properties_map) {}

ReactionDataBaseImpl::ReactionDataBaseImpl(
    Properties<NP::INT> required_int_props,
    std::map<int, std::string> properties_map)
    : ReactionDataBaseImpl(required_int_props, Properties<NP::REAL>(),
                           Properties<NP::INT>(), Properties<NP::REAL>(),
                           properties_map) {}

ReactionDataBaseImpl::ReactionDataBaseImpl(
    Properties<NP::REAL> required_real_props,
    std::map<int, std::string> properties_map)
    : ReactionDataBaseImpl(Properties<NP::INT>(), required_real_props,
                           Properties<NP::INT>(), Properties<NP::REAL>(),
                           properties_map) {}

ReactionDataBaseImpl::ReactionDataBaseImpl(
    Properties<NP::INT> required_int_props,
    Properties<NP::REAL> required_real_props,
    std::map<int, std::string> properties_map)
    : ReactionDataBaseImpl(required_int_props, required_real_props,
                           Properties<NP::INT>(), Properties<NP::REAL>(),
                           properties_map) {}

ReactionDataBaseImpl::~ReactionDataBaseImpl() = default;

ArgumentNameSet<NP::INT> ReactionDataBaseImpl::get_required_int_props() {
  return this->required_int_props;
}

void ReactionDataBaseImpl::set_required_int_props(
    const ArgumentNameSet<NP::INT> &props) {
  this->required_int_props = props;
  this->index_on_device_object();
}

std::vector<NP::Sym<NP::INT>>
ReactionDataBaseImpl::get_required_int_sym_vector() {
  return this->required_int_props.to_sym_vector();
}

ArgumentNameSet<NP::REAL> ReactionDataBaseImpl::get_required_real_props() {
  return this->required_real_props;
}

void ReactionDataBaseImpl::set_required_real_props(
    const ArgumentNameSet<NP::REAL> &props) {
  this->required_real_props = props;
  this->index_on_device_object();
}

std::vector<NP::Sym<NP::REAL>>
ReactionDataBaseImpl::get_required_real_sym_vector() {
  return this->required_real_props.to_sym_vector();
}

void ReactionDataBaseImpl::index_on_device_object() {}

}; // namespace VANTAGE::Reactions
