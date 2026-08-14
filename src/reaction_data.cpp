#include "../include/reactions_lib/reaction_data.hpp"

namespace VANTAGE::Reactions {

ReactionDataBaseImpl::ReactionDataBaseImpl(
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

ReactionDataBaseImpl::ReactionDataBaseImpl(
    std::map<int, std::string> properties_map)
    : ReactionDataBaseImpl(Properties<INT>(), Properties<REAL>(),
                           Properties<INT>(), Properties<REAL>(),
                           properties_map) {}

ReactionDataBaseImpl::ReactionDataBaseImpl(
    Properties<INT> required_int_props,
    std::map<int, std::string> properties_map)
    : ReactionDataBaseImpl(required_int_props, Properties<REAL>(),
                           Properties<INT>(), Properties<REAL>(),
                           properties_map) {}

ReactionDataBaseImpl::ReactionDataBaseImpl(
    Properties<REAL> required_real_props,
    std::map<int, std::string> properties_map)
    : ReactionDataBaseImpl(Properties<INT>(), required_real_props,
                           Properties<INT>(), Properties<REAL>(),
                           properties_map) {}

ReactionDataBaseImpl::ReactionDataBaseImpl(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    std::map<int, std::string> properties_map)
    : ReactionDataBaseImpl(required_int_props, required_real_props,
                           Properties<INT>(), Properties<REAL>(),
                           properties_map) {}

ReactionDataBaseImpl::~ReactionDataBaseImpl() = default;

ArgumentNameSet<INT> ReactionDataBaseImpl::get_required_int_props() {
  return this->required_int_props;
}

void ReactionDataBaseImpl::set_required_int_props(
    const ArgumentNameSet<INT> &props) {
  this->required_int_props = props;
  this->index_on_device_object();
}

std::vector<Sym<INT>> ReactionDataBaseImpl::get_required_int_sym_vector() {
  return this->required_int_props.to_sym_vector();
}

ArgumentNameSet<REAL> ReactionDataBaseImpl::get_required_real_props() {
  return this->required_real_props;
}

void ReactionDataBaseImpl::set_required_real_props(
    const ArgumentNameSet<REAL> &props) {
  this->required_real_props = props;
  this->index_on_device_object();
}

std::vector<Sym<REAL>> ReactionDataBaseImpl::get_required_real_sym_vector() {
  return this->required_real_props.to_sym_vector();
}

void ReactionDataBaseImpl::index_on_device_object() {}

}; // namespace VANTAGE::Reactions
