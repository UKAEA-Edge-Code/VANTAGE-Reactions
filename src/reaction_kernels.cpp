#include "../include/reactions_lib/reaction_kernels.hpp"

namespace VANTAGE::Reactions {

ReactionKernelsBase::ReactionKernelsBase(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    Properties<INT> required_int_props_ephemeral,
    Properties<REAL> required_real_props_ephemeral, INT pre_req_ndims,
    std::map<int, std::string> properties_map)
    : required_int_props(
          required_int_props.merge_with(required_int_props_ephemeral)),
      required_real_props(
          required_real_props.merge_with(required_real_props_ephemeral)),
      required_int_props_ephemeral(required_int_props_ephemeral),
      required_real_props_ephemeral(required_real_props_ephemeral),
      pre_req_ndims(pre_req_ndims) {
  NESOWARN(map_subset_check(properties_map),
           "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

  this->properties_map = properties_map;
}

ReactionKernelsBase::ReactionKernelsBase(
    std::map<int, std::string> properties_map)
    : ReactionKernelsBase(Properties<INT>(), Properties<REAL>(),
                          Properties<INT>(), Properties<REAL>(), 0,
                          properties_map) {}

ReactionKernelsBase::ReactionKernelsBase(
    Properties<INT> required_int_props, INT pre_req_ndims,
    std::map<int, std::string> properties_map)
    : ReactionKernelsBase(required_int_props, Properties<REAL>(),
                          Properties<INT>(), Properties<REAL>(), pre_req_ndims,
                          properties_map) {}

ReactionKernelsBase::ReactionKernelsBase(
    Properties<REAL> required_real_props, INT pre_req_ndims,
    std::map<int, std::string> properties_map)
    : ReactionKernelsBase(Properties<INT>(), required_real_props,
                          Properties<INT>(), Properties<REAL>(), pre_req_ndims,
                          properties_map) {}

ReactionKernelsBase::ReactionKernelsBase(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    INT pre_req_ndims, std::map<int, std::string> properties_map)
    : ReactionKernelsBase(required_int_props, required_real_props,
                          Properties<INT>(), Properties<REAL>(), pre_req_ndims,
                          properties_map) {}

std::vector<std::string> ReactionKernelsBase::get_required_int_props() {
  return this->required_int_props.get_prop_names(this->properties_map);
}

std::vector<std::string> ReactionKernelsBase::get_required_real_props() {
  return this->required_real_props.get_prop_names(this->properties_map);
}

std::vector<std::string>
ReactionKernelsBase::get_required_int_props_ephemeral() {
  return this->required_int_props_ephemeral.get_prop_names(
      this->properties_map);
}

std::vector<std::string>
ReactionKernelsBase::get_required_real_props_ephemeral() {
  return this->required_real_props_ephemeral.get_prop_names(
      this->properties_map);
}

const Properties<INT> &
ReactionKernelsBase::get_required_descendant_int_props() {
  return this->required_descendant_int_props;
}

const Properties<REAL> &
ReactionKernelsBase::get_required_descendant_real_props() {
  return this->required_descendant_real_props;
}

std::shared_ptr<ProductMatrixSpec>
ReactionKernelsBase::get_descendant_matrix_spec() {
  return this->descendant_matrix_spec;
}

const INT &ReactionKernelsBase::get_pre_ndims() const {
  return this->pre_req_ndims;
}

void ReactionKernelsBase::set_required_descendant_int_props(
    const Properties<INT> &required_descendant_int_props) {
  this->required_descendant_int_props = required_descendant_int_props;
}

void ReactionKernelsBase::set_required_descendant_real_props(
    const Properties<REAL> &required_descendant_real_props) {
  this->required_descendant_real_props = required_descendant_real_props;
}

} // namespace VANTAGE::Reactions
