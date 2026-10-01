#include "../include/reactions_lib/pair_reaction_kernels.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

PairReactionKernelsBase::PairReactionKernelsBase(
    Properties<INT> req_int_props_a, Properties<REAL> req_real_props_a,
    Properties<INT> req_int_props_b, Properties<REAL> req_real_props_b,
    INT pre_req_ndims, std::map<int, std::string> properties_map)
    : required_int_props_a(req_int_props_a),
      required_real_props_a(req_real_props_a),
      required_int_props_b(req_int_props_b),
      required_real_props_b(req_real_props_b), pre_req_ndims(pre_req_ndims) {
  NESOWARN(map_subset_check(properties_map),
           "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

  this->properties_map = properties_map;
}

PairReactionKernelsBase::PairReactionKernelsBase(
    std::map<int, std::string> properties_map)
    : PairReactionKernelsBase(Properties<INT>(), Properties<REAL>(),
                              Properties<INT>(), Properties<REAL>(), 0,
                              properties_map) {}

PairReactionKernelsBase::PairReactionKernelsBase(
    Properties<INT> required_int_props, INT pre_req_ndims,
    std::map<int, std::string> properties_map)
    : PairReactionKernelsBase(required_int_props, Properties<REAL>(),
                              required_int_props, Properties<REAL>(),
                              pre_req_ndims, properties_map) {}

PairReactionKernelsBase::PairReactionKernelsBase(
    Properties<REAL> required_real_props, INT pre_req_ndims,
    std::map<int, std::string> properties_map)
    : PairReactionKernelsBase(Properties<INT>(), required_real_props,
                              Properties<INT>(), required_real_props,
                              pre_req_ndims, properties_map) {}

PairReactionKernelsBase::PairReactionKernelsBase(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    INT pre_req_ndims, std::map<int, std::string> properties_map)
    : PairReactionKernelsBase(required_int_props, required_real_props,
                              required_int_props, required_real_props,
                              pre_req_ndims, properties_map) {}

std::vector<std::string> PairReactionKernelsBase::get_required_int_props_a() {
  return this->required_int_props_a.get_prop_names(this->properties_map);
}

std::vector<std::string> PairReactionKernelsBase::get_required_int_props_b() {
  return this->required_int_props_b.get_prop_names(this->properties_map);
}

std::vector<std::string> PairReactionKernelsBase::get_required_real_props_a() {
  return this->required_real_props_a.get_prop_names(this->properties_map);
}

std::vector<std::string> PairReactionKernelsBase::get_required_real_props_b() {
  return this->required_real_props_b.get_prop_names(this->properties_map);
}

const Properties<INT> &
PairReactionKernelsBase::get_required_descendant_int_props_a() {
  return this->required_descendant_int_props_a;
}

const Properties<REAL> &
PairReactionKernelsBase::get_required_descendant_real_props_a() {
  return this->required_descendant_real_props_a;
}

const Properties<INT> &
PairReactionKernelsBase::get_required_descendant_int_props_b() {
  return this->required_descendant_int_props_b;
}

const Properties<REAL> &
PairReactionKernelsBase::get_required_descendant_real_props_b() {
  return this->required_descendant_real_props_b;
}

std::shared_ptr<NP::ProductMatrixSpec>
PairReactionKernelsBase::get_descendant_matrix_spec_a() {
  return this->descendant_matrix_spec_a;
}

std::shared_ptr<NP::ProductMatrixSpec>
PairReactionKernelsBase::get_descendant_matrix_spec_b() {
  return this->descendant_matrix_spec_b;
}

const INT &PairReactionKernelsBase::get_pre_ndims() const {
  return this->pre_req_ndims;
}

const INT &PairReactionKernelsBase::get_num_products_a() const {
  return this->num_products_a;
}

const INT &PairReactionKernelsBase::get_num_products_b() const {
  return this->num_products_b;
}

void PairReactionKernelsBase::set_required_descendant_int_props_a(
    const Properties<INT> &required_descendant_int_props) {
  this->required_descendant_int_props_a = required_descendant_int_props;
}

void PairReactionKernelsBase::set_required_descendant_real_props_a(
    const Properties<REAL> &required_descendant_real_props) {
  this->required_descendant_real_props_a = required_descendant_real_props;
}

void PairReactionKernelsBase::set_required_descendant_int_props_b(
    const Properties<INT> &required_descendant_int_props) {
  this->required_descendant_int_props_b = required_descendant_int_props;
}

void PairReactionKernelsBase::set_required_descendant_real_props_b(
    const Properties<REAL> &required_descendant_real_props) {
  this->required_descendant_real_props_b = required_descendant_real_props;
}

}; // namespace VANTAGE::Reactions
