#pragma once
#include "particle_properties_map.hpp"
#include "typedefs.hpp"
#include <array>
#include <stdexcept>
#include <vector>

using namespace NESO::Particles;
using namespace ParticlePropertiesIndices;

struct Species {
  Species(std::string name_) : name(name_){};

  Species(std::string name_, REAL mass_) : name(), mass(mass_){};

  Species(std::string name_, REAL mass_, REAL charge_)
      : name(name_), mass(mass_), charge(charge_){};

  Species(std::string name_, REAL mass_, REAL charge_, INT id_)
      : name(name_), mass(mass_), charge(charge_), id(id_){};

public:
  const std::string get_name() const { return this->name; }

  const REAL get_mass() const { return this->mass; }

  const REAL get_charge() const { return this->charge; }

  const INT get_id() const { return this->id; }

  void set_name(std::string &name_in) { this->name = name_in; }

  void set_mass(REAL &mass_in) { this->mass = mass_in; }

  void set_charge(REAL &charge_in) { this->charge = charge_in; }

  void set_id(REAL &id_in) { this->id = id_in; }

private:
  std::string name;
  REAL mass;
  REAL charge;
  INT id;
};

template <typename PROP_TYPE> struct RequiredProperties {
  RequiredProperties(
      std::vector<int> required_simple_props_, // simple_props (including
                                                 // fluid_density for example)
      std::vector<Species> species_,
      std::vector<int> required_species_props_) // species_props
      : required_simple_props(required_simple_props_), species(species_),
        required_species_props(required_species_props_){};

  std::vector<std::string> required_simple_prop_names() {
    if (this->required_simple_props.empty()) {
      throw std::logic_error("No required_simple_props have been defined.");
    }

    std::vector<std::string> required_simple_prop_names_vec;
    for (auto req_prop : this->required_simple_props) {
      required_simple_prop_names_vec.push_back(
          std::string(default_map.at(req_prop)));
    }

    return required_simple_prop_names_vec;
  }

  int required_simple_prop_index(int prop) {
    int prop_index = 0;
    for (auto req_prop : this->required_simple_props) {
      if (req_prop == prop) {
        return prop_index;
      }
      ++prop_index;
    }
    std::string index_error_msg = default_map.at(prop) + " property not found in required_simple_props.";
    throw std::logic_error(index_error_msg);
  }

  std::vector<std::string> required_species_prop_names() {
    if (this->species.empty() || this->required_species_props.empty()) {
      throw std::logic_error("No species and/or required_species_props have been defined.");
    }

    std::vector<std::string> required_species_real_prop_names_vec;
    for (auto i_species : this->species) {
      std::string species_name = i_species.get_name();
      for (auto req_prop : this->required_species_props) {
        required_species_real_prop_names_vec.push_back(
            species_name + "_" + std::string(default_map.at(req_prop)));
      }
    }

    return required_species_real_prop_names_vec;
  }

  int required_species_prop_index(std::string species_name, int prop) {
    int prop_index = 0;
    for (auto req_prop : this->required_species_props) {
      if (req_prop == prop) {
        break;
      }
      ++prop_index;
    }

    int species_index = 0;
    for (auto i_species : this->species) {
      if (i_species.get_name() == species_name) {
        break;
      }
      ++species_index;
    }

    std::string index_error_msg = "";
    bool index_error = false;

    if (prop_index == this->required_species_props.size()) {
      index_error_msg += default_map.at(prop) + " property not found in required_species_props.";
      index_error = true;
    }

    if (species_index == this->species.size()) {
      index_error_msg += species_name + " not found in species.";
      index_error = true;
    }

    if (index_error) {
      throw std::logic_error(index_error_msg);
    }

    return prop_index + species_index * this->species.size();
  }

  const std::vector<int> get_required_simple_props() const {
    return this->required_simple_props;
  }

  const std::vector<Species> get_species() const { return this->species; }

  const std::vector<int> get_required_species_props() const {
    return this->required_species_props;
  }

private:
  std::vector<int> required_simple_props;
  std::vector<Species> species;
  std::vector<int> required_species_props;
};