#pragma once
#include "particle_properties_map.hpp"
#include <array>
#include <neso_particles.hpp>
#include <stdexcept>
#include <vector>

using namespace NESO::Particles;
using namespace ParticlePropertiesIndices;

/**
 * @brief Species struct to hold a limited description of a species that may be
 * used in reactions.
 *
 * @param name String defining the name of the species.
 * @param mass REAL value of the mass of the species (in atomic units).
 * @param charge REAL value of the charge of the species (in atomic units).
 * @param id INT value that corresponds to the ID of the species.
 */
struct Species {
  Species(std::string name_) : name(name_){};

  Species(std::string name_, REAL mass_) : name(), mass(mass_){};

  Species(std::string name_, REAL mass_, REAL charge_)
      : name(name_), mass(mass_), charge(charge_){};

  Species(std::string name_, REAL mass_, REAL charge_, INT id_)
      : name(name_), mass(mass_), charge(charge_), id(id_){};

public:
  /**
   * @brief Getters and setters for name, mass, charge and id of the species.
   */
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

/**
 * @brief Struct for defining the RequiredProperties that a ReactionData or
 * ReactionKernel object might need.
 *
 * @tparam PROP_TYPE Property type of the properties to be stored in this struct
 * (either INT or REAL).
 *
 * @param required_simple_props_ An integer vector defining the required simple
 * properties (either particle or field properties that don't depend on
 * species). The values in the vector will be enums from the
 * ParticlePropertiesIndices namespace.
 * @param species_ A vector of Species structs that contain the species(plural)
 * that the required_species_props_ need to be combined with in order to produce
 * the correct property names.
 * @param required_species_props_ An integer vector defining the required
 * species properties that are to be combined with species_ to produce property
 * names.
 */
template <typename PROP_TYPE> struct RequiredProperties {
  RequiredProperties(
      std::vector<int> required_simple_props_, // simple_props (including
                                               // fluid_density for example)
      std::vector<Species> species_,
      std::vector<int> required_species_props_) // species_props
      : required_simple_props(required_simple_props_), species(species_),
        required_species_props(required_species_props_){};

  /**
   * @brief Function to return a vector of strings containing the names of the
   * required simple properties.
   *
   * @return required_simple_prop_names
   */
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

  /**
   * @brief Function that return the index of the property in
   * required_simple_prop_names given a requested property.
   *
   * @param prop An integer that corresponds to a value from the enumerator in
   * ParticlePropertiesIndices (eg. for "VELOCITY" this would be the variable
   * name - velocity - which corresponds to 1.)
   *
   * @return required_simple_prop_index
   */
  int required_simple_prop_index(int prop) {
    int prop_index = 0;
    for (auto req_prop : this->required_simple_props) {
      if (req_prop == prop) {
        return prop_index;
      }
      ++prop_index;
    }
    std::string index_error_msg =
        default_map.at(prop) + " property not found in required_simple_props.";
    throw std::logic_error(index_error_msg);
  }

  /**
   * @brief Function to return a vector of strings containing the names of the
   * required species props combined with the species as a prefix. (eg.
   * "ELECTRON" + "_" + "DENSITY")
   *
   * @return required_species_prop_names
   */
  std::vector<std::string> required_species_prop_names() {
    if (this->species.empty() || this->required_species_props.empty()) {
      throw std::logic_error(
          "No species and/or required_species_props have been defined.");
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

  /**
   * @brief Function that returns the index of the property in
   * required_species_prop_names given a species name and a requested property.
   *
   * @param species_name Requested species name (eg. "ELECTRON")
   * @param prop An integer that corresponds to a value from the enumerator in
   * ParticlePropertiesIndices (eg. for "DENSITY" this would be the variable
   * name - density - which corresponds to 8).
   *
   * @return required_species_prop_index
   */
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
      index_error_msg += default_map.at(prop) +
                         " property not found in required_species_props.";
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

  /**
   * @brief Getters for required_simple_props, species and
   * required_species_props.
   */
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