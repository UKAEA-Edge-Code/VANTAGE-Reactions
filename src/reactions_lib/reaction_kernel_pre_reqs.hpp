#pragma once
#include "particle_properties_map.hpp"
#include <iterator>
#include <neso_particles.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <strings.h>
#include <vector>

using namespace NESO::Particles;
namespace Reactions {
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
  Species() = default;

  Species(std::string name_) : name(name_){};

  Species(std::string name_, REAL mass_) : name(name_), mass(mass_){};

  Species(std::string name_, REAL mass_, REAL charge_)
      : name(name_), mass(mass_), charge(charge_){};

  Species(std::string name_, REAL mass_, REAL charge_, INT id_)
      : name(name_), mass(mass_), charge(charge_), id(id_){};

public:
  /**
   * @brief Getters and setters for name, mass, charge and id of the species.
   */
  std::string get_name() const {
    NESOASSERT(this->name.has_value(),
               "The member variable: Species.name has not been assigned");
    return (this->name.value());
  }

  /**
   * @brief Return true if this species has an id associated to it
   *
   * @return True it this species ahs an id associated to it
   */
  bool has_id() const { return this->id.has_value(); }

  INT get_id() const {
    NESOASSERT(this->id.has_value(),
               "The member variable: Species.id has not been assigned");
    return (this->id.value());
  }

  REAL get_mass() const {
    NESOASSERT(this->mass.has_value(),
               "The member variable: Species.mass has not been assigned");
    return (this->mass.value());
  }

  REAL get_charge() const {
    NESOASSERT(this->charge.has_value(),
               "The member variable: Species.charge has not been assigned");
    return (this->charge.value());
  }

  void set_name(const std::string &name_in) { this->name = name_in; }

  void set_id(const INT &id_in) { this->id = id_in; }

  void set_mass(const REAL &mass_in) { this->mass = mass_in; }

  void set_charge(const REAL &charge_in) { this->charge = charge_in; }

private:
  std::optional<std::string> name;
  std::optional<INT> id;
  std::optional<REAL> mass;
  std::optional<REAL> charge;
};

//TODO: Make this more robust 
inline bool operator==(const Species &lhs, const Species &rhs) {

  if (lhs.has_id() && rhs.has_id()) {

    return lhs.get_name() == rhs.get_name() && lhs.get_id() == rhs.get_id();
  }

  return lhs.get_name() == rhs.get_name() && lhs.has_id() == rhs.has_id();
}

/**
 * @brief Struct for defining the Properties that a ReactionData or
 * ReactionKernel object might need.
 *
 * @tparam PROP_TYPE Property type of the properties to be stored in this struct
 * (either INT or REAL).
 *
 * @param simple_props_ An integer vector defining the required simple
 * properties (either particle or field properties that don't depend on
 * species). The values in the vector will be enums from the
 * ParticlePropertiesIndices namespace.
 * @param species_ A vector of Species structs that contain the species(plural)
 * that the species_props_ need to be combined with in order to produce
 * the correct property names.
 * @param species_props_ An integer vector defining the required
 * species properties that are to be combined with species_ to produce property
 * names.
 */
template <typename PROP_TYPE> struct Properties {
  Properties() = default;

  Properties(std::vector<int> simple_props_, // simple_props (including
                                             // fluid_density for example)
             std::vector<Species> species_,
             std::vector<int> species_props_) // species_props
      : simple_props(simple_props_), species(species_),
        species_props(species_props_) {
    this->all_props = this->simple_props;
    this->all_props.insert(this->all_props.end(), this->species_props.begin(),
                           this->species_props.end());
  };

  Properties(std::vector<int> simple_props_)
      : Properties(simple_props_, std::vector<Species>{}, std::vector<int>{}){};

  Properties(std::vector<Species> species_, std::vector<int> species_props_)
      : Properties(std::vector<int>{}, species_, species_props_){};

  template <size_t N, size_t M>
  Properties(
      const std::array<int, N> &simple_props_, // simple_props (including
                                               // fluid_density for example)
      std::vector<Species> species_,
      const std::array<int, M> &species_props_) // species_props
      : simple_props(
            std::vector<int>(simple_props_.begin(), simple_props_.end())),
        species(species_), species_props(std::vector<int>(
                               species_props_.begin(), species_props_.end())) {
    this->all_props = this->simple_props;
    this->all_props.insert(this->all_props.end(), this->species_props.begin(),
                           this->species_props.end());
  };

  template <size_t N>
  Properties(const std::array<int, N> &simple_props_)
      : Properties(simple_props_, std::vector<Species>{},
                   std::array<int, 0>{}){};

  template <size_t M>
  Properties(std::vector<Species> species_,
             const std::array<int, M> &species_props_)
      : Properties(std::array<int, 0>{}, species_, species_props_){};

  /**
   * @brief Merge with another property, taking care of duplicates. The
   * properties of this object are inserted first.
   *
   * @param other The Properties object to merge with
   */
  Properties<PROP_TYPE> merge_with(Properties<PROP_TYPE> other) {

    auto new_simple_props = this->simple_props;

    for (auto other_prop : other.simple_props) {

      auto it = std::find(new_simple_props.begin(), new_simple_props.end(),
                          other_prop);

      if (it == new_simple_props.end()) {

        new_simple_props.push_back(other_prop);
      }
    }

    auto new_species_props = this->species_props;

    for (auto other_prop : other.species_props) {

      auto it = std::find(new_species_props.begin(), new_species_props.end(),
                          other_prop);

      if (it == new_species_props.end()) {

        new_species_props.push_back(other_prop);
      }
    }

    auto new_species = this->species;

    for (auto other_species : other.species) {
      auto it =
          std::find(new_species.begin(), new_species.end(), other_species);

      if (it == new_species.end()) {

        new_species.push_back(other_species);
      }
    }
    return Properties<PROP_TYPE>(new_simple_props, new_species,
                                 new_species_props);
  }

  /**
   * @brief Function to return a vector of strings containing the names of the
   * required simple properties.
   *
   * @param properties_map_ A std::map<int, std::string> object to be used in
   * recovering the property names.
   *
   * @return simple_prop_names
   */
  std::vector<std::string> simple_prop_names(
      const std::map<int, std::string> &properties_map = default_map) {

    std::vector<std::string> simple_prop_names_vec;
    for (auto req_prop : this->simple_props) {
      std::string error_msg =
          "The property referenced by index: " + std::to_string(req_prop) +
          ", is not present in the property map provided";
      NESOASSERT(properties_map.find(req_prop) != properties_map.end(),
                 error_msg);
      simple_prop_names_vec.push_back(std::string(properties_map.at(req_prop)));
    }

    return simple_prop_names_vec;
  }

  /**
   * @brief Function that return the index of the property in
   * all_props given a requested property.
   *
   * @param prop An integer that corresponds to a value from the enumerator in
   * ParticlePropertiesIndices (eg. for "VELOCITY" this would be the variable
   * name - velocity - which corresponds to 1.)
   * @param properties_map_ A std::map<int, std::string> object to be used in
   * recovering the property indices.
   *
   * @return simple_prop_index
   */
  int simple_prop_index(
      int prop,
      const std::map<int, std::string> &properties_map = default_map) {
    int prop_index = 0;
    for (auto req_prop : this->all_props) {
      if (req_prop == prop) {
        return prop_index;
      }
      ++prop_index;
    }
    std::string index_error_msg =
        properties_map.at(prop) + " property not found in simple_props.";
    NESOASSERT(false, index_error_msg);
  }

  /**
   * @brief Function to return a vector of strings containing the names of the
   * required species props combined with the species as a prefix. (eg.
   * "ELECTRON" + "_" + "DENSITY")
   *
   * @param properties_map_ A std::map<int, std::string> object to be used in
   * recovering the property names.
   *
   * @return species_prop_names
   */
  std::vector<std::string> species_prop_names(
      const std::map<int, std::string> &properties_map = default_map) {
    std::vector<std::string> species_real_prop_names_vec;

    for (auto i_species : this->species) {
      std::string species_name = i_species.get_name();
      for (auto req_prop : this->species_props) {
        species_real_prop_names_vec.push_back(
            species_name + "_" + std::string(properties_map.at(req_prop)));
      }
    }

    return species_real_prop_names_vec;
  }

  /**
   * @brief Function that returns the index of the property in
   * all_props given a species name and a requested property.
   *
   * @param species_name Requested species name (eg. "ELECTRON")
   * @param prop An integer that corresponds to a value from the enumerator in
   * ParticlePropertiesIndices (eg. for "DENSITY" this would be the variable
   * name - density - which corresponds to 8).
   * @param properties_map_ A std::map<int, std::string> object to be used in
   * recovering the property indices.
   *
   * @return species_prop_index
   */
  int species_prop_index(
      std::string species_name, int prop,
      const std::map<int, std::string> &properties_map = default_map) {
    int prop_index = 0;
    for (auto req_prop : this->all_props) {
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

    if (prop_index == this->all_props.size()) {
      index_error_msg +=
          properties_map.at(prop) + " property not found in all_props.";
      index_error = true;
    }

    if (species_index == this->species.size()) {
      index_error_msg += species_name + " not found in species.";
      index_error = true;
    }

    NESOASSERT(not index_error, index_error_msg);

    return prop_index + species_index * this->species_props.size();
  }

  /**
   * @brief Getter for combined prop_names vector
   *
   * @param properties_map_ A std::map<int, std::string> object to be used in
   * recovering the property names.
   */
  const std::vector<std::string> get_prop_names(
      const std::map<int, std::string> &properties_map = default_map) {
    std::vector<std::string> simple_prop_names;
    std::vector<std::string> species_props_names;

    simple_prop_names = this->simple_prop_names(properties_map);

    species_props_names = this->species_prop_names(properties_map);

    std::vector<std::string> comb_prop_names;
    comb_prop_names.insert(comb_prop_names.end(), simple_prop_names.begin(),
                           simple_prop_names.end());
    comb_prop_names.insert(comb_prop_names.end(), species_props_names.begin(),
                           species_props_names.end());

    return comb_prop_names;
  }

  /**
   * @brief Getters for simple_props, species and
   * species_props and all props.
   */
  const std::vector<int> get_simple_props() const { return this->simple_props; }

  const std::vector<Species> get_species() const { return this->species; }

  const std::vector<int> get_species_props() const {
    return this->species_props;
  }

  const std::vector<int> get_props() const { return this->all_props; }

private:
  std::vector<int> simple_props;
  std::vector<Species> species;
  std::vector<int> species_props;
  std::vector<int> all_props;
};
}; // namespace Reactions
