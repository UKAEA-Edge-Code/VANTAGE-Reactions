#ifndef REACTIONS_REACTION_KERNEL_PRE_REQS_H
#define REACTIONS_REACTION_KERNEL_PRE_REQS_H
#include "particle_properties_map.hpp"
#include <iterator>
#include <neso_particles.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <strings.h>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
/**
 * @brief Species struct to hold a limited description of a species that may be
 * used in reactions.
 */
struct Species {
  Species() = default;

  /**
   * @brief Constructor for Species.
   *
   * @param name String defining the name of the species.
   * @param mass REAL value of the mass of the species (in atomic units).
   * @param charge REAL value of the charge of the species (in atomic units).
   * @param id INT value that corresponds to the ID of the species.
   */
  Species(std::string name, REAL mass, REAL charge, INT id)
      : name(name), mass(mass), charge(charge), id(id){};

  /**
   * \overload
   * @brief Constructor for Species that only sets name.
   *
   * @param name String defining the name of the species.
   */
  Species(std::string name) : name(name){};

  /**
   * \overload
   * @brief Constructor for Species that only sets name and mass.
   *
   * @param name String defining the name of the species.
   * @param mass REAL value of the mass of the species (in atomic units).
   */
  Species(std::string name, REAL mass) : name(name), mass(mass){};

  /**
   * \overload
   * @brief Constructor for Species that only sets name, mass and charge.
   *
   * @param name String defining the name of the species.
   * @param mass REAL value of the mass of the species (in atomic units).
   * @param charge REAL value of the charge of the species (in atomic units).
   */
  Species(std::string name, REAL mass, REAL charge)
      : name(name), mass(mass), charge(charge){};

public:
  /**
   * @brief Getters and setters for name, mass, charge and id of the species.
   */
  std::string get_name() const {
    NESOASSERT(this->name.has_value(),
               "The member variable: Species.name has not been assigned");
    return (this->name.value());
  }

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

  void set_name(const std::string &name) { this->name = name; }

  void set_id(const INT &id) { this->id = id; }

  void set_mass(const REAL &mass) { this->mass = mass; }

  void set_charge(const REAL &charge) { this->charge = charge; }

  /**
   * @brief Return true if this species has an id associated with it
   *
   * @return True if this species has an id associated with it
   */
  bool has_id() const { return this->id.has_value(); }

private:
  std::optional<std::string> name;
  std::optional<INT> id;
  std::optional<REAL> mass;
  std::optional<REAL> charge;
};

// TODO: Make this more robust
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
 */
template <typename PROP_TYPE> struct Properties {
  Properties() = default;

  /**
   * @brief Constructor for Properties.
   *
   * @param simple_props An integer vector defining the required simple
   * properties (either particle or field properties that don't depend on
   * species). The values in the vector will be enums from a
   * StandardPropertiesEnum (or derived) struct.
   * @param species A vector of Species structs that contain the species(plural)
   * that the species_props_ need to be combined with in order to produce
   * the correct property names.
   * @param species_props An integer vector defining the required
   * species properties that are to be combined with species_ to produce
   * property names.
   */
  Properties(std::vector<int> simple_props, // simple_props (including
                                            // fluid_density for example)
             std::vector<Species> species,
             std::vector<int> species_props) // species_props
      : simple_props(simple_props), species(species),
        species_props(species_props) {
    this->all_props = this->simple_props;
    this->all_props.insert(this->all_props.end(), this->species_props.begin(),
                           this->species_props.end());
  };

  /**
   * \overload
   * @brief Constructor for Properties that only sets the simple props.
   *
   * @param simple_props An integer vector defining the required simple
   * properties (either particle or field properties that don't depend on
   * species). The values in the vector will be enums from a
   * StandardPropertiesEnum (or derived) struct.
   */
  Properties(std::vector<int> simple_props)
      : Properties(simple_props, std::vector<Species>{}, std::vector<int>{}){};

  /**
   * \overload
   * @brief Constructor for Properties that only sets Species props.
   *
   * @param species A vector of Species structs that contain the species(plural)
   * that the species_props_ need to be combined with in order to produce
   * the correct property names.
   * @param species_props An integer vector defining the required
   * species properties that are to be combined with species_ to produce
   * property names.
   */
  Properties(std::vector<Species> species, std::vector<int> species_props)
      : Properties(std::vector<int>{}, species, species_props){};

  /**
   * \overload
   * @brief Constructor for Properties that uses std::arrays instead of
   * std::vectors for the props.
   *
   * @tparam N Size of simple props array.
   * @tparam M Size of species props array.
   *
   * @param simple_props An integer array defining the required simple
   * properties (either particle or field properties that don't depend on
   * species). The values in the array will be enums from a
   * StandardPropertiesEnum (or derived) struct.
   * @param species A vector of Species structs that contain the species(plural)
   * that the species_props_ need to be combined with in order to produce
   * the correct property names.
   * @param species_props An integer array defining the required
   * species properties that are to be combined with species_ to produce
   * property names.
   */
  template <size_t N, size_t M>
  Properties(
      const std::array<int, N> &simple_props, // simple_props (including
                                              // fluid_density for example)
      std::vector<Species> species,
      const std::array<int, M> &species_props) // species_props
      : simple_props(
            std::vector<int>(simple_props.begin(), simple_props.end())),
        species(species), species_props(std::vector<int>(species_props.begin(),
                                                         species_props.end())) {
    this->all_props = this->simple_props;
    this->all_props.insert(this->all_props.end(), this->species_props.begin(),
                           this->species_props.end());
  };

  /**
   * \overload
   * @brief Constructor for Properties that only sets the simple props using
   * std::array.
   *
   * @tparam N Size of simple props array.
   *
   * @param simple_props An integer array defining the required simple
   * properties (either particle or field properties that don't depend on
   * species). The values in the array will be enums from a
   * StandardPropertiesEnum (or derived) struct.
   */
  template <size_t N>
  Properties(const std::array<int, N> &simple_props)
      : Properties(simple_props, std::vector<Species>{},
                   std::array<int, 0>{}){};

  /**
   * \overload
   * @brief Constructor for Properties that only sets Species props using
   * std::array.
   *
   * @tparam M Size of species props array.
   *
   * @param species A vector of Species structs that contain the species(plural)
   * that the species_props_ need to be combined with in order to produce
   * the correct property names.
   * @param species_props An integer array defining the required
   * species properties that are to be combined with species_ to produce
   * property names.
   */
  template <size_t M>
  Properties(std::vector<Species> species,
             const std::array<int, M> &species_props)
      : Properties(std::array<int, 0>{}, species, species_props){};

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
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used in remapping the property names.
   *
   * @return simple_prop_names
   */
  std::vector<std::string> simple_prop_names(
      const std::map<int, std::string> &properties_map = get_default_map()) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

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
   * a StandardPropertiesEnum (or derived) struct (eg. for "VELOCITY" this would
   * be the variable name - velocity - which corresponds to 1.)
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used in remapping the property indices.
   *
   * @return simple_prop_index
   */
  int simple_prop_index(
      int prop,
      const std::map<int, std::string> &properties_map = get_default_map()) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

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
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used in remapping the property names.
   *
   * @return species_prop_names
   */
  std::vector<std::string> species_prop_names(
      const std::map<int, std::string> &properties_map = get_default_map()) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

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
   * a StandardPropertiesEnum (or derived) struct (eg. for "DENSITY" this would
   * be the variable name - density - which corresponds to 8).
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used in remapping the property indices.
   *
   * @return species_prop_index
   */
  int species_prop_index(
      std::string species_name, int prop,
      const std::map<int, std::string> &properties_map = get_default_map()) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

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
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used in remapping the property names.
   */
  const std::vector<std::string> get_prop_names(
      const std::map<int, std::string> &properties_map = get_default_map()) {
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
}; // namespace VANTAGE::Reactions
#endif