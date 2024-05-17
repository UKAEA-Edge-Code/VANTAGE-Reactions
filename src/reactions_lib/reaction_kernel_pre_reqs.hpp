#pragma once
#include "particle_properties_map.hpp"
#include <array>

using namespace ParticlePropertiesIndices;

template <int num_species> struct Species {
  Species(std::array<std::string, num_species> names_) : names(names_){};

  const std::array<std::string, num_species> get_species_names() const {
    return names;
  }

private:
  const std::array<std::string, num_species> names;
};

template <int num_required_props> struct RequiredProperties {
  RequiredProperties(std::array<int, num_required_props> required_props_)
      : required_props(required_props_){};

  const std::array<int, num_required_props> get_req_props() {
    return required_props;
  }

private:
  std::array<int, num_required_props> required_props;
};

template <int num_required_particle_real_props>
struct RequiredParticleRealProperties
    : public RequiredProperties<num_required_particle_real_props> {
  RequiredParticleRealProperties(
      std::array<int, num_required_particle_real_props>
          required_particle_real_props_)
      : RequiredProperties<num_required_particle_real_props>(
            required_particle_real_props_) {
    this->required_particle_real_props = this->get_req_props();
  }

  std::array<std::string, num_required_particle_real_props>
  required_particle_real_prop_names() {
    int i = 0;
    std::array<std::string, num_required_particle_real_props>
        required_prop_names_arr = {};
    for (auto req_prop : this->required_particle_real_props) {
      required_prop_names_arr[i++] = default_map.at(req_prop);
    }
    return required_prop_names_arr;
  }

  int required_particle_real_index(int req_prop) {
    int i = 0;
    for (auto ireq_prop : required_particle_real_props) {
      if (ireq_prop == req_prop) {
        return i;
      }
      ++i;
    }
    return -1;
  }

private:
  std::array<int, num_required_particle_real_props>
      required_particle_real_props;
};

template <int num_species, int num_field_real_props>
struct RequiredSpeciesFieldProperties
    : public Species<num_species>,
      RequiredProperties<num_field_real_props> {
  RequiredSpeciesFieldProperties(
      std::array<std::string, num_species> names_,
      std::array<int, num_field_real_props> required_field_real_props_)
      : Species<num_species>(names_),
        RequiredProperties<num_field_real_props>(required_field_real_props_) {
    this->names = this->get_species_names();
    this->required_field_real_props = this->get_req_props();
  }

public:
  std::array<std::string, num_species * num_field_real_props>
  required_species_field_real_prop_names() {
    int species_count = 0;
    int prop_count = 0;
    std::array<std::string, num_species *num_field_real_props>
        required_prop_names_arr = {};
    for (auto species_name : this->names) {
      prop_count = 0;
      for (auto req_prop : this->required_field_real_props) {
        required_prop_names_arr[species_count * num_field_real_props + prop_count] =
            species_name + "_" + std::string(default_map.at(req_prop));
        ++prop_count;
      }
      ++species_count;
    }

    return required_prop_names_arr;
  }

  int required_species_field_real_index(std::string species_name, int prop) {
    int prop_index = 0;
    int species_index = 0;
    for (auto req_prop : this->required_field_real_props) {
      if (req_prop == prop) break;
      ++prop_index;
    }

    for (auto name : this->names) {
      if (name == species_name) break;
      ++species_index;
    }

    return prop_index + species_index * num_field_real_props;
  };

private:
  std::array<std::string, num_species> names;
  std::array<int, num_field_real_props> required_field_real_props;
};