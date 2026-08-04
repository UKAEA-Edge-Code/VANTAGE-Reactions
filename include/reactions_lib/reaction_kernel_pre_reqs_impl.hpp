#ifndef VANTAGE_REACTIONS_REACTION_KERNEL_PRE_REQS_IMPL_H
#define VANTAGE_REACTIONS_REACTION_KERNEL_PRE_REQS_IMPL_H

#include "reaction_kernel_pre_reqs.hpp"

namespace VANTAGE::Reactions {

Species::Species(std::string name, REAL mass, REAL charge, INT id)
    : name(name), mass(mass), charge(charge), id(id) {}

Species::Species(std::string name) : name(name) {}

Species::Species(std::string name, REAL mass) : name(name), mass(mass) {}

Species::Species(std::string name, REAL mass, REAL charge)
    : name(name), mass(mass), charge(charge) {}

std::string Species::get_name() const {
  NESOASSERT(this->name.has_value(),
             "The member variable: Species.name has not been assigned");
  return (this->name.value());
}

INT Species::get_id() const {
  NESOASSERT(this->id.has_value(),
             "The member variable: Species.id has not been assigned");
  return (this->id.value());
}

REAL Species::get_mass() const {
  NESOASSERT(this->mass.has_value(),
             "The member variable: Species.mass has not been assigned");
  return (this->mass.value());
}

REAL Species::get_charge() const {
  NESOASSERT(this->charge.has_value(),
             "The member variable: Species.charge has not been assigned");
  return (this->charge.value());
}

void Species::set_name(const std::string &name) { this->name = name; }

void Species::set_id(const INT &id) { this->id = id; }

void Species::set_mass(const REAL &mass) { this->mass = mass; }

void Species::set_charge(const REAL &charge) { this->charge = charge; }

bool Species::has_id() const { return this->id.has_value(); }

bool operator==(const Species &lhs, const Species &rhs) {
  if (lhs.has_id() && rhs.has_id()) {
    return lhs.get_name() == rhs.get_name() && lhs.get_id() == rhs.get_id();
  }
  return lhs.get_name() == rhs.get_name() && lhs.has_id() == rhs.has_id();
}

std::string species_property(const Species &species,
                             const std::string &property) {
  return species.get_name() + "_" + property;
}

} // namespace VANTAGE::Reactions

#endif
