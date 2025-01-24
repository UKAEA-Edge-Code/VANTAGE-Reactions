#ifndef PARTICLE_SPEC_BUILDER
#define PARTICLE_SPEC_BUILDER

#include <neso_particles.hpp>
#include "reaction_kernel_pre_reqs.hpp"

using namespace NESO::Particles;

namespace Reactions {
/**
 * @brief Helper struct to build custom particle specs based on user provided
 * particle properties (or if necessary extend existing particle specs.)
 *
 * @param particle_spec_ ParticleSpec that is to be extended (optional pass via
 * a non-default constructor for ParticleSpecBuilder).
 */

struct ParticleSpecBuilder {
  ParticleSpecBuilder() = default;

  ParticleSpecBuilder(ParticleSpec particle_spec_) {
    this->add_particle_spec(particle_spec_);
  }

  /**
   * @brief Method to add particle properties to member particle_spec.
   *
   * @tparam PROP_TYPE Specifier for type of property (INT or REAL)
   * @param properties_ Properties object containing names of the particle
   * properties to be added.
   * @param ndim Number of dimensions for the properties to be added (note this
   * will apply to all properties from properties_)
   * @param positions Boolean to indicate whether the properties to be added are
   * particle position or cell id or not.
   * @param properties_map Property map to be used when adding properties into the spec. Defaults to default_map.
   */
  template <typename PROP_TYPE>
  void add_particle_prop(Properties<PROP_TYPE> properties_, int ndim = 1,
                         bool positions = false, const std::map<int, std::string> &properties_map = default_map ) {
    std::vector<std::string> simple_prop_names;
    try {
      simple_prop_names = properties_.simple_prop_names(properties_map);
    } catch (std::logic_error) {
      simple_prop_names = {};
    }

    std::vector<std::string> species_prop_names;
    try {
      species_prop_names = properties_.species_prop_names(properties_map);
    } catch (std::logic_error) {
      species_prop_names = {};
    }

    for (auto prop_name : simple_prop_names) {
      auto particle_prop =
          ParticleProp(Sym<PROP_TYPE>(prop_name), ndim, positions);
      auto particle_spec_contains = this->particle_spec.contains(particle_prop);
      if (particle_spec_contains) {
        continue;
      } else {
        this->particle_spec.push(particle_prop);
      }
    }

    for (auto prop_name : species_prop_names) {
      auto particle_prop =
          ParticleProp(Sym<PROP_TYPE>(prop_name), ndim, positions);
      auto particle_spec_contains = this->particle_spec.contains(particle_prop);
      if (particle_spec_contains) {
        continue;
      } else {
        this->particle_spec.push(particle_prop);
      }
    }
  }

  /**
   * @brief Method to merge an existing ParticleSpec into the particle_spec
   * member inside the struct.
   *
   * @param new_particle_spec ParticleSpec to merge into internal particle_spec
   * member.
   */
  void add_particle_spec(ParticleSpec new_particle_spec) {
    auto existing_properties_real = this->particle_spec.properties_real;
    auto existing_properties_int = this->particle_spec.properties_int;

    std::vector<ParticleProp<REAL>> new_real_props;
    std::vector<ParticleProp<INT>> new_int_props;

    for (auto prop : new_particle_spec.properties_real) {
      if (this->particle_spec.contains(prop)) {
        continue;
      } else {
        new_real_props.push_back(prop);
      }
    }

    for (auto prop : new_particle_spec.properties_int) {
      if (this->particle_spec.contains(prop)) {
        continue;
      } else {
        new_int_props.push_back(prop);
      }
    }

    existing_properties_real.insert(existing_properties_real.end(),
                                    new_real_props.begin(),
                                    new_real_props.end());

    existing_properties_int.insert(existing_properties_int.end(),
                                   new_int_props.begin(), new_int_props.end());

    this->particle_spec =
        ParticleSpec(existing_properties_real, existing_properties_int);
  }

  const ParticleSpec &get_particle_spec() { return particle_spec; }

private:
  ParticleSpec particle_spec;
};
} // namespace Reactions

#endif
