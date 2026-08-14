#ifndef REACTIONS_PARTICLE_SPEC_BUILDER
#define REACTIONS_PARTICLE_SPEC_BUILDER

#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {
/**
 * @brief Helper struct to build custom particle specs based on user provided
 * particle properties (or if necessary extend existing particle specs.)
 */

struct ParticleSpecBuilder {
  ParticleSpecBuilder() = delete;

  /**
   * @brief Constructor for ParticleSpecBuilder.
   *
   * @param particle_spec ParticleSpec that is to be extended (optional pass via
   * a non-recommended constructor for ParticleSpecBuilder).
   */
  ParticleSpecBuilder(ParticleSpec particle_spec);

  /**
   * \overload
   * @brief Recommended constructor, populating the generally required
   * properties in Reactions
   *
   * @param ndim Dimensionality of vector quantities
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  ParticleSpecBuilder(
      int ndim,
      const std::map<int, std::string> &properties_map = get_default_map());

  /**
   * @brief Method to add particle properties to member particle_spec.
   *
   * @tparam PROP_TYPE Specifier for type of property (INT or REAL)
   * @param properties Properties object containing names of the particle
   * properties to be added.
   * @param ndim Number of dimensions for the properties to be added (note this
   * will apply to all properties from properties_)
   * @param positions Boolean to indicate whether the properties to be added are
   * particle position or cell id or not.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  template <typename PROP_TYPE>
  void add_particle_prop(
      Properties<PROP_TYPE> properties, int ndim = 1, bool positions = false,
      const std::map<int, std::string> &properties_map = get_default_map()) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

    auto simple_prop_names = properties.simple_prop_names(properties_map);

    auto species_prop_names = properties.species_prop_names(properties_map);

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
  void add_particle_spec(ParticleSpec new_particle_spec);

  const ParticleSpec &get_particle_spec();

private:
  ParticleSpec particle_spec;
};
} // namespace VANTAGE::Reactions

#endif
