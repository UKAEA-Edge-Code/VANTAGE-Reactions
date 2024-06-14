#ifndef PARTICLE_SPEC_BUILDER
#define PARTICLE_SPEC_BUILDER

#include <algorithm>
#include <iterator>
#include <neso_particles.hpp>
#include <neso_particles/particle_spec.hpp>
#include <reaction_kernel_pre_reqs.hpp>

using namespace NESO::Particles;
using namespace ParticlePropertiesIndices;

namespace Reactions {

struct ParticleSpecBuilder {
  ParticleSpecBuilder() = default;

  ParticleSpecBuilder(const ParticleSpec &particle_spec_)
      : particle_spec(particle_spec_) {}

  template <typename PROP_TYPE>
  void add_particle_prop(std::string property_name, int ndim = 1,
                         bool positions = false) {
    particle_spec.push(
        ParticleProp(Sym<PROP_TYPE>(property_name), ndim, positions));
  }

  void add_particle_spec(const ParticleSpec &new_particle_spec) {
    auto existing_properties_real = this->particle_spec.properties_real;
    auto existing_properties_int = this->particle_spec.properties_int;

    auto new_properties_real = new_particle_spec.properties_real;
    auto new_properties_int = new_particle_spec.properties_int;

    existing_properties_real.insert(existing_properties_real.end(),
                                    new_properties_real.begin(),
                                    new_properties_real.end());

    existing_properties_int.insert(existing_properties_int.end(),
                                   new_properties_int.begin(),
                                   new_properties_int.end());

    this->particle_spec =
        ParticleSpec(existing_properties_real, existing_properties_int);
  }

  const ParticleSpec &get_particle_spec() { return particle_spec; }

private:
  ParticleSpec particle_spec;
};
} // namespace Reactions

#endif