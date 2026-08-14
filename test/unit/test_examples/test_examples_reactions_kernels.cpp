#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

#include "../example_sources/example_electron_impact_ion.hpp"
#include "../example_sources/example_general_absorption_kernels.hpp"
#include "../example_sources/example_general_linear_scattering_kernels.hpp"
#include "../example_sources/example_ionisation_kernels.hpp"
#include "../example_sources/example_recombination_kernels.hpp"

TEST(Examples, kernels) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  ionisation_kernels_example();
  electron_impact_ion_example(particle_group);
  recombination_kernels_example();
  general_absorption_kernels_example();
  general_linear_scattering_kernels_example(particle_group);

  particle_group->domain->mesh->free();
}
