#include "../include/mock_particle_group.hpp"
#include "../include/test_common.hpp"

using namespace VANTAGE::Reactions;

#include "../example_sources/example_cartesian_basis_reflection.hpp"
#include "../example_sources/example_spherical_basis_reflection.hpp"

TEST(Examples, basis_reflection) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  cartesian_basis_reflection_example();
  spherical_basis_reflection_example();

  particle_group->domain->mesh->free();
}
