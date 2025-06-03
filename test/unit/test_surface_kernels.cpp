#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include "reactions_lib/reaction_data/fixed_rate_data.hpp"
#include "reactions_lib/reaction_kernels/specular_reflection_kernels.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(SurfaceKernels, SpecularReflection) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_data = FixedRateData(1.0);

  auto test_kernels = SpecularReflectionKernels<2>();

  auto test_reaction =
      LinearReactionBase<0, FixedRateData, SpecularReflectionKernels<2>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, test_data,
          test_kernels);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  // Add data to subgroup
  particle_sub_group->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_normal, 2);

  particle_loop(
      "set_ephemeral_data_specular_reflection", particle_sub_group,
      [=](auto normal, auto velocity) {
        normal.at_ephemeral(0) = 1.0;
        velocity.at(0) = -1.0;
        velocity.at(1) = 1.0;
      },
      Access::write(BoundaryInteractionSpecification::intersection_normal),
      Access::write(Sym<REAL>("VELOCITY")))
      ->execute();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    test_reaction.descendant_product_loop(
        particle_sub_group, i, i + 1, 0.1, descendant_particles,
        true); // Apply to all of weight, the same way a surface reaction
               // controller would
    auto velocity = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    const int nrow = velocity->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(velocity->at(rowx, 0), 1.0);
      EXPECT_DOUBLE_EQ(velocity->at(rowx, 1), 1.0);
    }
  }

  particle_group->domain->mesh->free();
}
