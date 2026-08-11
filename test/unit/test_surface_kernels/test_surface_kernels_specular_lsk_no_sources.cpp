
#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include "reactions_lib/reaction_data/fixed_rate_data.hpp"
#include "reactions_lib/reaction_data/specular_reflection_data.hpp"
#include "reactions_lib/reaction_kernels/general_linear_scattering_kernels.hpp"
#include "reactions_lib/reaction_kernels/specular_reflection_kernels.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <neso_particles/boundary/boundary_interaction_specification.hpp>

using namespace VANTAGE::Reactions;

TEST(SurfaceKernels, SpecularReflection_LinearScatteringKernels_NoSources) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  particle_group->add_particle_dat(
      NP::BoundaryInteractionSpecification::intersection_normal, 2);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto test_data = FixedRateData(1.0);

  auto velocity_data = ExtractorData<2>(NP::Sym<NP::REAL>("VELOCITY"));

  auto specular_reflection = SpecularReflectionData<2>();

  auto pipeline = PipelineData(velocity_data, specular_reflection);

  auto projectile_species = Species("ION", 1.2, 0.0, 0);
  auto test_kernels = LinearScatteringKernels<2, false>(projectile_species);

  auto data_calculator = DataCalculator<decltype(pipeline)>(pipeline);
  auto test_reaction =
      LinearReactionBase<1, FixedRateData, decltype(test_kernels),
                         decltype(data_calculator)>(
          particle_group->sycl_target, 0, std::array<int, 1>{0}, test_data,
          test_kernels, data_calculator);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  particle_loop(
      "set__data_specular_reflection", particle_sub_group,
      [=](auto normal, auto velocity) {
        normal.at(0) = 1.0;
        normal.at(1) = 0.0;
        velocity.at(0) = -1.0;
        velocity.at(1) = 1.0;
      },
      NP::Access::write(
          NP::BoundaryInteractionSpecification::intersection_normal),
      NP::Access::write(NP::Sym<NP::REAL>("VELOCITY")))
      ->execute();

  auto descendant_particles = std::make_shared<NP::ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto weight =
        descendant_particles->get_cell(NP::Sym<NP::REAL>("WEIGHT"), i);
    auto vel_parent =
        particle_group->get_cell(NP::Sym<NP::REAL>("VELOCITY"), i);
    auto vel_child =
        descendant_particles->get_cell(NP::Sym<NP::REAL>("VELOCITY"), i);

    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(vel_child->at(rowx, 0), 1.0);
      EXPECT_DOUBLE_EQ(vel_child->at(rowx, 1), 1.0);
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.1);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
