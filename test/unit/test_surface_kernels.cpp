#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include "reactions_lib/reaction_data/fixed_rate_data.hpp"
#include "reactions_lib/reaction_data/specular_reflection_data.hpp"
#include "reactions_lib/reaction_kernels/general_linear_scattering_kernels.hpp"
#include "reactions_lib/reaction_kernels/specular_reflection_kernels.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <neso_particles/boundary/boundary_interaction_specification.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(SurfaceKernels, SpecularReflection) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  particle_group->add_particle_dat(
      BoundaryInteractionSpecification::intersection_normal, 2);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_data = FixedRateData(1.0);

  auto test_kernels = SpecularReflectionKernels<2>();

  auto test_reaction =
      LinearReactionBase<0, FixedRateData, SpecularReflectionKernels<2>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, test_data,
          test_kernels);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  particle_loop(
      "set__data_specular_reflection", particle_sub_group,
      [=](auto normal, auto velocity) {
        normal.at(0) = 1.0;
        normal.at(1) = 0.0;
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

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1, descendant_particles,
                        true); // Apply to all of weight, the same way a surface
                               // reaction controller would
    auto velocity = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    const int nrow = velocity->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(velocity->at(rowx, 0), 1.0);
      EXPECT_DOUBLE_EQ(velocity->at(rowx, 1), 1.0);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(SurfaceKernels, SpecularReflection_LinearScatteringKernels) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  particle_group->add_particle_dat(
      BoundaryInteractionSpecification::intersection_normal, 2);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_data = FixedRateData(1.0);

  auto velocity_data = ExtractorData<2>(Sym<REAL>("VELOCITY"));

  auto specular_reflection = SpecularReflectionData<2>();

  auto pipeline = PipelineData(velocity_data, specular_reflection);

  auto properties_map = PropertiesMap();
  properties_map[VANTAGE::Reactions::default_properties.source_momentum] =
      "ION_SOURCE_MOMENTUM";
  properties_map[VANTAGE::Reactions::default_properties.source_energy] =
      "ION_SOURCE_ENERGY";
  auto projectile_species = Species("ION", 1.2, 0.0, 0);
  auto test_kernels =
      LinearScatteringKernels<2>(projectile_species, properties_map.get_map());

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
      Access::write(BoundaryInteractionSpecification::intersection_normal),
      Access::write(Sym<REAL>("VELOCITY")))
      ->execute();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto weight = descendant_particles->get_cell(Sym<REAL>("WEIGHT"), i);
    auto vel_parent = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    auto vel_child = descendant_particles->get_cell(Sym<REAL>("VELOCITY"), i);

    auto source_momentum =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_MOMENTUM"), i);

    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_ENERGY"), i);
    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(vel_child->at(rowx, 0), 1.0);
      EXPECT_DOUBLE_EQ(vel_child->at(rowx, 1), 1.0);
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.1);

      EXPECT_DOUBLE_EQ(source_momentum->at(rowx, 0),
                       0.1 * 1.2 *
                           (vel_parent->at(rowx, 0) - vel_child->at(rowx, 0)));
      EXPECT_DOUBLE_EQ(source_momentum->at(rowx, 1),
                       0.1 * 1.2 *
                           (vel_parent->at(rowx, 1) - vel_child->at(rowx, 1)));
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       0.5 * 0.1 * 1.2 *
                           (std::pow(vel_parent->at(rowx, 0), 2) +
                            std::pow(vel_parent->at(rowx, 1), 2) -
                            std::pow(vel_child->at(rowx, 0), 2) -
                            std::pow(vel_child->at(rowx, 1), 2)));
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(SurfaceKernels, utils_get_basis) {

  std::array<REAL, 3> vel{1.0, 1.0, 1.0};
  std::array<REAL, 3> normal{1.0, 0.0, 0.0};

  auto basis = utils::get_normal_basis(vel, normal);

  std::array<REAL, 9> expected_basis{0,
                                     1 / std::sqrt(2),
                                     1 / std::sqrt(2),
                                     0,
                                     1 / std::sqrt(2),
                                     -1 / std::sqrt(2),
                                     -1.0,
                                     0,
                                     0};

  for (auto i = 0; i < 9; i++) {
    EXPECT_DOUBLE_EQ(basis[i], expected_basis[i]);
  }

  vel = std::array<REAL, 3>{1.0, 0, 1.0};
  normal = std::array<REAL, 3>{-1.0, 0.0, 0.0};

  basis = utils::get_normal_basis(vel, normal);

  expected_basis = std::array<REAL, 9>{0, 0, 1, 0, 1, 0, -1, 0, 0};

  for (auto i = 0; i < 9; i++) {
    EXPECT_DOUBLE_EQ(basis[i], expected_basis[i]);
  }
}

TEST(SurfaceKernels, utils_normal_to_cartesian) {

  std::array<REAL, 3> coords{2.0, M_PI / 4, 3 * M_PI / 4};

  std::array<REAL, 9> basis{0,
                            1 / std::sqrt(2),
                            1 / std::sqrt(2),
                            0,
                            1 / std::sqrt(2),
                            -1 / std::sqrt(2),
                            -1.0,
                            0,
                            0};

  auto new_coords = utils::normal_basis_to_cartesian(coords, basis);

  std::array<REAL, 3> expected_coords{-std::sqrt(2), 0, -std::sqrt(2)};

  for (auto i = 0; i < 3; i++) {
    EXPECT_NEAR(new_coords[i], expected_coords[i], 1e-15);
  }
}

TEST(SurfaceKernels, SphericalBasisReflectionData) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group<3>(N_total);
  particle_group->add_particle_dat(
      BoundaryInteractionSpecification::intersection_normal, 3);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_data = FixedRateData(1.0);

  std::array<REAL, 3> coords{2.0, M_PI / 4, 3 * M_PI / 4};
  auto coord_data = FixedArrayData<3>(coords);

  auto spherical_reflection = SphericalBasisReflectionData();

  auto pipeline = PipelineData(coord_data, spherical_reflection);

  auto properties_map = PropertiesMap();
  properties_map[VANTAGE::Reactions::default_properties.source_momentum] =
      "ION_SOURCE_MOMENTUM";
  properties_map[VANTAGE::Reactions::default_properties.source_energy] =
      "ION_SOURCE_ENERGY";
  auto projectile_species = Species("ION", 1.2, 0.0, 0);
  auto test_kernels =
      LinearScatteringKernels<3>(projectile_species, properties_map.get_map());

  auto data_calculator = DataCalculator<decltype(pipeline)>(pipeline);
  auto test_reaction =
      LinearReactionBase<1, FixedRateData, decltype(test_kernels),
                         decltype(data_calculator)>(
          particle_group->sycl_target, 0, std::array<int, 1>{0}, test_data,
          test_kernels, data_calculator);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto vel = std::array<REAL, 3>{1.0, 1.0, 1.0};
  auto norm = std::array<REAL, 3>{-1.0, 0.0, 0.0};
  particle_loop(
      "set__data_spherical_reflection", particle_sub_group,
      [=](auto normal, auto velocity) {
        for (auto i = 0; i < 3; i++) {

          normal.at(i) = norm[i];
          velocity.at(i) = vel[i];
        }
      },
      Access::write(BoundaryInteractionSpecification::intersection_normal),
      Access::write(Sym<REAL>("VELOCITY")))
      ->execute();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  std::array<REAL, 3> expected_vel{-std::sqrt(2), 0, -std::sqrt(2)};
  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto weight = descendant_particles->get_cell(Sym<REAL>("WEIGHT"), i);
    auto vel_parent = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    auto vel_child = descendant_particles->get_cell(Sym<REAL>("VELOCITY"), i);

    auto source_momentum =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_MOMENTUM"), i);

    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_ENERGY"), i);
    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.1);

      for (int dim = 0; dim < 3; dim++) {
        EXPECT_NEAR(vel_child->at(rowx, dim), expected_vel[dim], 1e-14);
        EXPECT_NEAR(source_momentum->at(rowx, dim),
                    0.1 * 1.2 *
                        (vel_parent->at(rowx, dim) - vel_child->at(rowx, dim)),
                    1e-14);
      }
      EXPECT_NEAR(source_energy->at(rowx, 0),
                  0.5 * 0.1 * 1.2 *
                      (std::pow(vel_parent->at(rowx, 0), 2) +
                       std::pow(vel_parent->at(rowx, 1), 2) +
                       std::pow(vel_parent->at(rowx, 2), 2) -
                       std::pow(vel_child->at(rowx, 0), 2) -
                       std::pow(vel_child->at(rowx, 1), 2) -
                       std::pow(vel_child->at(rowx, 2), 2)),
                  1e-14);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(SurfaceKernels, CartesianBasisReflectionData) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group<3>(N_total);
  particle_group->add_particle_dat(
      BoundaryInteractionSpecification::intersection_normal, 3);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_data = FixedRateData(1.0);

  std::array<REAL, 3> coords{2.0, M_PI / 4, 3 * M_PI / 4};
  auto coord_data = FixedArrayData<3>(coords);

  auto cartesian_reflection = CartesianBasisReflectionData();

  auto pipeline = PipelineData(coord_data, cartesian_reflection);

  auto properties_map = PropertiesMap();
  properties_map[VANTAGE::Reactions::default_properties.source_momentum] =
      "ION_SOURCE_MOMENTUM";
  properties_map[VANTAGE::Reactions::default_properties.source_energy] =
      "ION_SOURCE_ENERGY";
  auto projectile_species = Species("ION", 1.2, 0.0, 0);
  auto test_kernels =
      LinearScatteringKernels<3>(projectile_species, properties_map.get_map());

  auto data_calculator = DataCalculator<decltype(pipeline)>(pipeline);
  auto test_reaction =
      LinearReactionBase<1, FixedRateData, decltype(test_kernels),
                         decltype(data_calculator)>(
          particle_group->sycl_target, 0, std::array<int, 1>{0}, test_data,
          test_kernels, data_calculator);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto vel = std::array<REAL, 3>{1.0, 0.0, 1.0};
  auto norm = std::array<REAL, 3>{-1.0, 0.0, 0.0};
  particle_loop(
      "set__data_spherical_reflection", particle_sub_group,
      [=](auto normal, auto velocity) {
        for (auto i = 0; i < 3; i++) {

          normal.at(i) = norm[i];
          velocity.at(i) = vel[i];
        }
      },
      Access::write(BoundaryInteractionSpecification::intersection_normal),
      Access::write(Sym<REAL>("VELOCITY")))
      ->execute();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  std::array<REAL, 3> expected_vel{-coords[2], coords[1],
                                   coords[0]}; // see basis test
  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto weight = descendant_particles->get_cell(Sym<REAL>("WEIGHT"), i);
    auto vel_parent = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    auto vel_child = descendant_particles->get_cell(Sym<REAL>("VELOCITY"), i);

    auto source_momentum =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_MOMENTUM"), i);

    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_ENERGY"), i);
    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.1);

      for (int dim = 0; dim < 3; dim++) {
        EXPECT_NEAR(vel_child->at(rowx, dim), expected_vel[dim], 1e-14);
        EXPECT_NEAR(source_momentum->at(rowx, dim),
                    0.1 * 1.2 *
                        (vel_parent->at(rowx, dim) - vel_child->at(rowx, dim)),
                    1e-14);
      }
      EXPECT_NEAR(source_energy->at(rowx, 0),
                  0.5 * 0.1 * 1.2 *
                      (std::pow(vel_parent->at(rowx, 0), 2) +
                       std::pow(vel_parent->at(rowx, 1), 2) +
                       std::pow(vel_parent->at(rowx, 2), 2) -
                       std::pow(vel_child->at(rowx, 0), 2) -
                       std::pow(vel_child->at(rowx, 1), 2) -
                       std::pow(vel_child->at(rowx, 2), 2)),
                  1e-14);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
