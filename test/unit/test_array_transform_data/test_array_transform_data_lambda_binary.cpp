
#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <gtest/gtest.h>

using namespace VANTAGE::Reactions;

TEST(ArrayTransformData, binary_lambda_full_array) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(NP::Sym<NP::REAL>("POSITION"));

  auto binary_lambda = [](const std::array<NP::REAL, 2> &a,
                          const std::array<NP::REAL, 2> &b) {
    return std::array<NP::REAL, 2>{a[0] * b[1], b[1]};
  };

  auto lambda_wrapper =
      utils::LambdaWrapper<decltype(binary_lambda), 2>{binary_lambda};
  auto binary_transform_data =
      batData(lambda_wrapper, position_data, position_data);

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<NP::ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto position = particle_group->get_cell(NP::Sym<NP::REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto source_density = particle_group->get_cell(
        NP::Sym<NP::REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy = particle_group->get_cell(
        NP::Sym<NP::REAL>("ELECTRON_SOURCE_ENERGY"), i);
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0),
                       position->at(rowx, 0) * position->at(rowx, 1));
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0), position->at(rowx, 1));
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
TEST(ArrayTransformData, binary_lambda_elementwise) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(NP::Sym<NP::REAL>("POSITION"));

  auto binary_lambda = [](const NP::REAL &a, const NP::REAL &b) {
    return 2 * a + b;
  };

  auto lambda_wrapper =
      utils::LambdaWrapper<decltype(binary_lambda), 1>{binary_lambda};
  auto binary_transform_data =
      betData(lambda_wrapper, position_data, position_data);

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<NP::ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto position = particle_group->get_cell(NP::Sym<NP::REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto source_density = particle_group->get_cell(
        NP::Sym<NP::REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy = particle_group->get_cell(
        NP::Sym<NP::REAL>("ELECTRON_SOURCE_ENERGY"), i);
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), position->at(rowx, 0) * 3);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0), position->at(rowx, 1) * 3);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
