#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ArrayTransformData, unary_lambda_full_array) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  auto unary_lambda = [=](const std::array<REAL, 2> &a) {
    return std::array<REAL, 2>{a[0] * a[0], a[1] * a[1]};
  };

  auto lambda_wrapper =
      utils::LambdaWrapper<decltype(unary_lambda), 2>(unary_lambda);
  auto unary_transform_data =
      uatData<2, decltype(lambda_wrapper)>(lambda_wrapper);

  auto pipeline = PipelineData(position_data, unary_transform_data);
  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(pipeline)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(pipeline));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto source_density =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), i);
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0),
                       position->at(rowx, 0) * position->at(rowx, 0));
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 1) * position->at(rowx, 1));
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
TEST(ArrayTransformData, unary_lambda_elementwise) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  REAL capturable = 2.0;
  auto unary_lambda = [=](const REAL &a) { return capturable * a; };

  auto lambda_wrapper = utils::LambdaWrapper(unary_lambda);
  auto unary_transform_data =
      uetData<2, decltype(lambda_wrapper)>(lambda_wrapper);

  auto pipeline = PipelineData(position_data, unary_transform_data);
  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(pipeline)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(pipeline));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto source_density =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), i);
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), 2 * position->at(rowx, 0));
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0), 2 * position->at(rowx, 1));
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
