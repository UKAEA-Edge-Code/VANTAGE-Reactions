#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ArrayTransformData, polynomial_linear) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  auto linear_poly =
      PolynomialArrayTransform<2, 1>(std::array<REAL, 2>{1.0, 2.0});

  auto unary_transform_data = UnaryArrayTransformData(linear_poly);

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
                       position->at(rowx, 0) * 2 + 1.0);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 1) * 2 + 1.0);
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_add) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data + position_data;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), position->at(rowx, 0) * 2);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0), position->at(rowx, 1) * 2);
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_add_left_scalar) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto position_data_1 = ExtractorData<1>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data_1 + position_data;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), position->at(rowx, 0) * 2);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 1) + position->at(rowx, 0));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_add_right_scalar) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto position_data_1 = ExtractorData<1>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data + position_data_1;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), position->at(rowx, 0) * 2);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 1) + position->at(rowx, 0));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_sub) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto flow_speed = ExtractorData<2>(Sym<REAL>("FLUID_FLOW_SPEED"));

  auto binary_transform_data = position_data - flow_speed;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    auto speed = particle_group->get_cell(Sym<REAL>("FLUID_FLOW_SPEED"), i);
    const int nrow = position->nrow;

    auto source_density =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), i);
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0),
                       position->at(rowx, 0) - speed->at(rowx, 0));
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 1) - speed->at(rowx, 1));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_sub_left_scalar) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto position_data_1 = ExtractorData<1>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data_1 - position_data;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), 0);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 0) - position->at(rowx, 1));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_sub_right_scalar) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto position_data_1 = ExtractorData<1>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data - position_data_1;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), 0);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 1) - position->at(rowx, 0));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}
TEST(ArrayTransformData, binary_mult) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data * position_data;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}
TEST(ArrayTransformData, binary_mult_right_scalar) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto position_data_1 = ExtractorData<1>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data * position_data_1;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
                       position->at(rowx, 1) * position->at(rowx, 0));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_mult_left_scalar) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto position_data_1 = ExtractorData<1>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data_1 * position_data;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
                       position->at(rowx, 1) * position->at(rowx, 0));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_div) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto flow_speed = ExtractorData<2>(Sym<REAL>("FLUID_FLOW_SPEED"));

  auto binary_transform_data = position_data / flow_speed;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    auto speed = particle_group->get_cell(Sym<REAL>("FLUID_FLOW_SPEED"), i);
    const int nrow = position->nrow;

    auto source_density =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), i);
    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0),
                       position->at(rowx, 0) / speed->at(rowx, 0));
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 1) / speed->at(rowx, 1));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}
TEST(ArrayTransformData, binary_div_right_scalar) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto position_data_1 = ExtractorData<1>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data / position_data_1;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), 1.0);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 1) / position->at(rowx, 0));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_div_left_scalar) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));
  auto position_data_1 = ExtractorData<1>(Sym<REAL>("POSITION"));

  auto binary_transform_data = position_data_1 / position_data;

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), 1.0);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       position->at(rowx, 0) / position->at(rowx, 1));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}
TEST(ArrayTransformData, unary_project) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  auto unary_transform_data = UnaryArrayTransformData(
      UnaryProjectArrayTransform(std::array<REAL, 2>{1.0, 0.5}));

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
                       position->at(rowx, 0) + 0.5 * position->at(rowx, 1));
      EXPECT_DOUBLE_EQ(
          source_energy->at(rowx, 0),
          0.5 * (position->at(rowx, 0) + 0.5 * position->at(rowx, 1)));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_project) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  auto binary_transform_data = BinaryArrayTransformData(
      BinaryProjectArrayTransform<2>(), position_data, position_data);

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(binary_transform_data)>>(

          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(binary_transform_data));

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
      auto norm2 = position->at(rowx, 0) * position->at(rowx, 0) +
                   position->at(rowx, 1) * position->at(rowx, 1);
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0),
                       norm2 * position->at(rowx, 0));
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0),
                       norm2 * position->at(rowx, 1));
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}

TEST(ArrayTransformData, binary_dot) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto position_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  auto binary_transform_data = dot_product(position_data, position_data);

  auto data_calculator =
      DataCalculator(binary_transform_data, binary_transform_data);

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         decltype(data_calculator)>(
          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          data_calculator);

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
      auto norm2 = position->at(rowx, 0) * position->at(rowx, 0) +
                   position->at(rowx, 1) * position->at(rowx, 1);
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), norm2);
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0), norm2);
    }
  }

  particle_group->domain->mesh->free();
  descendant_particles->domain->mesh->free();
}
