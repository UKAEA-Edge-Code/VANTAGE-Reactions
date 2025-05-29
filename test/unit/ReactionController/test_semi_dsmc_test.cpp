#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ReactionController, semi_dsmc_test) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto test_removal_wrapper = std::make_shared<TransformationWrapper>(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
              Sym<REAL>("WEIGHT"), 1.0e-12)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());
  auto reaction_controller = ReactionController(
      std::vector<std::shared_ptr<TransformationWrapper>>{test_removal_wrapper},
      std::vector<std::shared_ptr<TransformationWrapper>>{});

  auto loop = particle_loop(
      "set_weights", particle_group,
      [=](auto id, auto weight) { weight[0] = id[0] % 2 ? 1.0 : 0.5; },
      Access::read(Sym<INT>("ID")), Access::write(Sym<REAL>("WEIGHT")));
  loop->execute();

  auto rng_lambda = [&]() -> REAL {
    return 0.90;
  }; // should only react particles with weight 1
  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, 1);
  reaction_controller.set_rng_kernel(rng_kernel);

  auto test_reaction_1 = std::make_shared<
      LinearReactionBase<1, FixedCoefficientData, TestReactionKernels<1>>>(
      particle_group->sycl_target, 0, std::array<int, 1>{1},
      FixedCoefficientData(1.0), TestReactionKernels<1>());

  reaction_controller.add_reaction(test_reaction_1);

  auto test_reaction_2 = std::make_shared<
      LinearReactionBase<1, FixedCoefficientData, TestReactionKernels<1>>>(
      particle_group->sycl_target, 0, std::array<int, 1>{2},
      FixedCoefficientData(3.0), TestReactionKernels<1>());

  reaction_controller.add_reaction(test_reaction_2);

  auto start_npart = particle_group->get_npart_local();
  reaction_controller.apply_reactions(particle_group, 1.0, ControllerMode::semi_dsmc_mode);

  int cell_count = particle_group->domain->mesh->get_cell_count();
  for (int i = 0; i < cell_count; i++) {

    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);
    auto id = particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);

    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      switch (id->at(rowx, 0)) {
      case 0:

        EXPECT_DOUBLE_EQ(
            weight->at(rowx, 0),
            0.5); // any remaining 0 ID particles must have 0.5 weight
        break;

      case 1:

        EXPECT_DOUBLE_EQ(weight->at(rowx, 0),
                         0.25); // 1:3 ratio of reactions 1 and 2
        break;

      case 2:

        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.75);
        break;

      default:
        EXPECT_TRUE(false);
      }
    }
  }
  particle_group->domain->mesh->free();
}