#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_common.hpp"
#include "../include/test_reaction_controller_functors.hpp"
#include <memory>
#include <utility>

using namespace VANTAGE::Reactions;

TEST(ReactionController, surface_mode_test) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto accessor = NP::Access::read(NP::Sym<NP::REAL>("WEIGHT"));

  auto test_removal_wrapper = std::make_shared<TransformationWrapper>(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_direct_marking_strategy("small", SmallWeightMarker{}, accessor)},
      make_lambda_transformation_strategy("remove", RemoveSubgroupTransform{}));

  auto reaction_controller = ReactionController(
      std::vector<std::shared_ptr<TransformationWrapper>>{test_removal_wrapper},
      std::vector<std::shared_ptr<TransformationWrapper>>{});

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
  reaction_controller.apply(particle_group, 1.0, ControllerMode::surface_mode);
  reaction_controller.apply_parent_transforms(particle_group);

  int cell_count = particle_group->domain->mesh->get_cell_count();
  for (int i = 0; i < cell_count; i++) {

    auto weight = particle_group->get_cell(NP::Sym<NP::REAL>("WEIGHT"), i);
    auto id = particle_group->get_cell(NP::Sym<NP::INT>("INTERNAL_STATE"), i);

    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      switch (id->at(rowx, 0)) {
      case 0:

        EXPECT_TRUE(false); // any remaining particles must have nonzero id
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

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
