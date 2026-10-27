#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_common.hpp"
#include "../include/test_reaction_controller_functors.hpp"
#include <memory>
#include <utility>

using namespace VANTAGE::Reactions;

namespace {

void expect_parent_weights(NP::ParticleGroupSharedPtr particle_group,
                           REAL expected_even, REAL expected_odd) {
  INT parent_count = 0;
  const int cell_count = particle_group->domain->mesh->get_cell_count();
  for (int cell = 0; cell < cell_count; cell++) {
    auto states = particle_group->get_cell(NP::Sym<INT>("INTERNAL_STATE"), cell);
    auto ids = particle_group->get_cell(NP::Sym<INT>("ID"), cell);
    auto weights = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), cell);
    for (int row = 0; row < states->nrow; row++) {
      if (states->at(row, 0) == 0) {
        parent_count++;
        EXPECT_DOUBLE_EQ(weights->at(row, 0),
                         ids->at(row, 0) % 2 ? expected_odd : expected_even);
      }
    }
  }
  EXPECT_GT(parent_count, 0);
}

} // namespace

TEST(ReactionController, mode_switching) {
  for (const bool semi_dsmc_first : {false, true}) {
    SCOPED_TRACE(semi_dsmc_first);
    auto particle_group = create_test_particle_group(160);
    const auto parent_count = particle_group->get_npart_local();
    auto reaction_controller = ReactionController();
    reaction_controller.add_reaction(std::make_shared<TestReaction<1>>(
        particle_group->sycl_target, 1.0, 0, std::array<int, 1>{1}));
    reaction_controller.set_rng_kernel(
        NP::host_per_particle_block_rng<REAL>([]() -> REAL { return 0.9; }, 1));

    // Reject every semi-DSMC event. Initialize the flag so stale filtered
    // subgroups also deterministically reject particles in standard mode.
    particle_loop(
        "reset_reacted_flags", particle_group,
        [](auto reacted) { reacted[0] = 0; },
        NP::Access::write(NP::Sym<INT>("PARTICLE_REACTED_FLAG")))
        ->execute();

    const auto first_mode = semi_dsmc_first ? ControllerMode::semi_dsmc_mode
                                          : ControllerMode::standard_mode;
    const auto second_mode = semi_dsmc_first ? ControllerMode::standard_mode
                                           : ControllerMode::semi_dsmc_mode;
    reaction_controller.apply(particle_group, 0.25, first_mode);
    const REAL first_weight = semi_dsmc_first ? 1.0 : 0.75;
    expect_parent_weights(particle_group, first_weight, first_weight);
    EXPECT_EQ(particle_group->get_npart_local(),
              semi_dsmc_first ? parent_count : 2 * parent_count);

    reaction_controller.apply(particle_group, 0.25, second_mode);
    expect_parent_weights(particle_group, 0.75, 0.75);
    EXPECT_EQ(particle_group->get_npart_local(), 2 * parent_count);

    particle_group->sycl_target->free();
    particle_group->domain->mesh->free();
  }
}

TEST(ReactionController, target_switching) {
  auto particle_group = create_test_particle_group(160);
  const auto parent_count = particle_group->get_npart_local();
  auto reaction_controller = ReactionController();
  reaction_controller.add_reaction(std::make_shared<TestReaction<1>>(
      particle_group->sycl_target, 1.0, 0, std::array<int, 1>{1}));
  auto even_particles = NP::particle_sub_group(
      particle_group, [](auto id) { return id[0] % 2 == 0; },
      NP::Access::read(NP::Sym<INT>("ID")));
  auto odd_particles = NP::particle_sub_group(
      particle_group, [](auto id) { return id[0] % 2 != 0; },
      NP::Access::read(NP::Sym<INT>("ID")));

  // Exercise whole group -> subgroup -> different subgroup -> whole group.
  // All targets share the same parent group, as required by the controller.
  reaction_controller.apply(particle_group, 0.25);
  expect_parent_weights(particle_group, 0.75, 0.75);
  EXPECT_EQ(particle_group->get_npart_local(), 2 * parent_count);

  reaction_controller.apply(even_particles, 0.25);
  expect_parent_weights(particle_group, 0.5, 0.75);
  EXPECT_EQ(particle_group->get_npart_local(), 5 * parent_count / 2);

  reaction_controller.apply(odd_particles, 0.25);
  expect_parent_weights(particle_group, 0.5, 0.5);
  EXPECT_EQ(particle_group->get_npart_local(), 3 * parent_count);

  reaction_controller.apply(particle_group, 0.25);
  expect_parent_weights(particle_group, 0.25, 0.25);
  EXPECT_EQ(particle_group->get_npart_local(), 4 * parent_count);

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ReactionController, semi_dsmc_test) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto accessor = NP::Access::read(NP::Sym<REAL>("WEIGHT"));

  auto test_removal_wrapper = std::make_shared<TransformationWrapper>(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_direct_marking_strategy("small", SmallWeightMarker{}, accessor)},
      make_lambda_transformation_strategy("remove", RemoveSubgroupTransform{}));
  auto reaction_controller = ReactionController(
      std::vector<std::shared_ptr<TransformationWrapper>>{test_removal_wrapper},
      std::vector<std::shared_ptr<TransformationWrapper>>{});

  auto loop = particle_loop(
      "set_weights", particle_group,
      [=](auto id, auto weight) { weight[0] = id[0] % 2 ? 1.0 : 0.5; },
      NP::Access::read(NP::Sym<INT>("ID")),
      NP::Access::write(NP::Sym<REAL>("WEIGHT")));
  loop->execute();

  auto rng_lambda = [&]() -> REAL {
    return 0.90;
  }; // should only react particles with weight 1
  auto rng_kernel = NP::host_per_particle_block_rng<REAL>(rng_lambda, 1);
  reaction_controller.set_rng_kernel(rng_kernel);

  auto squared_reaction_data = FixedCoefficientData(1.0) * extract<1>("WEIGHT");

  auto test_reaction_1 = std::make_shared<
      LinearReactionBase<1, SquaredWeightData, TestReactionKernels<1>>>(
      particle_group->sycl_target, 0, std::array<int, 1>{1},
      squared_reaction_data, TestReactionKernels<1>());

  reaction_controller.add_reaction(test_reaction_1);

  auto squared_reaction_data_2 =
      FixedCoefficientData(3.0) * extract<1>("WEIGHT");
  auto test_reaction_2 = std::make_shared<
      LinearReactionBase<1, SquaredWeightData, TestReactionKernels<1>>>(
      particle_group->sycl_target, 0, std::array<int, 1>{2},
      squared_reaction_data_2, TestReactionKernels<1>());

  reaction_controller.add_reaction(test_reaction_2);

  auto start_npart = particle_group->get_npart_local();
  reaction_controller.apply(particle_group, 1.0,
                            ControllerMode::semi_dsmc_mode);

  int cell_count = particle_group->domain->mesh->get_cell_count();
  for (int i = 0; i < cell_count; i++) {

    auto weight = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), i);
    auto id = particle_group->get_cell(NP::Sym<INT>("INTERNAL_STATE"), i);

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

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
