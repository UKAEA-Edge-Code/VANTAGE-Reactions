#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_common.hpp"
#include "../include/test_reaction_controller_functors.hpp"
#include <memory>
#include <utility>

using namespace VANTAGE::Reactions;

TEST(ReactionController, multi_reaction_multi_apply) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto particle_group_2 = create_test_particle_group(N_total);
  auto loop2 = particle_loop(
      "set_internal_state2", particle_group_2,
      [=](auto internal_state) { internal_state[0] = 2; },
      NP::Access::write(NP::Sym<INT>("INTERNAL_STATE")));

  loop2->execute();

  particle_group->add_particles_local(particle_group_2);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto test_wrapper = std::make_shared<TransformationWrapper>(child_transform);
  auto reaction_controller = ReactionController(test_wrapper);

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 1;

  auto test_reaction1 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0,
      std::array<int, num_products_per_parent>{1});

  test_rate = 10.0; // example rate

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 2,
      std::array<int, num_products_per_parent>{3});

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction1));

  reaction_controller.add_reaction(
      std::make_shared<TestReaction<num_products_per_parent>>(test_reaction2));

  auto reduction = std::make_shared<NP::CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(particle_group, WeightReducer{},
                NP::Access::read(NP::Sym<REAL>("WEIGHT")),
                NP::Access::add(reduction))
      ->execute();

  reaction_controller.apply(particle_group, 0.1);
  auto reduction_after = std::make_shared<NP::CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(particle_group, WeightReducer{},
                NP::Access::read(NP::Sym<REAL>("WEIGHT")),
                NP::Access::add(reduction_after))
      ->execute();

  auto merged_group =
      particle_sub_group(particle_group, InternalStateEquals(1),
                         NP::Access::read(NP::Sym<INT>("INTERNAL_STATE")));

  auto merged_group2 =
      particle_sub_group(particle_group, InternalStateEquals(3),
                         NP::Access::read(NP::Sym<INT>("INTERNAL_STATE")));

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 2);
    EXPECT_EQ(merged_group2->get_npart_cell(icell), 2);

    EXPECT_DOUBLE_EQ(reduction_after->get_cell(icell)->at(0, 0),
                     reduction->get_cell(icell)->at(0, 0)); //, 1e-12);
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();

  particle_group_2->sycl_target->free();
  particle_group_2->domain->mesh->free();
}
