#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(LinearReactionBase, calc_rate) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_subgroup = particle_sub_group(
      particle_group, [=](auto ISTATE) { return (ISTATE[0] == 0); },
      NP::Access::read(NP::Sym<INT>("INTERNAL_STATE")));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 0;

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0, std::array<int, 0>{});

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_subgroup, i, i + 1);
    test_reaction.calculate_rates(particle_subgroup, i, i + 1);

    auto position = particle_group->get_cell(NP::Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(NP::Sym<REAL>("TOT_REACTION_RATE"), i);

    const int nrow = position->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(tot_reaction_rate->at(rowx, 0), 2 * test_rate)
          << "calc_rate did not set TOT_REACTION_RATE correctly...";
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(LinearReactionBase, calc_var_rate) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_subgroup = particle_sub_group(particle_group);

  auto test_reaction = TestReactionVarRate(particle_group->sycl_target, 0);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_subgroup, i, i + 1);
    test_reaction.calculate_rates(particle_subgroup, i, i + 1);

    auto position = particle_group->get_cell(NP::Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(NP::Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = position->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(tot_reaction_rate->at(rowx, 0),
                       2 * position->at(rowx, 0))
          << "calc_rate dP not set TOT_REACTION_RATE correctly...";
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(LinearReactionBase, zero_denominator_guards) {
  const int N_total = 32;

  // 1. Zero individual rate with a positive competing rate
  {
    auto particle_group = create_test_particle_group(N_total);
    auto subgroup = particle_sub_group(particle_group);
    auto descendants = std::make_shared<NP::ParticleGroup>(
        particle_group->domain, particle_group->get_particle_spec(),
        particle_group->sycl_target);
    const int cell_count = particle_group->domain->mesh->get_cell_count();

    auto zero_reaction = TestReaction<1>(particle_group->sycl_target, 0.0, 0,
                                         std::array<int, 1>{1});
    auto pos_reaction = TestReaction<1>(particle_group->sycl_target, 2.0, 0,
                                        std::array<int, 1>{2});

    zero_reaction.calculate_rates(subgroup, 0, cell_count);
    pos_reaction.calculate_rates(subgroup, 0, cell_count);
    zero_reaction.apply(subgroup, 0, cell_count, 0.25, descendants);
    pos_reaction.apply(subgroup, 0, cell_count, 0.25, descendants);

    EXPECT_EQ(descendants->get_npart_local(),
              particle_group->get_npart_local());
    for (int cx = 0; cx < cell_count; cx++) {
      auto parent_weights = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), cx);
      for (int row = 0; row < parent_weights->nrow; row++) {
        EXPECT_DOUBLE_EQ(parent_weights->at(row, 0), 0.5);
      }
      auto child_states =
          descendants->get_cell(NP::Sym<INT>("INTERNAL_STATE"), cx);
      auto child_weights = descendants->get_cell(NP::Sym<REAL>("WEIGHT"), cx);
      for (int row = 0; row < child_states->nrow; row++) {
        EXPECT_EQ(child_states->at(row, 0), 2);
        EXPECT_DOUBLE_EQ(child_weights->at(row, 0), 0.5);
      }
    }
    particle_group->sycl_target->free();
    particle_group->domain->mesh->free();
  }

  // 2. All-zero rates (both standard and full_weight)
  for (const bool full_weight : {false, true}) {
    auto particle_group = create_test_particle_group(N_total);
    auto subgroup = particle_sub_group(particle_group);
    auto descendants = std::make_shared<NP::ParticleGroup>(
        particle_group->domain, particle_group->get_particle_spec(),
        particle_group->sycl_target);
    const int cell_count = particle_group->domain->mesh->get_cell_count();

    auto zero_reaction = TestReaction<1>(particle_group->sycl_target, 0.0, 0,
                                         std::array<int, 1>{1});
    zero_reaction.calculate_rates(subgroup, 0, cell_count);
    zero_reaction.apply(subgroup, 0, cell_count, 0.25, descendants, full_weight);

    EXPECT_EQ(descendants->get_npart_local(), 0);
    for (int cx = 0; cx < cell_count; cx++) {
      auto parent_weights = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), cx);
      for (int row = 0; row < parent_weights->nrow; row++) {
        EXPECT_DOUBLE_EQ(parent_weights->at(row, 0), 1.0);
      }
    }
    particle_group->sycl_target->free();
    particle_group->domain->mesh->free();
  }

  // 3. Zero timestep (dt = 0.0)
  {
    auto particle_group = create_test_particle_group(N_total);
    auto subgroup = particle_sub_group(particle_group);
    auto descendants = std::make_shared<NP::ParticleGroup>(
        particle_group->domain, particle_group->get_particle_spec(),
        particle_group->sycl_target);
    const int cell_count = particle_group->domain->mesh->get_cell_count();

    auto pos_reaction = TestReaction<1>(particle_group->sycl_target, 2.0, 0,
                                        std::array<int, 1>{1});
    pos_reaction.calculate_rates(subgroup, 0, cell_count);
    pos_reaction.apply(subgroup, 0, cell_count, 0.0, descendants);

    EXPECT_EQ(descendants->get_npart_local(), 0);
    for (int cx = 0; cx < cell_count; cx++) {
      auto parent_weights = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), cx);
      for (int row = 0; row < parent_weights->nrow; row++) {
        EXPECT_DOUBLE_EQ(parent_weights->at(row, 0), 1.0);
      }
    }
    particle_group->sycl_target->free();
    particle_group->domain->mesh->free();
  }

  // 4. Zero particle weight
  for (const bool full_weight : {false, true}) {
    auto particle_group = create_test_particle_group(N_total);
    particle_loop(
        "zero_weights", particle_group, [](auto w) { w[0] = 0.0; },
        NP::Access::write(NP::Sym<REAL>("WEIGHT")))
        ->execute();
    auto subgroup = particle_sub_group(particle_group);
    auto descendants = std::make_shared<NP::ParticleGroup>(
        particle_group->domain, particle_group->get_particle_spec(),
        particle_group->sycl_target);
    const int cell_count = particle_group->domain->mesh->get_cell_count();

    auto pos_reaction = TestReaction<1>(particle_group->sycl_target, 2.0, 0,
                                        std::array<int, 1>{1});
    pos_reaction.calculate_rates(subgroup, 0, cell_count);
    pos_reaction.apply(subgroup, 0, cell_count, 0.25, descendants, full_weight);

    EXPECT_EQ(descendants->get_npart_local(), 0);
    for (int cx = 0; cx < cell_count; cx++) {
      auto parent_weights = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), cx);
      for (int row = 0; row < parent_weights->nrow; row++) {
        EXPECT_DOUBLE_EQ(parent_weights->at(row, 0), 0.0);
      }
    }
    particle_group->sycl_target->free();
    particle_group->domain->mesh->free();
  }
}
