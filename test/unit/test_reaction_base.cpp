#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(LinearReactionBase, calc_rate) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(
      particle_group, [=](auto ISTATE) { return (ISTATE[0] == 0); },
      Access::read(Sym<INT>("INTERNAL_STATE")));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 0;

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, test_rate, 0, std::array<int, 0>{});

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {

    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);

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
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_reaction = TestReactionVarRate(particle_group->sycl_target, 0);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    auto tot_reaction_rate =
        particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
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

TEST(LinearReactionBase, split_group_single_reaction) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto loop = particle_loop(
      "set_internal_state", particle_group,
      [=](auto internal_state) { internal_state[0] = 2; },
      Access::write(Sym<INT>("INTERNAL_STATE")));

  auto particle_group_2 = create_test_particle_group(N_total);
  auto loop2 = particle_loop(
      "set_internal_state2", particle_group_2,
      [=](auto internal_state) { internal_state[0] = 3; },
      Access::write(Sym<INT>("INTERNAL_STATE")));

  loop->execute();
  loop2->execute();

  particle_group->add_particles_local(particle_group_2);

  auto test_reaction1 =
      TestReaction<0>(particle_group->sycl_target, 1, 2, std::array<int, 0>{});

  auto test_reaction2 =
      TestReaction<1>(particle_group->sycl_target, 2, 3, std::array<int, 1>{4});

  std::vector<std::shared_ptr<AbstractReaction>> reactions = {
      std::make_shared<TestReaction<0>>(test_reaction1),
      std::make_shared<TestReaction<1>>(test_reaction2)};
  std::vector<std::shared_ptr<ParticleSubGroup>> subgroups;

  auto parent_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  int cell_count = particle_group->domain->mesh->get_cell_count();
  int num_reactions = static_cast<int>(reactions.size());
  for (int i = 0; i < cell_count; i++) {
    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      auto sub_group_selector =
          make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
              Sym<INT>("INTERNAL_STATE"), (reaction + 2));
      auto particle_sub_group = sub_group_selector->make_marker_subgroup(
          std::make_shared<ParticleSubGroup>(particle_group));
      subgroups.push_back(particle_sub_group);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->calculate_rates(subgroups[reaction], i, i + 1);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->apply(
          subgroups[reaction], i, i + 1, 0.1, descendant_particles);
    }

    auto internal_state =
        particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      if (internal_state->at(rowx, 0) == 2) {
        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.9);
      } else if (internal_state->at(rowx, 0) == 3) {
        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.8);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
  particle_group_2->sycl_target->free();
  particle_group_2->domain->mesh->free();
}

TEST(LinearReactionBase, single_group_multi_reaction) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);

  auto sub_group_selector =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("INTERNAL_STATE"), 0);

  auto particle_sub_group = sub_group_selector->make_marker_subgroup(
      std::make_shared<ParticleSubGroup>(particle_group));

  auto test_reaction1 =
      TestReaction<0>(particle_group->sycl_target, 1, 0, std::array<int, 0>{});

  auto test_reaction2 =
      TestReaction<0>(particle_group->sycl_target, 1, 0, std::array<int, 0>{});

  const INT num_products_per_parent = 1;

  auto test_reaction3 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, 2, 0, std::array<int, 1>{1});

  std::vector<std::shared_ptr<AbstractReaction>> reactions{};
  reactions.push_back(std::make_shared<TestReaction<0>>(test_reaction1));
  reactions.push_back(std::make_shared<TestReaction<0>>(test_reaction2));
  reactions.push_back(std::make_shared<TestReaction<1>>(test_reaction3));

  int cell_count = particle_group->domain->mesh->get_cell_count();
  int num_reactions = static_cast<int>(reactions.size());

  auto parent_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto particle_sub_group =
        std::make_shared<ParticleSubGroup>(particle_group);

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->calculate_rates(particle_sub_group, i, i + 1);
    }

    for (int reaction = 0; reaction < num_reactions; reaction++) {
      reactions[reaction]->apply(particle_sub_group, i, i + 1,
                                                   0.1, descendant_particles);
    }

    auto internal_state =
        particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      if (internal_state->at(rowx, 0) == 0) {
        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.6); //, 1e-12);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(LinearReactionBase, device_rate_buffer_reallocation) {
  auto particle_group = create_test_particle_group(1600);

  struct TestDeviceRateBufferReaction
      : public LinearReactionBase<0, FixedRateData, IoniseReactionKernels<2>,
                                  DataCalculator<FixedRateData>> {

    TestDeviceRateBufferReaction(ParticleGroupSharedPtr particle_group)
        : LinearReactionBase<0, FixedRateData, IoniseReactionKernels<2>,
                             DataCalculator<FixedRateData>>(
              particle_group->sycl_target, 0, std::array<int, 0>{},
              FixedRateData(1),
              IoniseReactionKernels<2>(Species("ION", 1.0, 1.0, 0),
                                       Species("ELECTRON"),
                                       Species("ELECTRON")),
              DataCalculator<FixedRateData>(FixedRateData(1))) {}

    const LocalArraySharedPtr<REAL> &get_device_rate_buffer_derived() {
      return this->get_device_rate_buffer();
    }
  };

  auto test_reaction = TestDeviceRateBufferReaction(particle_group);

  // Starting particle number in cell #0: 100
  test_reaction.blockwise_flush_buffer(
      std::make_shared<ParticleSubGroup>(particle_group), 0, 1);
  EXPECT_EQ(test_reaction.get_device_rate_buffer_derived()->size, 200);

  // Subtract 70 particles
  std::vector<INT> cells;
  std::vector<INT> layers;
  cells.reserve(70);
  layers.reserve(70);

  for (int i = 0; i < 70; i++) {
    cells.push_back(0);
    layers.push_back(i);
  }

  particle_group->remove_particles(cells.size(), cells, layers);

  // Check resize of device_rate_buffer to n_part_cell*2 = 60
  test_reaction.blockwise_flush_buffer(
      std::make_shared<ParticleSubGroup>(particle_group), 0, 1);
  EXPECT_EQ(test_reaction.get_device_rate_buffer_derived()->size, 60);

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(LinearReactionBase, data_calc_pre_req_ndim_mismatch) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  auto particle_group = create_test_particle_group(100);

  struct TestDataCalcNdimReaction
      : public LinearReactionBase<0, FixedRateData, IoniseReactionKernels<>> {
    TestDataCalcNdimReaction(ParticleGroupSharedPtr particle_group)
        : LinearReactionBase<0, FixedRateData, IoniseReactionKernels<>>(
              particle_group->sycl_target, 0, std::array<int, 0>{},
              FixedRateData(1),
              IoniseReactionKernels<>(Species("ION", 1.0, 1.0, 0),
                                      Species("ELECTRON"),
                                      Species("ELECTRON"))) {}
  };

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW((TestDataCalcNdimReaction(particle_group)), std::logic_error);
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
