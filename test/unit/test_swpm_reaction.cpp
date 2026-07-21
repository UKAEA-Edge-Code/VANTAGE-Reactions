#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(SWPMReactions, simple_hs_scattering) {
  const int N_total = 800;

  auto [A, B] = create_test_particle_groups_pairs(N_total);

  particle_loop(
      "set_vel_A", A,
      [=](auto vel) {
        vel[0] = 2;
        vel[1] = 0;
      },
      Access::write(Sym<REAL>("VELOCITY")))
      ->execute();

  particle_loop(
      "set_vel_B", B,
      [=](auto vel) {
        vel[0] = 4;
        vel[1] = 0;
      },
      Access::write(Sym<REAL>("VELOCITY")))
      ->execute();

  int cell_count = A->domain->mesh->get_cell_count();

  auto cellwise_pair_list =
      std::make_shared<CellwisePairListSimple>(A->sycl_target, cell_count);

  std::vector<int> c;
  std::vector<int> i;
  std::vector<int> j;

  int npart_cell = std::round(A->get_npart_local() /
                              (double)A->domain->mesh->get_cell_count());
  c.reserve(cell_count * npart_cell);
  i.reserve(cell_count * npart_cell);
  j.reserve(cell_count * npart_cell);

  std::mt19937 rng(9124234 + A->sycl_target->comm_pair.rank_parent);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    std::vector<int> pairs(npart_cell);
    std::iota(pairs.begin(), pairs.end(), 0);
    std::shuffle(pairs.begin(), pairs.end(), rng);
    for (int px = 0; px < npart_cell; px++) {
      c.push_back(cellx);
      i.push_back(pairs.at(px));
      j.push_back(pairs.at(px));
    }
  }

  cellwise_pair_list->push_back(c, i, j);

  auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      A, B, cellwise_pair_list);

  auto cs_data = CSPairData<2>(ConstantRateCrossSection(0.1));

  auto rng_lambda = [&]() -> REAL { return 0.6; };

  auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 2);

  // Mocking the species to test the correct reduced mass effects
  auto species_1 = Species("ION", 1.2, 0.0, 0);
  auto species_2 = Species("ION2", 2.0, 0.0, 1);
  auto hs_scattering_data =
      HSScatteringData<2>(species_1, species_2, rng_kernel);
  auto pair_data_calculator = PairDataCalculator(hs_scattering_data);

  auto scattering_kernels = PairScatteringKernels<2>();

  auto swpm_reaction =
      SWPMReaction<2, decltype(cs_data), decltype(scattering_kernels),
                   decltype(pair_data_calculator)>(
          A->sycl_target, std::array<int, 2>{0, 0}, std::array<int, 2>{1, 2},
          cs_data, scattering_kernels, pair_data_calculator);

  //  for (int i = 0; i < cell_count; i++) {
  //    swpm_reaction.calculate_rates(pair_list, i, i + 1);
  //  }
  swpm_reaction.calculate_rates(pair_list, 0, cell_count);
  particle_loop(
      "copy_weight_change_test", A,
      [=](auto tot_rate, auto weight_change) {
        weight_change[0] = tot_rate[0];
      },
      Access::read(Sym<REAL>("TOT_REACTION_RATE")),
      Access::write(Sym<REAL>("WEIGHT_CHANGE")))
      ->execute();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      A->domain, A->get_particle_spec(), A->sycl_target);

  //  for (int i = 0; i < cell_count; i++) {
  //    swpm_reaction.apply(pair_list, i, i + 1, 1.0, descendant_particles);
  //  }
  swpm_reaction.apply(pair_list, 0, cell_count, 1.0, descendant_particles);

  // Expected velocities:
  // Centre-of-mass velocity: (1.2*2+2*4)/3.2 = 3.25
  // Relative velocity: 2
  // Random components: sample 0.6 in both directions, normalized to
  // 0.6/*sqrt(2*0.6^2) = a
  // Final velocities: [2.75 + 2*a, 2*a], and with -a for particle B

  REAL vel_com = 3.25;
  REAL expected_vel_random = 2 * 0.6 / std::sqrt(2 * 0.6 * 0.6);
  for (int i = 0; i < cell_count; i++) {

    auto weight = descendant_particles->get_cell(Sym<REAL>("WEIGHT"), i);
    auto weight_parent_a = A->get_cell(Sym<REAL>("WEIGHT"), i);
    auto weight_parent_b = B->get_cell(Sym<REAL>("WEIGHT"), i);
    auto velocity = descendant_particles->get_cell(Sym<REAL>("VELOCITY"), i);
    auto species_id =
        descendant_particles->get_cell(Sym<INT>("INTERNAL_STATE"), i);

    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.1);
      if (species_id->at(rowx, 0) == 1) {
        EXPECT_DOUBLE_EQ(velocity->at(rowx, 0), vel_com + expected_vel_random);
        EXPECT_DOUBLE_EQ(velocity->at(rowx, 1), expected_vel_random);
      } else {
        EXPECT_DOUBLE_EQ(velocity->at(rowx, 0), vel_com - expected_vel_random);
        EXPECT_DOUBLE_EQ(velocity->at(rowx, 1), -expected_vel_random);
      }
    }
    for (int rowx = 0; rowx < weight_parent_a->nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight_parent_a->at(rowx, 0), 0.9);
      EXPECT_DOUBLE_EQ(weight_parent_b->at(rowx, 0), 0.9);
    }
  }
  A->sycl_target->free();
  A->domain->mesh->free();
}
