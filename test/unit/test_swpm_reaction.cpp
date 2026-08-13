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

  auto cs_data =
      CSPairData<2, ConstantCrossSection>(ConstantCrossSection(0.05, 1.0));

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

  for (int i = 0; i < cell_count; i++) {
    swpm_reaction.calculate_rates(pair_list, i, i + 1);
  }
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

  for (int i = 0; i < cell_count; i++) {
    swpm_reaction.apply(pair_list, i, i + 1, 1.0, descendant_particles);
  }

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

TEST(SWPMSpecification, SWPMDSMCSpecification) {

  const int N_total = 800;

  auto A = create_test_particle_group(N_total);

  auto particle_subgroup = particle_sub_group(A);
  int cell_count = A->domain->mesh->get_cell_count();

  std::vector<int> subdivision_order(cell_count, 1);

  auto coll_cell_h = make_coll_cell_hierarchy<CartesianCollCellH>(
      A->sycl_target,
      std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
      subdivision_order);

  auto cc_manager = std::make_shared<CollisionCellManager>(
      A->sycl_target, coll_cell_h, std::vector<INT>{0});

  cc_manager->bin_particles(particle_subgroup);
  cc_manager->construct_cell_partition(particle_subgroup);
  auto rng_lambda = [&]() -> REAL { return 0.6; };

  auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 1);
  auto spec = SWPMDSMCSpecification(rng_kernel);

  auto result_buffer =
      cc_manager->get_empty_coll_cellwise_data<REAL>(A->sycl_target);
  auto timestep_buffer =
      cc_manager->get_empty_coll_cellwise_data<REAL>(A->sycl_target);
  auto sigma_v_bound =
      cc_manager->get_empty_coll_cellwise_data<REAL>(A->sycl_target);
  sigma_v_bound->fill(2.0);

  spec.calculate_exponential_parameter(particle_subgroup, cc_manager, 0, 0,
                                       sigma_v_bound, result_buffer,
                                       timestep_buffer);

  auto npart_cell = N_total / cell_count;
  REAL expected = 2 * npart_cell * (npart_cell - 1) * 2 * M_PI / 0.25;

  NDHostArraySharedPtr<REAL, 2> host_result;
  result_buffer->get(host_result);
  for (int i = 0; i < cell_count; i++) {
    EXPECT_DOUBLE_EQ(host_result->at(i, 0), expected);
  }

  timestep_buffer->get(host_result);
  for (int i = 0; i < cell_count; i++) {
    EXPECT_DOUBLE_EQ(host_result->at(i, 0), npart_cell / expected);
  }

  auto cellwise_pair_listA =
      std::make_shared<CellwisePairListSimple>(A->sycl_target, cell_count);

  std::vector<int> c;
  std::vector<int> i;
  std::vector<int> j;

  c.reserve(cell_count * npart_cell / 2);
  i.reserve(cell_count * npart_cell / 2);
  j.reserve(cell_count * npart_cell / 2);

  std::mt19937 rng(9124234 + A->sycl_target->comm_pair.rank_parent);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    npart_cell = A->get_npart_cell(cellx);
    std::vector<int> pairs(npart_cell);
    std::iota(pairs.begin(), pairs.end(), 0);
    std::shuffle(pairs.begin(), pairs.end(), rng);
    for (int px = 0; px < (npart_cell / 2); px++) {
      c.push_back(cellx);
      i.push_back(pairs.at(2 * px));
      j.push_back(pairs.at(2 * px + 1));
    }
  }

  cellwise_pair_listA->push_back(c, i, j);
  auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      A, A, cellwise_pair_listA);

  particle_pair_loop(
      "set_tot_reaction_rate", pair_list,
      [](auto a, auto b, auto id, auto id_b, auto w) {
        a[0] = 1.0 + id[0] % 2;
        b[0] = a[0];
        id_b[0] = id[0];
        w[0] = id[0] % 2 ? 0.5 : 1.0;
      },
      Access::A(Access::write(Sym<REAL>("TOT_REACTION_RATE"))),
      Access::B(Access::write(Sym<REAL>("TOT_REACTION_RATE"))),
      Access::A(Access::read(Sym<INT>("ID"))),
      Access::B(Access::write(Sym<INT>("ID"))),
      Access::A(Access::write(Sym<REAL>("WEIGHT"))))
      ->execute(0, cell_count);

  spec.calculate_exponential_parameter(particle_subgroup, cc_manager, 0, 0,
                                       sigma_v_bound, result_buffer,
                                       timestep_buffer);

  spec.calculate_weight_transfer(pair_list, 0, cell_count);

  for (int i = 0; i < cell_count; i++) {
    auto weight_change = A->get_cell(Sym<REAL>("WEIGHT_CHANGE"), i);
    auto tot_reaction_rate = A->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    auto id = A->get_cell(Sym<INT>("ID"), i);

    const int nrow = weight_change->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      if (id->at(rowx, 0) % 2) {
        EXPECT_DOUBLE_EQ(weight_change->at(rowx, 0), 0.5);
      } else {
        EXPECT_DOUBLE_EQ(weight_change->at(rowx, 0), 0.0);
      }
    }
  }
}
