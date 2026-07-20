#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(PairDataCalculator, cs_reaction_data_single_simple) {
  // This is a hard-coded number of particles so we get a round number for pair
  // construction (800/16 = 40 particles per cell, which is 20 pairs)
  const int N_total = 800;

  auto A = create_test_particle_group(N_total);

  int cell_count = A->domain->mesh->get_cell_count();

  auto cellwise_pair_listA =
      std::make_shared<CellwisePairListSimple>(A->sycl_target, cell_count);

  std::vector<int> c;
  std::vector<int> i;
  std::vector<int> j;

  int npart_cell = std::round(A->get_npart_local() /
                              (double)A->domain->mesh->get_cell_count());

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

  auto cs_data = CSPairData<2>(ConstantRateCrossSection(2.5));
  auto pair_data_calculator = PairDataCalculator(cs_data);

  auto nd_arr = std::make_shared<NDLocalArray<REAL, 2>>(
      A->sycl_target, cell_count * npart_cell / 2, 1);

  nd_arr->fill(0);

  auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      A, A, cellwise_pair_listA);

  pair_data_calculator.fill_buffer(nd_arr, pair_list, 0, cell_count);

  auto nd_arr_host = nd_arr->get();

  for (int i = 0; i < nd_arr_host.size(); i++) {
    EXPECT_DOUBLE_EQ(nd_arr_host.at(i), 2.5);
  }

  A->sycl_target->free();
  A->domain->mesh->free();
}

TEST(PairDataCalculator, cs_reaction_data_simple_and_constant) {
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

  auto cs_data = CSPairData<2>(ConstantRateCrossSection(2.5));
  auto cs_data_constant =
      CSPairData<2, ConstantCrossSection>(ConstantCrossSection(2.5, 1.0));
  auto pair_data_calculator = PairDataCalculator(cs_data, cs_data_constant);

  auto nd_arr = std::make_shared<NDLocalArray<REAL, 2>>(
      A->sycl_target, cell_count * npart_cell, 2);

  nd_arr->fill(0);

  auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      A, B, cellwise_pair_list);

  pair_data_calculator.fill_buffer(nd_arr, pair_list, 0, cell_count);

  auto nd_arr_host = nd_arr->get();

  for (int i = 0; i < nd_arr_host.size() - 1; i += 2) {
    EXPECT_DOUBLE_EQ(nd_arr_host.at(i), 2.5);
    EXPECT_DOUBLE_EQ(nd_arr_host.at(i + 1), 5.0);
  }

  A->sycl_target->free();
  A->domain->mesh->free();
}

TEST(PairDataCalculator,
     cs_reaction_data_simple_and_constant_concatenator_data) {
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

  auto cs_data = CSPairData<2>(ConstantRateCrossSection(2.5));
  auto cs_data_constant =
      CSPairData<2, ConstantCrossSection>(ConstantCrossSection(2.5, 1.0));
  auto pair_data_calculator =
      PairDataCalculator(ConcatenatorData(cs_data, cs_data_constant));

  auto nd_arr = std::make_shared<NDLocalArray<REAL, 2>>(
      A->sycl_target, cell_count * npart_cell, 2);

  nd_arr->fill(0);

  auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      A, B, cellwise_pair_list);

  pair_data_calculator.fill_buffer(nd_arr, pair_list, 0, cell_count);

  auto nd_arr_host = nd_arr->get();

  for (int i = 0; i < nd_arr_host.size() - 1; i += 2) {
    EXPECT_DOUBLE_EQ(nd_arr_host.at(i), 2.5);
    EXPECT_DOUBLE_EQ(nd_arr_host.at(i + 1), 5.0);
  }

  A->sycl_target->free();
  A->domain->mesh->free();
}

TEST(PairDataCalculator, cs_reaction_data_single_simple_pipeline_scale_by) {
  // This is a hard-coded number of particles so we get a round number for pair
  // construction (800/16 = 40 particles per cell, which is 20 pairs)
  const int N_total = 800;

  auto A = create_test_particle_group(N_total);

  int cell_count = A->domain->mesh->get_cell_count();

  auto cellwise_pair_listA =
      std::make_shared<CellwisePairListSimple>(A->sycl_target, cell_count);

  std::vector<int> c;
  std::vector<int> i;
  std::vector<int> j;

  int npart_cell = std::round(A->get_npart_local() /
                              (double)A->domain->mesh->get_cell_count());

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

  auto cs_data = CSPairData<2>(ConstantRateCrossSection(2.5));
  auto pair_data_calculator = PairDataCalculator(
      pipe(cs_data, scale_by<1, PairReactionDataArgumentPack>(2.0)));

  auto nd_arr = std::make_shared<NDLocalArray<REAL, 2>>(
      A->sycl_target, cell_count * npart_cell / 2, 1);

  nd_arr->fill(0);

  auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      A, A, cellwise_pair_listA);

  pair_data_calculator.fill_buffer(nd_arr, pair_list, 0, cell_count);

  auto nd_arr_host = nd_arr->get();

  for (int i = 0; i < nd_arr_host.size(); i++) {
    EXPECT_DOUBLE_EQ(nd_arr_host.at(i), 5.0);
  }

  A->sycl_target->free();
  A->domain->mesh->free();
}

TEST(PairDataCalculator, cs_reaction_data_simple_and_constant_binary_add) {
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

  auto cs_data = CSPairData<2>(ConstantRateCrossSection(2.5));
  auto cs_data_constant =
      CSPairData<2, ConstantCrossSection>(ConstantCrossSection(2.5, 1.0));
  auto pair_data_calculator = PairDataCalculator(cs_data + cs_data_constant);

  auto nd_arr = std::make_shared<NDLocalArray<REAL, 2>>(
      A->sycl_target, cell_count * npart_cell, 1);

  nd_arr->fill(0);

  auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      A, B, cellwise_pair_list);

  pair_data_calculator.fill_buffer(nd_arr, pair_list, 0, cell_count);

  auto nd_arr_host = nd_arr->get();

  for (int i = 0; i < nd_arr_host.size() - 1; i++) {
    EXPECT_DOUBLE_EQ(nd_arr_host.at(i), 7.5);
  }

  A->sycl_target->free();
  A->domain->mesh->free();
}
