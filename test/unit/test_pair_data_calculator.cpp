#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

template <size_t ndim = 2>
inline auto create_test_particle_groups_pairs(int N_total)
    -> std::tuple<ParticleGroupSharedPtr, ParticleGroupSharedPtr> {

  auto dims = std::vector<int>(ndim, 2);

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;

  const int pre_subdivision_cells =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  const int global_cell_count =
      pre_subdivision_cells * std::pow(std::pow(2, subdivision_order), ndim);
  const int npart_per_cell =
      std::round((double)N_total / (double)global_cell_count);

  auto mesh =
      std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                                       subdivision_order, stencil_width);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("POSITION"), ndim, true),
      ParticleProp(Sym<REAL>("VELOCITY"), ndim),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("REACTIONS_PANIC_FLAG"), 1),
      ParticleProp(Sym<INT>("REACTIONS_GROUPING_INDEX"), 1),
      ParticleProp(Sym<INT>("REACTIONS_LINEAR_INDEX"), 1),
      ParticleProp(Sym<INT>("PARTICLE_REACTED_FLAG"), 1),
      ParticleProp(Sym<INT>("ID"), 1),
      ParticleProp(Sym<REAL>("TOT_REACTION_RATE"), 1),
      ParticleProp(Sym<REAL>("WEIGHT"), 1),
      ParticleProp(Sym<INT>("INTERNAL_STATE"), 1),
      ParticleProp(Sym<REAL>("ELECTRON_TEMPERATURE"), 1),
      ParticleProp(Sym<REAL>("ELECTRON_DENSITY"), 1),
      ParticleProp(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), 1),
      ParticleProp(Sym<REAL>("ELECTRON_SOURCE_MOMENTUM"), ndim),
      ParticleProp(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), 1),
      ParticleProp(Sym<REAL>("ION_SOURCE_DENSITY"), 1),
      ParticleProp(Sym<REAL>("ION_SOURCE_MOMENTUM"), ndim),
      ParticleProp(Sym<REAL>("ION_SOURCE_ENERGY"), 1),
      ParticleProp(Sym<REAL>("ION2_SOURCE_DENSITY"), 1),
      ParticleProp(Sym<REAL>("ION2_SOURCE_MOMENTUM"), ndim),
      ParticleProp(Sym<REAL>("ION2_SOURCE_ENERGY"), 1),
      ParticleProp(Sym<REAL>("FLUID_DENSITY"), 1),
      ParticleProp(Sym<REAL>("FLUID_FLOW_SPEED"), ndim),
      ParticleProp(Sym<REAL>("FLUID_TEMPERATURE"), 1)};
  auto particle_group_a =
      std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  auto particle_group_b =
      std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

  const int cell_count = domain->mesh->get_cell_count();
  const int N = npart_per_cell * cell_count;

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells,
                                 rng_pos);

  auto velocities =
      NESO::Particles::normal_distribution(N, ndim, 0.0, 0.5, rng_vel);
  // std::uniform_int_distribution<int> uniform_dist(
  //     0, size - 1);
  ParticleSet initial_distribution(N, particle_group_a->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("POSITION")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("VELOCITY")][px][dimx] =
          velocities.at(dimx).at(px);
      initial_distribution[Sym<REAL>("ELECTRON_SOURCE_MOMENTUM")][px][dimx] =
          0.0;
      initial_distribution[Sym<REAL>("FLUID_FLOW_SPEED")][px][dimx] =
          1.0 + 2.0 * dimx;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    initial_distribution[Sym<REAL>("TOT_REACTION_RATE")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("WEIGHT")][px][0] = 1.0;
    initial_distribution[Sym<INT>("INTERNAL_STATE")][px][0] = 0;
    initial_distribution[Sym<REAL>("ELECTRON_TEMPERATURE")][px][0] = 2.0;
    initial_distribution[Sym<REAL>("ELECTRON_DENSITY")][px][0] = 3.0e18;
    initial_distribution[Sym<REAL>("ELECTRON_SOURCE_ENERGY")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("ELECTRON_SOURCE_DENSITY")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("FLUID_DENSITY")][px][0] = 3.0e18;
    initial_distribution[Sym<REAL>("FLUID_TEMPERATURE")][px][0] = 2.0;
  }
  particle_group_a->add_particles_local(initial_distribution);

  for (int px = 0; px < N; px++) {
    initial_distribution[Sym<INT>("INTERNAL_STATE")][px][0] = 1;
  }

  particle_group_b->add_particles_local(initial_distribution);
  auto pbc_a = std::make_shared<CartesianPeriodic>(
      sycl_target, mesh, particle_group_a->position_dat);
  auto ccb_a = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, particle_group_a->position_dat,
      particle_group_a->cell_id_dat);

  auto pbc_b = std::make_shared<CartesianPeriodic>(
      sycl_target, mesh, particle_group_b->position_dat);
  auto ccb_b = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, particle_group_b->position_dat,
      particle_group_a->cell_id_dat);

  pbc_a->execute();
  particle_group_a->hybrid_move();
  ccb_a->execute();
  particle_group_a->cell_move();

  pbc_b->execute();
  particle_group_b->hybrid_move();
  ccb_b->execute();
  particle_group_b->cell_move();

  MPI_Barrier(sycl_target->comm_pair.comm_parent);

  return std::tuple(particle_group_a, particle_group_b);
}

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
