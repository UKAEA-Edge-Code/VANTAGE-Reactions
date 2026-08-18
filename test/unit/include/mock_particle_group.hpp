#ifndef REACTIONS_MOCK_PARTICLE_GROUP_H
#define REACTIONS_MOCK_PARTICLE_GROUP_H
#include "test_common.hpp"
#include "test_extern_templates.hpp"

using namespace VANTAGE::Reactions;

inline auto mesh_sycl_target_only() {
  auto dims = std::vector<int>(2, 2);

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;

  auto mesh = std::make_shared<NP::CartesianHMesh>(
      MPI_COMM_WORLD, 2, dims, cell_extent, subdivision_order, stencil_width);

  auto sycl_target =
      std::make_shared<NP::SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  return std::tuple(mesh, sycl_target);
}

template <size_t ndim = 2>
inline auto create_test_particle_group(int N_total)
    -> std::shared_ptr<NP::ParticleGroup> {

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

  auto mesh = std::make_shared<NP::CartesianHMesh>(
      MPI_COMM_WORLD, ndim, dims, cell_extent, subdivision_order,
      stencil_width);

  auto sycl_target =
      std::make_shared<NP::SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<NP::Domain>(mesh, cart_local_mapper);

  NP::ParticleSpec particle_spec{
      NP::ParticleProp(NP::Sym<REAL>("POSITION"), ndim, true),
      NP::ParticleProp(NP::Sym<REAL>("VELOCITY"), ndim),
      NP::ParticleProp(NP::Sym<INT>("CELL_ID"), 1, true),
      NP::ParticleProp(NP::Sym<INT>("REACTIONS_PANIC_FLAG"), 1),
      NP::ParticleProp(NP::Sym<INT>("REACTIONS_GROUPING_INDEX"), 1),
      NP::ParticleProp(NP::Sym<INT>("REACTIONS_LINEAR_INDEX"), 1),
      NP::ParticleProp(NP::Sym<INT>("PARTICLE_REACTED_FLAG"), 1),
      NP::ParticleProp(NP::Sym<INT>("ID"), 1),
      NP::ParticleProp(NP::Sym<REAL>("TOT_REACTION_RATE"), 1),
      NP::ParticleProp(NP::Sym<REAL>("WEIGHT"), 1),
      NP::ParticleProp(NP::Sym<INT>("INTERNAL_STATE"), 1),
      NP::ParticleProp(NP::Sym<REAL>("ELECTRON_TEMPERATURE"), 1),
      NP::ParticleProp(NP::Sym<REAL>("ELECTRON_DENSITY"), 1),
      NP::ParticleProp(NP::Sym<REAL>("ELECTRON_SOURCE_ENERGY"), 1),
      NP::ParticleProp(NP::Sym<REAL>("ELECTRON_SOURCE_MOMENTUM"), ndim),
      NP::ParticleProp(NP::Sym<REAL>("ELECTRON_SOURCE_DENSITY"), 1),
      NP::ParticleProp(NP::Sym<REAL>("ION_SOURCE_DENSITY"), 1),
      NP::ParticleProp(NP::Sym<REAL>("ION_SOURCE_MOMENTUM"), ndim),
      NP::ParticleProp(NP::Sym<REAL>("ION_SOURCE_ENERGY"), 1),
      NP::ParticleProp(NP::Sym<REAL>("ION2_SOURCE_DENSITY"), 1),
      NP::ParticleProp(NP::Sym<REAL>("ION2_SOURCE_MOMENTUM"), ndim),
      NP::ParticleProp(NP::Sym<REAL>("ION2_SOURCE_ENERGY"), 1),
      NP::ParticleProp(NP::Sym<REAL>("FLUID_DENSITY"), 1),
      NP::ParticleProp(NP::Sym<REAL>("FLUID_FLOW_SPEED"), ndim),
      NP::ParticleProp(NP::Sym<REAL>("FLUID_TEMPERATURE"), 1)};
  auto particle_group =
      std::make_shared<NP::ParticleGroup>(domain, particle_spec, sycl_target);

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

  auto velocities = NP::normal_distribution(N, ndim, 0.0, 0.5, rng_vel);
  // std::uniform_int_distribution<int> uniform_dist(
  //     0, size - 1);
  NP::ParticleSet initial_distribution(N, particle_group->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[NP::Sym<REAL>("POSITION")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[NP::Sym<REAL>("VELOCITY")][px][dimx] =
          velocities.at(dimx).at(px);
      initial_distribution[NP::Sym<REAL>("ELECTRON_SOURCE_MOMENTUM")][px]
                          [dimx] = 0.0;
      initial_distribution[NP::Sym<REAL>("FLUID_FLOW_SPEED")][px][dimx] =
          1.0 + 2.0 * dimx;
    }
    initial_distribution[NP::Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[NP::Sym<INT>("ID")][px][0] = px;
    initial_distribution[NP::Sym<REAL>("TOT_REACTION_RATE")][px][0] = 0.0;
    initial_distribution[NP::Sym<REAL>("WEIGHT")][px][0] = 1.0;
    initial_distribution[NP::Sym<INT>("INTERNAL_STATE")][px][0] = 0;
    initial_distribution[NP::Sym<REAL>("ELECTRON_TEMPERATURE")][px][0] = 2.0;
    initial_distribution[NP::Sym<REAL>("ELECTRON_DENSITY")][px][0] = 3.0e18;
    initial_distribution[NP::Sym<REAL>("ELECTRON_SOURCE_ENERGY")][px][0] = 0.0;
    initial_distribution[NP::Sym<REAL>("ELECTRON_SOURCE_DENSITY")][px][0] = 0.0;
    initial_distribution[NP::Sym<REAL>("FLUID_DENSITY")][px][0] = 3.0e18;
    initial_distribution[NP::Sym<REAL>("FLUID_TEMPERATURE")][px][0] = 2.0;
  }
  particle_group->add_particles_local(initial_distribution);

  auto pbc = std::make_shared<NP::CartesianPeriodic>(
      sycl_target, mesh, particle_group->position_dat);
  auto ccb = std::make_shared<NP::CartesianCellBin>(
      sycl_target, mesh, particle_group->position_dat,
      particle_group->cell_id_dat);

  pbc->execute();
  particle_group->hybrid_move();
  ccb->execute();
  particle_group->cell_move();

  MPI_Barrier(sycl_target->comm_pair.comm_parent);

  return particle_group;
}
#endif
