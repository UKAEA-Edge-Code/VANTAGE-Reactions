#pragma once
#include <reactions.hpp>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

inline auto create_test_particle_group_marking(int N_total)
    -> std::shared_ptr<ParticleGroup> {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 2;
  dims[1] = 2;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;

  const int global_cell_count =
      dims[0] * dims[1] * std::pow(std::pow(2, subdivision_order), ndim);
  const int npart_per_cell =
      std::round((double)N_total / (double)global_cell_count);

  auto mesh =
      std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                                       subdivision_order, stencil_width);

  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("POSITION"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("WEIGHT"), 1),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<INT>("MOCK_INT"), 1),
                             ParticleProp(Sym<REAL>("V"), 2),
                             ParticleProp(Sym<REAL>("MOCK_SOURCE2D"), 2),
                             ParticleProp(Sym<REAL>("MOCK_SOURCE1D"), 1)};

  auto particle_group =
      std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);

  const int cell_count = domain->mesh->get_cell_count();
  const int N = npart_per_cell * cell_count;

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells,
                                 rng_pos);

  ParticleSet initial_distribution(N, particle_group->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("POSITION")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("V")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("WEIGHT")][px][0] =
        (px >= N / 2) ? 0.2 : 1.0;
    initial_distribution[Sym<INT>("ID")][px][0] = (px >= N / 2) ? 1 : 2;
    initial_distribution[Sym<INT>("MOCK_INT")][px][0] = 1;
    initial_distribution[Sym<REAL>("MOCK_SOURCE2D")][px][0] = 0.1;
    initial_distribution[Sym<REAL>("MOCK_SOURCE2D")][px][1] = 0.2;
    initial_distribution[Sym<REAL>("MOCK_SOURCE1D")][px][0] = 0.5;
  }
  particle_group->add_particles_local(initial_distribution);

  auto pbc = std::make_shared<CartesianPeriodic>(sycl_target, mesh,
                                                 particle_group->position_dat);
  auto ccb = std::make_shared<CartesianCellBin>(sycl_target, mesh,
                                                particle_group->position_dat,
                                                particle_group->cell_id_dat);

  pbc->execute();
  particle_group->hybrid_move();
  ccb->execute();
  particle_group->cell_move();

  MPI_Barrier(sycl_target->comm_pair.comm_parent);

  return particle_group;
}