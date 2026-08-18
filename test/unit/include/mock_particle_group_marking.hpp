#ifndef REACTIONS_MOCK_PARTICLE_GROUP_MARKING_H
#define REACTIONS_MOCK_PARTICLE_GROUP_MARKING_H
#include "test_common.hpp"
#include "test_extern_templates.hpp"

using namespace VANTAGE::Reactions;

auto create_test_particle_group_marking(int N_total)
    -> std::shared_ptr<NP::ParticleGroup> {

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

  auto mesh = std::make_shared<NP::CartesianHMesh>(
      MPI_COMM_WORLD, ndim, dims, cell_extent, subdivision_order,
      stencil_width);

  auto sycl_target = std::make_shared<NP::SYCLTarget>(0, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<NP::Domain>(mesh, cart_local_mapper);

  NP::ParticleSpec particle_spec{
      NP::ParticleProp(NP::Sym<REAL>("POSITION"), ndim, true),
      NP::ParticleProp(NP::Sym<INT>("CELL_ID"), 1, true),
      NP::ParticleProp(NP::Sym<REAL>("WEIGHT"), 1),
      NP::ParticleProp(NP::Sym<INT>("ID"), 1),
      NP::ParticleProp(NP::Sym<INT>("MOCK_INT"), 1),
      NP::ParticleProp(NP::Sym<REAL>("V"), 2),
      NP::ParticleProp(NP::Sym<REAL>("MOCK_SOURCE2D"), 2),
      NP::ParticleProp(NP::Sym<REAL>("MOCK_SOURCE1D"), 1)};

  auto particle_group =
      std::make_shared<NP::ParticleGroup>(domain, particle_spec, sycl_target);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);

  const int cell_count = domain->mesh->get_cell_count();
  const int N = npart_per_cell * cell_count;

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells,
                                 rng_pos);

  NP::ParticleSet initial_distribution(N, particle_group->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[NP::Sym<REAL>("POSITION")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[NP::Sym<REAL>("V")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[NP::Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[NP::Sym<REAL>("WEIGHT")][px][0] =
        (px >= N / 2) ? 0.2 : 1.0;
    initial_distribution[NP::Sym<INT>("ID")][px][0] = (px >= N / 2) ? 1 : 2;
    initial_distribution[NP::Sym<INT>("MOCK_INT")][px][0] = 1;
    initial_distribution[NP::Sym<REAL>("MOCK_SOURCE2D")][px][0] = 0.1;
    initial_distribution[NP::Sym<REAL>("MOCK_SOURCE2D")][px][1] = 0.2;
    initial_distribution[NP::Sym<REAL>("MOCK_SOURCE1D")][px][0] = 0.5;
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