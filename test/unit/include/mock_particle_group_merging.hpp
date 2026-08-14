#ifndef REACTIONS_MOCK_PARTICLE_GROUP_MERGING_H
#define REACTIONS_MOCK_PARTICLE_GROUP_MERGING_H
#include "test_common.hpp"
#include "test_extern_templates.hpp"

using namespace VANTAGE::Reactions;

auto create_test_particle_group_merging(int N_total, int ndim)
    -> std::shared_ptr<NP::ParticleGroup> {

  std::vector<int> dims(ndim);
  for (int dim = 0; dim < ndim; dim++) {
    dims[dim] = 2;
  }

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;

  const int global_cell_count =
      std::pow(2 * std::pow(2, subdivision_order), ndim);
  const int npart_per_cell =
      std::round((double)N_total / (double)global_cell_count);

  auto mesh = std::make_shared<NP::CartesianHMesh>(
      MPI_COMM_WORLD, ndim, dims, cell_extent, subdivision_order,
      stencil_width);

  auto sycl_target = std::make_shared<NP::SYCLTarget>(0, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<NP::Domain>(mesh, cart_local_mapper);

  NP::ParticleSpec particle_spec{
      NP::ParticleProp(NP::Sym<NP::REAL>("POSITION"), ndim, true),
      NP::ParticleProp(NP::Sym<NP::INT>("CELL_ID"), 1, true),
      NP::ParticleProp(NP::Sym<NP::REAL>("WEIGHT"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("VELOCITY"), ndim)};

  auto particle_group =
      std::make_shared<NP::ParticleGroup>(domain, particle_spec, sycl_target);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  const int cell_count = domain->mesh->get_cell_count();
  const int N = npart_per_cell * cell_count;

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells,
                                 rng_pos);
  auto velocities = NP::normal_distribution(N, ndim, 0.0, 1.0, rng_vel);
  NP::ParticleSet initial_distribution(N, particle_group->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[NP::Sym<NP::REAL>("POSITION")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[NP::Sym<NP::REAL>("VELOCITY")][px][dimx] =
          velocities.at(dimx).at(px);
    }
    initial_distribution[NP::Sym<NP::INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[NP::Sym<NP::REAL>("WEIGHT")][px][0] = 1.0;
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