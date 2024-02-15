#include "common_markers.hpp"
#include "common_transformations.hpp"
#include "transformation_wrapper.hpp"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

using namespace NESO::Particles;
using namespace Reactions;

auto create_test_particle_group_marking(int N_total)
    -> shared_ptr<ParticleGroup> {

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

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("WEIGHT"), 1),
                             ParticleProp(Sym<INT>("ID"), 1)};

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
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("WEIGHT")][px][0] =
        (px >= N / 2) ? 0.2 : 1.0;
    initial_distribution[Sym<INT>("ID")][px][0] = (px >= N / 2) ? 1 : 2;
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

TEST(TransformationWrapper, SimpleRemovalTransformationStrategy_less_than) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto test_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{make_marking_strategy<
          ComparisonMarkerSingle<LessThanComp<REAL>, REAL>>(Sym<REAL>("WEIGHT"),
                                                            0.5)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), cellx);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(W->at(rowx, 0), 1.0);
    };
  };

  particle_group->domain->mesh->free();
}

TEST(TransformationWrapper, SimpleRemovalTransformationStrategy_equals) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto test_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(
              Sym<INT>("ID"), 1)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto id = particle_group->get_cell(Sym<INT>("ID"), cellx);
    int nrow = id->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(id->at(rowx, 0), 2);
    };
  };

  particle_group->domain->mesh->free();
}

TEST(TransformationWrapper, SimpleRemovalTransformationStrategy_compose) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto test_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(
              Sym<INT>("ID"), 1),
          make_marking_strategy<
              ComparisonMarkerSingle<LessThanComp<REAL>, REAL>>(
              Sym<REAL>("WEIGHT"), 0.5)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto id = particle_group->get_cell(Sym<INT>("ID"), cellx);
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), cellx);
    int nrow = id->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(id->at(rowx, 0), 2);
      EXPECT_EQ(W->at(rowx, 0), 1.0);
    };
  };

  particle_group->domain->mesh->free();
}