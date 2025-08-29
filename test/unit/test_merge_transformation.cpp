#include <gtest/gtest.h>
#include <reactions/reactions.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

auto create_test_particle_group_merging(int N_total, int ndim)
    -> std::shared_ptr<ParticleGroup> {

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

  auto mesh =
      std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                                       subdivision_order, stencil_width);

  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("POSITION"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("WEIGHT"), 1),
                             ParticleProp(Sym<REAL>("VELOCITY"), ndim)};

  auto particle_group =
      std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

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
  auto velocities =
      NESO::Particles::normal_distribution(N, ndim, 0.0, 1.0, rng_vel);
  ParticleSet initial_distribution(N, particle_group->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("POSITION")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("VELOCITY")][px][dimx] =
          velocities.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("WEIGHT")][px][0] = 1.0;
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

TEST(MergeTransformationStrategy, transform_2D) {

  const INT N_total = 1600;

  auto particle_group = create_test_particle_group_merging(N_total, 2);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_merger = MergeTransformationStrategy<2>();

  auto subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  auto reduction = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 5, 1);

  particle_loop(
      subgroup,
      [=](auto W, auto P, auto V, auto GA) {
        GA.fetch_add(0, 0, W[0] * P[0]);
        GA.fetch_add(1, 0, W[0] * P[1]);
        GA.fetch_add(2, 0, W[0] * V[0]);
        GA.fetch_add(3, 0, W[0] * V[1]);
        GA.fetch_add(4, 0, W[0] * (V[0] * V[0] + V[1] * V[1]));
      },
      Access::read(Sym<REAL>("WEIGHT")), Access::read(Sym<REAL>("POSITION")),
      Access::read(Sym<REAL>("VELOCITY")), Access::add(reduction))
      ->execute();
  test_merger.transform(subgroup);

  REAL wt = 100.0;

  for (int ncell = 0; ncell < particle_group->domain->mesh->get_cell_count();
       ncell++) {
    auto reduction_data = reduction->get_cell(ncell);

    EXPECT_EQ(particle_group->get_npart_cell(ncell), 2);

    std::vector<INT> cells = {ncell, ncell};
    std::vector<INT> layers = {0, 1};

    auto particles = particle_group->get_particles(cells, layers);
    REAL energy_tot = reduction_data->at(4, 0);
    REAL energy_merged = 0;
    for (int i = 0; i < 2; i++) {
      EXPECT_NEAR(particles->at(Sym<REAL>("WEIGHT"), i, 0), wt / 2, 1e-12);
      EXPECT_NEAR(particles->at(Sym<REAL>("POSITION"), i, 0),
                  reduction_data->at(0, 0) / wt, 1e-12);
      EXPECT_NEAR(particles->at(Sym<REAL>("POSITION"), i, 1),
                  reduction_data->at(1, 0) / wt, 1e-12);
      energy_merged += particles->at(Sym<REAL>("VELOCITY"), i, 0) *
                           particles->at(Sym<REAL>("VELOCITY"), i, 0) +
                       particles->at(Sym<REAL>("VELOCITY"), i, 1) *
                           particles->at(Sym<REAL>("VELOCITY"), i, 1);

      // Result can be out by as much as ULP=9 so EXPECT_DOUBLE_EQ is not
      // appropriate.
      EXPECT_NEAR(particles->at(Sym<REAL>("VELOCITY"), 0, i) +
                      particles->at(Sym<REAL>("VELOCITY"), 1, i),
                  reduction_data->at(2 + i, 0) * 2 / wt, 1e-12);
    }
    // Result can be out by as much as ULP=7 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(energy_merged * wt / 2, energy_tot, 1e-12);
  }

  particle_group->domain->mesh->free();
}

TEST(MergeTransformationStrategy, transform_3D) {

  const INT N_total = 1600 * 4;

  auto particle_group = create_test_particle_group_merging(N_total, 3);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_merger = MergeTransformationStrategy<3>();

  auto subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  auto reduction = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 7, 1);

  auto red_min = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);
  auto red_max = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);

  red_min->fill(1e16);
  red_max->fill(-1e16);
  particle_loop(
      subgroup,
      [=](auto W, auto P, auto V, auto GA, auto GA_min, auto GA_max) {
        for (int i = 0; i < 3; i++) {
          GA.fetch_add(i, 0, W[0] * P[i]);
          GA.fetch_add(3 + i, 0, W[0] * V[i]);
          GA.fetch_add(6, 0, W[0] * V[i] * V[i]);
          GA_min.fetch_min(i, 0, V[i]);
          GA_max.fetch_max(i, 0, V[i]);
        }
      },
      Access::read(Sym<REAL>("WEIGHT")), Access::read(Sym<REAL>("POSITION")),
      Access::read(Sym<REAL>("VELOCITY")), Access::add(reduction),
      Access::min(red_min), Access::max(red_max))
      ->execute();

  test_merger.transform(subgroup);

  REAL wt = 100.0;

  for (int ncell = 0; ncell < particle_group->domain->mesh->get_cell_count();
       ncell++) {
    auto reduction_data = reduction->get_cell(ncell);
    auto reduction_data_min = red_min->get_cell(ncell);
    auto reduction_data_max = red_max->get_cell(ncell);
    EXPECT_EQ(particle_group->get_npart_cell(ncell), 2);

    std::vector<INT> cells = {ncell, ncell};
    std::vector<INT> layers = {0, 1};

    auto particles = particle_group->get_particles(cells, layers);
    REAL energy_tot = reduction_data->at(6, 0);
    REAL energy_merged = 0;
    std::vector<REAL> diag(3);
    std::vector<REAL> mom_a(3);
    for (int dim = 0; dim < 3; dim++) {
      diag[dim] =
          reduction_data_max->at(dim, 0) - reduction_data_min->at(dim, 0);
      mom_a[dim] = particles->at(Sym<REAL>("VELOCITY"), 0, dim);
    }

    std::vector<REAL> tot_mom_merged = {0, 0, 0};
    for (int i = 0; i < 2; i++) {

      EXPECT_DOUBLE_EQ(particles->at(Sym<REAL>("WEIGHT"), i, 0),
                       wt / 2); //, 1e-12);
      for (int dim = 0; dim < 3; dim++) {
        // Result can be out by as much as ULP=7 so EXPECT_DOUBLE_EQ is not
        // appropriate.
        EXPECT_NEAR(particles->at(Sym<REAL>("POSITION"), i, dim),
                    reduction_data->at(dim, 0) / wt, 1e-12);
        energy_merged += particles->at(Sym<REAL>("VELOCITY"), i, dim) *
                         particles->at(Sym<REAL>("VELOCITY"), i, dim);
        tot_mom_merged[dim] += particles->at(Sym<REAL>("VELOCITY"), i, dim);
      }
    }
    // Result can be out by as much as ULP=5 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(energy_merged * wt / 2, energy_tot, 1e-12);
    for (int dim = 0; dim < 3; dim++) {
      // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
      // appropriate.
      EXPECT_NEAR(tot_mom_merged[dim], reduction_data->at(3 + dim, 0) * 2 / wt,
                  1e-12);
    }

    auto rotation_axis = utils::cross_product(tot_mom_merged, diag);

    // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(std::inner_product(mom_a.begin(), mom_a.end(),
                                   rotation_axis.begin(), 0.0),
                0, 1e-12);
  }

  particle_group->domain->mesh->free();
}
