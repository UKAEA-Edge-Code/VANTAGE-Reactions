
#include "reactions/neso_particles_namespace_alias.hpp"
#include <gtest/gtest.h>
#include <reactions/reactions.hpp>

using namespace VANTAGE::Reactions;

auto create_vranic_test_particle_group(int N_total, int ndim)
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
      NP::ParticleProp(NP::Sym<NP::REAL>("VELOCITY"), ndim),
      NP::ParticleProp(NP::Sym<NP::INT>("REACTIONS_GROUPING_INDEX"), 1),
      NP::ParticleProp(NP::Sym<NP::INT>("REACTIONS_LINEAR_INDEX"), 1),
  };

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
  auto velocities =
      NESO::Particles::normal_distribution(N, ndim, 0.0, 1.0, rng_vel);
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

TEST(VranicMergeTransform, transform_2D) {

  const NP::INT N_total = 1600;

  auto particle_group = create_vranic_test_particle_group(N_total, 2);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_merger = make_vranic_merging_strategy<2>(particle_group, 1);

  auto subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto reduction = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);

  particle_loop(
      subgroup,
      [=](auto W, auto V, auto GA) {
        GA.fetch_add(0, 0, W[0] * V[0]);
        GA.fetch_add(1, 0, W[0] * V[1]);
        GA.fetch_add(2, 0, W[0] * (V[0] * V[0] + V[1] * V[1]));
      },
      NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
      NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")),
      NP::Access::add(reduction))
      ->execute();
  test_merger->transform(subgroup);

  NP::REAL wt = 100.0;

  for (int ncell = 0; ncell < cell_count; ncell++) {
    auto reduction_data = reduction->get_cell(ncell);

    EXPECT_EQ(particle_group->get_npart_cell(ncell), 2);

    std::vector<NP::INT> cells = {ncell, ncell};
    std::vector<NP::INT> layers = {0, 1};

    auto particles = particle_group->get_particles(cells, layers);
    NP::REAL energy_tot = reduction_data->at(2, 0);
    NP::REAL energy_merged = 0;
    for (int i = 0; i < 2; i++) {
      EXPECT_NEAR(particles->at(NP::Sym<NP::REAL>("WEIGHT"), i, 0), wt / 2,
                  1e-12);
      energy_merged += particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, 0) *
                           particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, 0) +
                       particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, 1) *
                           particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, 1);

      // Result can be out by as much as ULP=9 so EXPECT_DOUBLE_EQ is not
      // appropriate.
      EXPECT_NEAR(particles->at(NP::Sym<NP::REAL>("VELOCITY"), 0, i) +
                      particles->at(NP::Sym<NP::REAL>("VELOCITY"), 1, i),
                  reduction_data->at(i, 0) * 2 / wt, 1e-12);
    }
    // Result can be out by as much as ULP=7 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(energy_merged * wt / 2, energy_tot, 1e-12);
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(VranicMergeTransform, transform_zero_momentum_2D) {

  const NP::INT N_total = 1600;

  auto particle_group = create_vranic_test_particle_group(N_total, 2);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_merger = make_vranic_merging_strategy<2>(particle_group, 1);
  auto subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto reduction = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
      subgroup,
      [=](auto V) {
        for (int dx = 0; dx < 2; dx++) {
          V.at(dx) = 0.0;
        }
      },
      NP::Access::write(NP::Sym<NP::REAL>("VELOCITY")))
      ->execute();

  particle_loop(
      subgroup,
      [=](auto W, auto V, auto GA) {
        GA.fetch_add(0, 0, W[0] * (V[0] * V[0] + V[1] * V[1]));
      },
      NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
      NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")),
      NP::Access::add(reduction))
      ->execute();
  test_merger->transform(subgroup);

  NP::REAL wt = 100.0;

  for (int ncell = 0; ncell < cell_count; ncell++) {
    auto reduction_data = reduction->get_cell(ncell);

    EXPECT_EQ(particle_group->get_npart_cell(ncell), 2);

    std::vector<NP::INT> cells = {ncell, ncell};
    std::vector<NP::INT> layers = {0, 1};

    auto particles = particle_group->get_particles(cells, layers);
    NP::REAL energy_tot = reduction_data->at(0, 0);

    EXPECT_NEAR(energy_tot, 0.0, 1e-12);
    for (int i = 0; i < 2; i++) {
      EXPECT_NEAR(particles->at(NP::Sym<NP::REAL>("WEIGHT"), i, 0), wt / 2,
                  1e-12);

      EXPECT_NEAR(particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, 0), 0.0,
                  1.0e-15);
      EXPECT_NEAR(particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, 1), 0.0,
                  1.0e-15);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(VranicMergeTransform, transform_3D) {

  const NP::INT N_total = 1600 * 4;

  auto particle_group = create_vranic_test_particle_group(N_total, 3);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_merger = make_vranic_merging_strategy<3>(particle_group, 1);

  auto subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto reduction = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 4, 1);

  auto red_min = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);
  auto red_max = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);

  red_min->fill(1e16);
  red_max->fill(-1e16);
  particle_loop(
      subgroup,
      [=](auto W, auto V, auto GA, auto GA_min, auto GA_max) {
        for (int i = 0; i < 3; i++) {
          GA.fetch_add(i, 0, W[0] * V[i]);
          GA.fetch_add(3, 0, W[0] * V[i] * V[i]);
          GA_min.fetch_min(i, 0, V[i]);
          GA_max.fetch_max(i, 0, V[i]);
        }
      },
      NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
      NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")),
      NP::Access::add(reduction), NP::Access::min(red_min),
      NP::Access::max(red_max))
      ->execute();

  test_merger->transform(subgroup);

  NP::REAL wt = 100.0;

  for (int ncell = 0; ncell < cell_count; ncell++) {
    auto reduction_data = reduction->get_cell(ncell);
    auto reduction_data_min = red_min->get_cell(ncell);
    auto reduction_data_max = red_max->get_cell(ncell);
    EXPECT_EQ(particle_group->get_npart_cell(ncell), 2);

    std::vector<NP::INT> cells = {ncell, ncell};
    std::vector<NP::INT> layers = {0, 1};

    auto particles = particle_group->get_particles(cells, layers);
    NP::REAL energy_tot = reduction_data->at(3, 0);
    NP::REAL energy_merged = 0;
    std::vector<NP::REAL> diag(3);
    std::vector<NP::REAL> mom_a(3);
    for (int dim = 0; dim < 3; dim++) {
      diag[dim] =
          reduction_data_max->at(dim, 0) - reduction_data_min->at(dim, 0);
      mom_a[dim] = particles->at(NP::Sym<NP::REAL>("VELOCITY"), 0, dim);
    }

    std::vector<NP::REAL> tot_mom_merged = {0, 0, 0};
    for (int i = 0; i < 2; i++) {

      EXPECT_DOUBLE_EQ(particles->at(NP::Sym<NP::REAL>("WEIGHT"), i, 0),
                       wt / 2); //, 1e-12);
      for (int dim = 0; dim < 3; dim++) {
        energy_merged += particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, dim) *
                         particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, dim);
        tot_mom_merged[dim] +=
            particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, dim);
      }
    }
    // Result can be out by as much as ULP=5 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(energy_merged * wt / 2, energy_tot, 1e-12);
    for (int dim = 0; dim < 3; dim++) {
      // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
      // appropriate.
      EXPECT_NEAR(tot_mom_merged[dim], reduction_data->at(dim, 0) * 2 / wt,
                  1e-12);
    }

    auto rotation_axis = utils::cross_product(tot_mom_merged, diag);

    // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(std::inner_product(mom_a.begin(), mom_a.end(),
                                   rotation_axis.begin(), 0.0),
                0, 1e-12);
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(VranicMergeTransform, transform_zero_momentum_3D) {

  const NP::INT N_total = 1600 * 4;

  auto particle_group = create_vranic_test_particle_group(N_total, 3);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_merger = make_vranic_merging_strategy<3>(particle_group, 1);
  auto subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);

  particle_loop(
      subgroup,
      [=](auto V) {
        for (int dx = 0; dx < 3; dx++) {
          V.at(dx) = 0.0;
        }
      },
      NP::Access::write(NP::Sym<NP::REAL>("VELOCITY")))
      ->execute();

  auto reduction = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 4, 1);

  auto red_min = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);
  auto red_max = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);

  red_min->fill(1e16);
  red_max->fill(-1e16);
  particle_loop(
      subgroup,
      [=](auto W, auto V, auto GA, auto GA_min, auto GA_max) {
        for (int i = 0; i < 3; i++) {
          GA.fetch_add(i, 0, W[0] * V[i]);
          GA.fetch_add(3, 0, W[0] * V[i] * V[i]);
          GA_min.fetch_min(i, 0, V[i]);
          GA_max.fetch_max(i, 0, V[i]);
        }
      },
      NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
      NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")),
      NP::Access::add(reduction), NP::Access::min(red_min),
      NP::Access::max(red_max))
      ->execute();

  test_merger->transform(subgroup);

  NP::REAL wt = 100.0;

  for (int ncell = 0; ncell < cell_count; ncell++) {
    auto reduction_data = reduction->get_cell(ncell);
    auto reduction_data_min = red_min->get_cell(ncell);
    auto reduction_data_max = red_max->get_cell(ncell);
    EXPECT_EQ(particle_group->get_npart_cell(ncell), 2);

    std::vector<NP::INT> cells = {ncell, ncell};
    std::vector<NP::INT> layers = {0, 1};

    auto particles = particle_group->get_particles(cells, layers);
    NP::REAL energy_tot = reduction_data->at(3, 0);
    EXPECT_NEAR(energy_tot, 0.0, 1.0e-15);
    for (int dim = 0; dim < 3; dim++) {
      EXPECT_NEAR(reduction_data_max->at(dim, 0), 0.0, 1.0e-15);
      EXPECT_NEAR(reduction_data_min->at(dim, 0), 0.0, 1.0e-15);
    }

    for (int i = 0; i < 2; i++) {

      EXPECT_DOUBLE_EQ(particles->at(NP::Sym<NP::REAL>("WEIGHT"), i, 0),
                       wt / 2); //, 1e-12);
      for (int dim = 0; dim < 3; dim++) {
        // Result can be out by as much as ULP=7 so EXPECT_DOUBLE_EQ is not
        // appropriate.
        EXPECT_NEAR(particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, dim), 0.0,
                    1e-15);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(VranicMergeTransform, transform_3D_simple_grouping) {

  const NP::INT N_total = 1600 * 4;

  auto particle_group = create_vranic_test_particle_group(N_total, 3);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_merger = make_vranic_merging_strategy<3>(particle_group, 2);

  auto subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);

  particle_loop(
      "set_grouping", subgroup,
      [=](auto grouping_index, auto velocity) {
        grouping_index[0] = velocity[0] > 0 ? 1 : 0;
      },
      NP::Access::write(NP::Sym<NP::INT>("REACTIONS_GROUPING_INDEX")),
      NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")))
      ->execute();

  auto reduction = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 5, 2);

  particle_loop(
      subgroup,
      [=](auto W, auto V, auto GA, auto grouping_index) {
        for (int i = 0; i < 3; i++) {
          GA.fetch_add(i, grouping_index[0], W[0] * V[i]);
          GA.fetch_add(3, grouping_index[0], W[0] * V[i] * V[i]);
        }
        GA.fetch_add(4, grouping_index[0], W[0]);
      },
      NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
      NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")),
      NP::Access::add(reduction),
      NP::Access::read(NP::Sym<NP::INT>("REACTIONS_GROUPING_INDEX")))
      ->execute();

  test_merger->transform(subgroup);

  for (int ncell = 0; ncell < cell_count; ncell++) {
    auto reduction_data = reduction->get_cell(ncell);
    EXPECT_EQ(particle_group->get_npart_cell(ncell), 4);

    std::vector<NP::INT> cells = {ncell, ncell, ncell, ncell};
    std::vector<NP::INT> layers = {0, 1, 2, 3};

    auto particles = particle_group->get_particles(cells, layers);
    for (auto group = 0; group < 1; group++) {
      NP::REAL energy_tot = reduction_data->at(3, group);
      NP::REAL wt = reduction_data->at(4, group);
      NP::REAL energy_merged = 0;

      std::vector<NP::REAL> tot_mom_merged = {0, 0, 0};
      for (int i = 0; i < 4; i++) {
        if (particles->at(NP::Sym<NP::INT>("REACTIONS_GROUPING_INDEX"), i, 0) ==
            group) {

          EXPECT_DOUBLE_EQ(particles->at(NP::Sym<NP::REAL>("WEIGHT"), i, 0),
                           wt / 2); //, 1e-12);
          for (int dim = 0; dim < 3; dim++) {
            energy_merged +=
                particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, dim) *
                particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, dim);
            tot_mom_merged[dim] +=
                particles->at(NP::Sym<NP::REAL>("VELOCITY"), i, dim);
          }
        }
      }
      // Result can be out by as much as ULP=5 so EXPECT_DOUBLE_EQ is not
      // appropriate.
      EXPECT_NEAR(energy_merged * wt / 2, energy_tot, 1e-12);
      for (int dim = 0; dim < 3; dim++) {
        // Result can be out by as much as ULP>10 so EXPECT_DOUBLE_EQ is not
        // appropriate.
        EXPECT_NEAR(tot_mom_merged[dim],
                    reduction_data->at(dim, group) * 2 / wt, 1e-12);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(UniformVelocityBin, utility_function) {

  EXPECT_EQ(utils::bin_uniform_symmetric_guard_1d(0.1, 10, -4.5), 1);
  EXPECT_EQ(utils::bin_uniform_symmetric_guard_1d(0.1, 10, 1.0), 6);
  EXPECT_EQ(utils::bin_uniform_symmetric_guard_1d(0.1, 10, 100.0), 11);
  EXPECT_EQ(utils::bin_uniform_symmetric_guard_1d(0.1, 10, -100.0), 0);
}

TEST(VranicMergeTransform, transform_3D_velocity_binning) {

  const NP::INT N_total = 1600 * 4;

  auto particle_group = create_vranic_test_particle_group(N_total, 3);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_merger =
      make_vranic_merging_strategy<3>(particle_group, 12 * 12 * 12);

  auto subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto velocity_bin = uniform_velocity_bin_transform<3>(
      std::array<NP::REAL, 3>{3.0, 3.0, 3.0},
      std::array<NP::INT, 3>{10, 10, 10},
      NP::Sym<NP::INT>("REACTIONS_GROUPING_INDEX"),
      NP::Sym<NP::REAL>("VELOCITY"));

  auto reduction = std::make_shared<NP::CellDatConst<NP::REAL>>(
      particle_group->sycl_target, cell_count, 5, 12 * 12 * 12);

  velocity_bin->transform(subgroup);
  particle_loop(
      subgroup,
      [=](auto W, auto V, auto GA, auto grouping_index) {
        for (int i = 0; i < 3; i++) {
          GA.fetch_add(i, grouping_index[0], W[0] * V[i]);
          GA.fetch_add(3, grouping_index[0], W[0] * V[i] * V[i]);
        }
        GA.fetch_add(4, grouping_index[0], W[0]);
      },
      NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
      NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")),
      NP::Access::add(reduction),
      NP::Access::read(NP::Sym<NP::INT>("REACTIONS_GROUPING_INDEX")))
      ->execute();

  test_merger->transform(subgroup);

  particle_loop(
      subgroup,
      [=](auto W, auto V, auto GA, auto grouping_index) {
        for (int i = 0; i < 3; i++) {
          GA.fetch_add(i, grouping_index[0], -W[0] * V[i]);
          GA.fetch_add(3, grouping_index[0], -W[0] * V[i] * V[i]);
        }
        GA.fetch_add(4, grouping_index[0], -W[0]);
      },
      NP::Access::read(NP::Sym<NP::REAL>("WEIGHT")),
      NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")),
      NP::Access::add(reduction),
      NP::Access::read(NP::Sym<NP::INT>("REACTIONS_GROUPING_INDEX")))
      ->execute();

  for (int ncell = 0; ncell < cell_count; ncell++) {
    auto reduction_data = reduction->get_cell(ncell);

    for (auto group = 0; group < 12 * 12 * 12; group++) {
      for (auto i = 0; i < 5; i++) {

        EXPECT_NEAR(reduction_data->at(i, group), 0, 1e-12);
      }
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
