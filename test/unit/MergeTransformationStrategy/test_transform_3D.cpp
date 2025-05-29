#include "test_particle_group_merging.hpp"

using namespace NESO::Particles;
using namespace Reactions;

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