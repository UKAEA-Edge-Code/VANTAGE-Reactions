#include "test_particle_group_merging.hpp"

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

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
      EXPECT_NEAR(particles->at(Sym<REAL>("WEIGHT"), i, 0),
                       wt / 2, 1e-12);
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