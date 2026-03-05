#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(SimpleThinningTransform, deterministic_rng_test) {

  const INT N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto rng_lambda = [&]() -> REAL { return 0.6; };

  auto rng_kernel =
      NESO::Particles::host_per_particle_block_rng<REAL>(rng_lambda, 1);

  auto test_thinner =
      make_simple_thinning_strategy(particle_group, 1, 0.5, rng_kernel);

  auto subgroup = particle_sub_group(particle_group);

  test_thinner->transform(subgroup);

  for (int i = 0; i < cell_count; i++) {

    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);
    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 2.0);
    }
  }

  auto test_thinner_all_remove =
      make_simple_thinning_strategy(particle_group, 1, 0.01, rng_kernel);

  test_thinner_all_remove->transform(subgroup);
  EXPECT_EQ(particle_group->get_npart_local(), 0);

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
