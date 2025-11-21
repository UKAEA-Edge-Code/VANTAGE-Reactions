#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(AbsorptionKernels, general) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto test_data = FixedRateData(1.0);
  auto target_species = Species("ION", 1.2);
  target_species.set_id(0);

  auto properties_map = PropertiesMap();
  properties_map[VANTAGE::Reactions::default_properties.source_density] =
      "ION_SOURCE_DENSITY";
  properties_map[VANTAGE::Reactions::default_properties.source_momentum] =
      "ION_SOURCE_MOMENTUM";
  properties_map[VANTAGE::Reactions::default_properties.source_energy] =
      "ION_SOURCE_ENERGY";

  auto test_kernels =
      GeneralAbsorptionKernels<2>(target_species, properties_map.get_map());

  auto test_reaction =
      LinearReactionBase<0, decltype(test_data), decltype(test_kernels)>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, test_data,
          test_kernels);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto source_density =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_DENSITY"), i);

    auto source_momentum =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_MOMENTUM"), i);

    auto source_energy =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_ENERGY"), i);

    auto position = particle_group->get_cell(Sym<REAL>("POSITION"), i);
    const int nrow = position->nrow;

    auto vel_parent = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    auto weight = particle_group->get_cell(Sym<REAL>("WEIGHT"), i);

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.9);

      for (int dim = 0; dim < 2; dim++) {
        EXPECT_NEAR(source_momentum->at(rowx, dim),
                    0.1 * 1.2 * vel_parent->at(rowx, dim), 1e-14);
      }
      EXPECT_NEAR(source_density->at(rowx, 0), 0.1, 1e-14);
      EXPECT_NEAR(source_energy->at(rowx, 0),
                  0.5 * 0.1 * 1.2 *
                      (std::pow(vel_parent->at(rowx, 0), 2) +
                       std::pow(vel_parent->at(rowx, 1), 2)),
                  1e-14);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
