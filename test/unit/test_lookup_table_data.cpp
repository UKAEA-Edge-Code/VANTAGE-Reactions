#include "include/mock_particle_group.hpp"
#include "include/mock_reactions.hpp"
#include "reactions_lib/reaction_data/array_lookup_data.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(LookupTableData, ArrayLookupTable) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto default_array = std::array<REAL, 1>{1.0};
  std::map<int, std::array<REAL, 1>> lookup_table_map;

  lookup_table_map[0] = std::array<REAL, 1>{2.0};
  lookup_table_map[1] = std::array<REAL, 1>{3.0};

  particle_loop(
      "set_array_lookup_table_test_ids", particle_sub_group,
      [=](auto ID, auto IS) { IS.at(0) = ID.at(0) % 3; },
      Access::read(Sym<INT>("ID")), Access::write(Sym<INT>("INTERNAL_STATE")))
      ->execute();

  auto test_reaction =
      LinearReactionBase<0, ArrayLookupData<1>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{},
          ArrayLookupData<1>(Sym<INT>("INTERNAL_STATE"), 0, lookup_table_map,
                             default_array, particle_group->sycl_target),
          TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    auto is = particle_group->get_cell(Sym<INT>("INTERNAL_STATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      switch (is->at(rowx, 0)) {
      case 0:
        EXPECT_DOUBLE_EQ(rate->at(rowx, 0), 2.0);
        break;
      case 1:
        EXPECT_DOUBLE_EQ(rate->at(rowx, 0), 3.0);
        break;
      default:
        EXPECT_DOUBLE_EQ(rate->at(rowx, 0), 1.0);
        break;
      }
    }
  }

  particle_group->domain->mesh->free();
}

TEST(LookupTableData, ArrayLookupDataEphemeralKey) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto default_array = std::array<REAL, 1>{1.0};
  std::map<int, std::array<REAL, 1>> lookup_table_map;

  lookup_table_map[0] = std::array<REAL, 1>{2.0};
  lookup_table_map[1] = std::array<REAL, 1>{3.0};

  int cell_count = particle_group->domain->mesh->get_cell_count();

  // Add data to subgroup
  particle_sub_group->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_metadata,
      BoundaryInteractionSpecification::intersection_metadata_ncomp);

  particle_loop(
      "set_array_lookup_table_test_metadaat", particle_sub_group,
      [=](auto ID, auto metadata) {
        metadata.at(1) = ID.at(0) % 3;
        metadata.at(0) = -1;
      },
      Access::read(Sym<INT>("ID")),
      Access::write(BoundaryInteractionSpecification::intersection_metadata))
      ->execute();

  auto test_reaction =
      LinearReactionBase<0, ArrayLookupData<1, true>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{},
          ArrayLookupData<1, true>(
              BoundaryInteractionSpecification::intersection_metadata, 1,
              lookup_table_map, default_array, particle_group->sycl_target),
          TestReactionKernels<0>());

  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    auto id = particle_group->get_cell(Sym<INT>("ID"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      switch (id->at(rowx, 0) % 3) {
      case 0:
        EXPECT_DOUBLE_EQ(rate->at(rowx, 0), 2.0);
        break;
      case 1:
        EXPECT_DOUBLE_EQ(rate->at(rowx, 0), 3.0);
        break;
      default:
        EXPECT_DOUBLE_EQ(rate->at(rowx, 0), 1.0);
        break;
      }
    }
  }

  particle_group->domain->mesh->free();
}
