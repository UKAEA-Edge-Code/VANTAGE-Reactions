#include "include/mock_particle_group.hpp"
#include "reactions_lib/coll_cell_hierarchies/coll_cell_cartesian_hierarchy.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(CollisionCellManager, CartesianHierarchySingleSpecies) {

  const int N_total = 800;

  auto A = create_test_particle_group(N_total);

  auto particle_subgroup = particle_sub_group(A);
  int cell_count = A->domain->mesh->get_cell_count();

  std::vector<int> subdivision_order(cell_count, 1);

  auto coll_cell_h = make_coll_cell_hierarchy<CartesianCollCellH>(
      A->sycl_target,
      std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
      subdivision_order);

  auto cc_manager =
      CollisionCellManager(A->sycl_target, coll_cell_h, std::vector<INT>{0});

  std::vector<REAL> resolutions(cell_count, 1.0);
  cc_manager.set_coll_cell_linear_resolution(resolutions);
  cc_manager.bin_particles(particle_subgroup);
  cc_manager.construct_cell_partition(particle_subgroup);
  for (int i = 0; i < cell_count; i++) {

    auto collision_cell = A->get_cell(Sym<INT>("COLLISION_CELL"), i);

    const int nrow = collision_cell->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(collision_cell->at(rowx, 0), 0);
    }
  }

  auto num_coll_cells = cc_manager.get_num_coll_cells();

  for (auto coll_cells : num_coll_cells) {
    EXPECT_EQ(coll_cells, 1);
  }

  auto volumes = cc_manager.get_coll_cell_volumes();

  for (int i = 0; i < cell_count; i++) {
    EXPECT_DOUBLE_EQ(volumes->at(i, 0), 0.25);
  }

  auto npart_coll_cell = cc_manager.get_npart_coll_cell(particle_subgroup, 0);

  for (int i = 0; i < cell_count; i++) {
    EXPECT_EQ(npart_coll_cell->at(i, 0),
              particle_subgroup->get_npart_local() / cell_count);
  }

  resolutions[0] = 0.4;
  resolutions[1] = 0.2;

  cc_manager.set_coll_cell_linear_resolution(resolutions);
  cc_manager.construct_cell_partition(particle_subgroup);
  num_coll_cells = cc_manager.get_num_coll_cells();
  volumes = cc_manager.get_coll_cell_volumes();
  EXPECT_EQ(num_coll_cells[0], 4);
  for (int i = 0; i < 4; i++) {
    EXPECT_DOUBLE_EQ(volumes->at(0, i), std::pow(0.25, 2));
  }
  EXPECT_EQ(num_coll_cells[1], 9);
  for (int i = 0; i < 9; i++) {
    EXPECT_DOUBLE_EQ(volumes->at(1, i), std::pow(0.5 / 3, 2));
  }
  for (int i = 2; i < cell_count; i++) {
    EXPECT_DOUBLE_EQ(volumes->at(i, 0), 0.25);
  }
}
