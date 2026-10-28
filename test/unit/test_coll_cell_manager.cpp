#include "include/mock_particle_group.hpp"
#include "reactions_lib/coll_cell_hierarchies/coll_cell_cartesian_hierarchy.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace VANTAGE::Reactions;

TEST(CollisionCellManager, CartesianHierarchySingleSpecies) {

  const int N_total = 800;

  auto A = create_test_particle_group(N_total);

  auto particle_subgroup = NP::particle_sub_group(A);
  int cell_count = A->domain->mesh->get_cell_count();

  std::vector<int> subdivision_order(cell_count, 1);

  auto coll_cell_h = make_coll_cell_hierarchy<CartesianCollCellH>(
      A->sycl_target,
      std::dynamic_pointer_cast<NP::CartesianHMesh>(A->domain->mesh),
      subdivision_order);

  auto cc_manager =
      CollisionCellManager(A->sycl_target, coll_cell_h, std::vector<INT>{0});

  std::vector<REAL> resolutions(cell_count, 1.0);
  cc_manager.set_coll_cell_linear_resolution(resolutions);
  cc_manager.bin_particles(particle_subgroup);
  cc_manager.construct_cell_partition(particle_subgroup);
  for (int i = 0; i < cell_count; i++) {

    auto collision_cell = A->get_cell(NP::Sym<INT>("COLLISION_CELL"), i);

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
  cc_manager.bin_particles(particle_subgroup);
  cc_manager.construct_cell_partition(particle_subgroup);
  num_coll_cells = cc_manager.get_num_coll_cells();
  volumes = cc_manager.get_coll_cell_volumes();
  npart_coll_cell = cc_manager.get_npart_coll_cell(particle_subgroup, 0);
  EXPECT_EQ(num_coll_cells[0], 4);
  int cell0_sum = 0;
  int cell0_nonempty_subcells = 0;
  for (int i = 0; i < 4; i++) {
    EXPECT_DOUBLE_EQ(volumes->at(0, i), std::pow(0.25, 2));
    cell0_sum += npart_coll_cell->at(0, i);
    if (npart_coll_cell->at(0, i) > 0) {
      cell0_nonempty_subcells++;
    }
  }
  EXPECT_EQ(cell0_sum, particle_subgroup->get_npart_cell(0));
  EXPECT_GT(cell0_nonempty_subcells, 1);

  EXPECT_EQ(num_coll_cells[1], 9);
  int cell1_sum = 0;
  int cell1_nonempty_subcells = 0;
  for (int i = 0; i < 9; i++) {
    EXPECT_DOUBLE_EQ(volumes->at(1, i), std::pow(0.5 / 3, 2));
    cell1_sum += npart_coll_cell->at(1, i);
    if (npart_coll_cell->at(1, i) > 0) {
      cell1_nonempty_subcells++;
    }
  }
  EXPECT_EQ(cell1_sum, particle_subgroup->get_npart_cell(1));
  EXPECT_GT(cell1_nonempty_subcells, 1);

  for (int i = 2; i < cell_count; i++) {
    EXPECT_DOUBLE_EQ(volumes->at(i, 0), 0.25);
    EXPECT_EQ(npart_coll_cell->at(i, 0), particle_subgroup->get_npart_cell(i));
  }

  A->sycl_target->free();
  A->domain->mesh->free();
}

namespace {

struct CountingCartesianHierarchy : CartesianCollCellH {
  using CartesianCollCellH::CartesianCollCellH;
  int bin_calls = 0;

  void bin_particles(NP::ParticleSubGroupSharedPtr target,
                     NP::Sym<INT> coll_cell_sym) override {
    this->bin_calls++;
    CartesianCollCellH::bin_particles(target, coll_cell_sym);
  }
};

} // namespace

TEST(CollisionCellManager, BinningAndPartitionValidity) {
  for (const bool explicit_binning : {false, true}) {
    SCOPED_TRACE(explicit_binning);
    // Nine particles per mesh cell, placed at known offsets in a 3x3 pattern.
    auto group = create_test_particle_group(144);
    auto target = NP::particle_sub_group(group);
    auto mesh =
        std::dynamic_pointer_cast<NP::CartesianHMesh>(group->domain->mesh);
    const int cell_count = mesh->get_cell_count();
    const REAL width = mesh->cell_width_fine;
    std::vector<int> divisions(cell_count, 1);
    auto hierarchy = std::make_shared<CountingCartesianHierarchy>(
        group->sycl_target, mesh, divisions);
    CollisionCellManager manager(group->sycl_target, hierarchy, {0});
    bool mirrored = false;

    auto set_positions = [&]() {
      NP::particle_loop(
          "set_collision_cell_test_positions", group,
          [=](auto position, auto id) {
            for (int dim = 0; dim < 2; dim++) {
              const int digit = (id[0] / (dim == 0 ? 1 : 3)) % 3;
              REAL offset = digit == 0 ? 0.1 : (digit == 1 ? 0.4 : 0.9);
              if (mirrored) {
                offset = 1.0 - offset;
              }
              const REAL origin =
                  NP::Kernel::floor(position[dim] / width) * width;
              position[dim] = origin + width * offset;
            }
          },
          NP::Access::write(NP::Sym<REAL>("POSITION")),
          NP::Access::read(NP::Sym<INT>("ID")))
          ->execute();
    };
    set_positions();

    auto check_partition = [&](int expected_bin_calls) {
      if (explicit_binning) {
        manager.bin_particles(target);
      }
      manager.construct_cell_partition(target);
      EXPECT_EQ(hierarchy->bin_calls, expected_bin_calls);
      auto counts = manager.get_npart_coll_cell(target, 0);
      auto num_cells = manager.get_num_coll_cells();
      auto volumes = manager.get_coll_cell_volumes();
      for (int cell = 0; cell < cell_count; cell++) {
        SCOPED_TRACE(cell);
        const int n = divisions[cell];
        EXPECT_EQ(num_cells[cell], n * n);
        std::vector<int> expected_counts(n * n, 0);
        auto ids = group->get_cell(NP::Sym<INT>("ID"), cell);
        auto bins = group->get_cell(NP::Sym<INT>("COLLISION_CELL"), cell);
        for (int row = 0; row < ids->nrow; row++) {
          int expected_bin = 0;
          for (int dim = 1; dim >= 0; dim--) {
            const int digit = (ids->at(row, 0) / (dim == 0 ? 1 : 3)) % 3;
            REAL offset = digit == 0 ? 0.1 : (digit == 1 ? 0.4 : 0.9);
            if (mirrored) {
              offset = 1.0 - offset;
            }
            expected_bin = expected_bin * n + static_cast<int>(offset * n);
          }
          EXPECT_EQ(bins->at(row, 0), expected_bin);
          expected_counts[expected_bin]++;
        }
        for (int bin = 0; bin < n * n; bin++) {
          EXPECT_EQ(counts->at(cell, bin), expected_counts[bin]);
          EXPECT_DOUBLE_EQ(volumes->at(cell, bin), std::pow(width / n, 2));
        }
      }
    };

    check_partition(1); // Initial construction, including implicit binning.
    manager.bin_particles(target);
    check_partition(1); // Reuse valid bins and partition.

    std::vector<REAL> resolutions(cell_count, width);
    resolutions[0] = width / 2.5;
    divisions[0] = 3;
    manager.set_coll_cell_linear_resolution(resolutions);
    check_partition(2); // Refinement must redistribute particles.
    manager.bin_particles(target);
    check_partition(2); // Construction must not leave stale bins cached.

    resolutions[0] = width / 2.6;
    manager.set_coll_cell_linear_resolution(resolutions);
    check_partition(3); // Resolution changes invalidate even with equal counts.

    resolutions[0] = width;
    divisions[0] = 1;
    manager.set_coll_cell_linear_resolution(resolutions);
    check_partition(4); // Coarsening must replace obsolete collision-cell IDs.

    std::fill(resolutions.begin(), resolutions.end(), width / 2.5);
    std::fill(divisions.begin(), divisions.end(), 3);
    manager.set_coll_cell_linear_resolution(resolutions);
    check_partition(5);
    mirrored = true;
    set_positions();
    manager.invalidate_partition();
    check_partition(6); // Movement must update membership, not just counts.
    check_partition(6);

    resolutions[0] = width / 1.5;
    divisions[0] = 2;
    hierarchy->set_coll_cell_linear_resolution(resolutions);
    check_partition(7); // Detect count changes made directly on the hierarchy.

    group->sycl_target->free();
    mesh->free();
  }
}
