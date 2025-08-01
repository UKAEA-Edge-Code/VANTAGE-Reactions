#include <reactions.hpp>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

auto create_test_particle_group_marking(int N_total)
    -> std::shared_ptr<ParticleGroup> {

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

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("POSITION"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("WEIGHT"), 1),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<INT>("MOCK_INT"), 1),
                             ParticleProp(Sym<REAL>("V"), 2),
                             ParticleProp(Sym<REAL>("MOCK_SOURCE2D"), 2),
                             ParticleProp(Sym<REAL>("MOCK_SOURCE1D"), 1)};

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
      initial_distribution[Sym<REAL>("POSITION")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("V")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("WEIGHT")][px][0] =
        (px >= N / 2) ? 0.2 : 1.0;
    initial_distribution[Sym<INT>("ID")][px][0] = (px >= N / 2) ? 1 : 2;
    initial_distribution[Sym<INT>("MOCK_INT")][px][0] = 1;
    initial_distribution[Sym<REAL>("MOCK_SOURCE2D")][px][0] = 0.1;
    initial_distribution[Sym<REAL>("MOCK_SOURCE2D")][px][1] = 0.2;
    initial_distribution[Sym<REAL>("MOCK_SOURCE1D")][px][0] = 0.5;
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
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
              Sym<REAL>("WEIGHT"), 0.5)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), cellx);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(W->at(rowx, 0), 1.0);
    };
  };

  particle_group->domain->mesh->free();
}

TEST(TransformationWrapper, SimpleRemovalTransformationStrategy_equals) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto test_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
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
          make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
              Sym<INT>("ID"), 1)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());
  test_wrapper.add_marking_strategy(
      make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
          Sym<REAL>("WEIGHT"), 0.5));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto id = particle_group->get_cell(Sym<INT>("ID"), cellx);
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), cellx);
    int nrow = id->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(id->at(rowx, 0), 2);
      EXPECT_DOUBLE_EQ(W->at(rowx, 0), 1.0);
    };
  };

  particle_group->domain->mesh->free();
}

TEST(TransformationWrapper, CompositeTransformZeroer) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto composite = std::make_shared<CompositeTransform>(
      std::vector<std::shared_ptr<TransformationStrategy>>{
          make_transformation_strategy<ParticleDatZeroer<REAL>>(
              std::vector<std::string>{"V"})});
  auto zeroerID = make_transformation_strategy<ParticleDatZeroer<INT>>(
      std::vector<std::string>{"ID"});

  composite->add_transformation(zeroerID);

  auto test_wrapper = TransformationWrapper(
      std::dynamic_pointer_cast<TransformationStrategy>(composite));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto id = particle_group->get_cell(Sym<INT>("ID"), cellx);
    auto V = particle_group->get_cell(Sym<REAL>("V"), cellx);
    int nrow = id->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(id->at(rowx, 0), 0.0);
      EXPECT_DOUBLE_EQ(V->at(rowx, 0), 0.0);
      EXPECT_DOUBLE_EQ(V->at(rowx, 1), 0.0);
    };
  };

  particle_group->domain->mesh->free();
}
TEST(TransformationWrapper, CellwiseAccumulator) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto accumulator_transform = std::make_shared<CellwiseAccumulator<REAL>>(
      particle_group,
      std::vector<std::string>{"MOCK_SOURCE1D", "MOCK_SOURCE2D"});

  auto test_wrapper = TransformationWrapper(
      std::dynamic_pointer_cast<TransformationStrategy>(accumulator_transform));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();
  auto accumulated_1d = accumulator_transform->get_cell_data("MOCK_SOURCE1D");
  auto accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  // The two mock sources have constant values, so we expect those multiplied by
  // the number of particles in the cell
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    // Kept as EXPECT_NEAR for consistency with accumulated_2d checks
    EXPECT_NEAR(accumulated_1d[cellx]->at(0, 0), 0.5 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(0, 0), 0.1 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(1, 0), 0.2 * num_parts, 1e-10);
  };

  // Testing out zeroing features and repeated accumulation

  accumulator_transform->zero_buffer("MOCK_SOURCE2D");
  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_1d[cellx]->at(0, 0), 0.5 * num_parts);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(1, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(0, 0), 0);
  };

  test_wrapper.transform(particle_group);
  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  accumulated_1d = accumulator_transform->get_cell_data("MOCK_SOURCE1D");
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    // Kept as EXPECT_NEAR for consistency with accumulated_2d checks
    EXPECT_NEAR(accumulated_1d[cellx]->at(0, 0), 1.0 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(0, 0), 0.1 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(1, 0), 0.2 * num_parts, 1e-10);
  };

  accumulator_transform->zero_all_buffers();

  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  accumulated_1d = accumulator_transform->get_cell_data("MOCK_SOURCE1D");
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_1d[cellx]->at(0, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(1, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(0, 0), 0);
  };

  // Testing out setting modified cell data

  test_wrapper.transform(particle_group);

  double scale_1d = 2.5;
  double scale_2d = 3.7;
  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  accumulated_1d = accumulator_transform->get_cell_data("MOCK_SOURCE1D");
  for (int cellx = 0; cellx < num_cells; cellx++) {
    accumulated_1d[cellx]->at(0, 0) *= scale_1d;
    accumulated_2d[cellx]->at(0, 0) *= scale_2d;
    accumulated_2d[cellx]->at(1, 0) *= scale_2d;
  }

  accumulator_transform->set_cell_data("MOCK_SOURCE1D", accumulated_1d);
  accumulator_transform->set_cell_data("MOCK_SOURCE2D", accumulated_2d);

  auto updated_accumulated_1d =
      accumulator_transform->get_cell_data("MOCK_SOURCE1D");
  auto updated_accumulated_2d =
      accumulator_transform->get_cell_data("MOCK_SOURCE2D");

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_NEAR(updated_accumulated_1d[cellx]->at(0, 0),
                scale_1d * 0.5 * num_parts, 1e-10);
    EXPECT_NEAR(updated_accumulated_2d[cellx]->at(0, 0),
                scale_2d * 0.1 * num_parts, 1e-10);
    EXPECT_NEAR(updated_accumulated_2d[cellx]->at(1, 0),
                scale_2d * 0.2 * num_parts, 1e-10);
  }

  particle_group->domain->mesh->free();
}
TEST(TransformationWrapper, WeightedCellwiseAccumulator) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto accumulator_transform =
      std::make_shared<WeightedCellwiseAccumulator<REAL>>(
          particle_group, std::vector<std::string>{"MOCK_SOURCE2D"},
          "MOCK_SOURCE1D");

  auto test_wrapper = TransformationWrapper(
      std::dynamic_pointer_cast<TransformationStrategy>(accumulator_transform));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();
  auto accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  auto accumulated_weight = accumulator_transform->get_weight_cell_data();
  // Here we use the 1D mock source as a weight for this test (because there are
  // different weights in the default particle group used in this set of tests)
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    // Kept as EXPECT_NEAR for consistency with accumulated_2d checks
    EXPECT_NEAR(accumulated_weight[cellx]->at(0, 0), 0.5 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(0, 0), 0.5 * 0.1 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(1, 0), 0.5 * 0.2 * num_parts, 1e-10);
  };
  // Testing out zeroing features and repeated accumulation

  accumulator_transform->zero_buffer("MOCK_SOURCE2D");
  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_weight[cellx]->at(0, 0), 0.5 * num_parts);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(1, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(0, 0), 0);
  };

  test_wrapper.transform(particle_group);
  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  accumulated_weight = accumulator_transform->get_weight_cell_data();
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    // Kept as EXPECT_NEAR for consistency with accumulated_2d checks
    EXPECT_NEAR(accumulated_weight[cellx]->at(0, 0), 1.0 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(0, 0), 0.5 * 0.1 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(1, 0), 0.5 * 0.2 * num_parts, 1e-10);
  };

  accumulator_transform->zero_all_buffers();

  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  accumulated_weight = accumulator_transform->get_weight_cell_data();
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_weight[cellx]->at(0, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(1, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(0, 0), 0);
  };

  particle_group->domain->mesh->free();
}

TEST(TransformationWrapper, CellwiseAccumulatorINT) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto accumulator_transform = std::make_shared<CellwiseAccumulator<INT>>(
      particle_group, std::vector<std::string>{"MOCK_INT"});

  auto test_wrapper = TransformationWrapper(
      std::dynamic_pointer_cast<TransformationStrategy>(accumulator_transform));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();
  auto accumulated_1d = accumulator_transform->get_cell_data("MOCK_INT");
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_1d[cellx]->at(0, 0), num_parts); //, 1e-10);
  };

  particle_group->domain->mesh->free();
}
TEST(TransformationWrapper, WeightedCellwiseAccumulatorINT) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto accumulator_transform =
      std::make_shared<WeightedCellwiseAccumulator<INT>>(
          particle_group, std::vector<std::string>{"MOCK_INT"},
          "MOCK_SOURCE1D");

  auto test_wrapper = TransformationWrapper(
      std::dynamic_pointer_cast<TransformationStrategy>(accumulator_transform));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();
  auto accumulated_1d = accumulator_transform->get_cell_data("MOCK_INT");
  auto accumulated_weight = accumulator_transform->get_weight_cell_data();
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_weight[cellx]->at(0, 0),
                     0.5 * num_parts); //, 1e-10);
    EXPECT_DOUBLE_EQ(accumulated_1d[cellx]->at(0, 0),
                     0.5 * num_parts); //, 1e-10);
  };

  particle_group->domain->mesh->free();
}
