#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(SWPMReactionController, single_reaction) {

  const int N_total = 800;

  auto test_rng_lambda = [=](auto dt, auto rng_val) {
    auto A = create_test_particle_group(N_total);
    auto B = create_test_particle_group(N_total);

    particle_loop(
        "set_vels_ids_a", A,
        [=](auto vel, auto sp_id, auto id) {
          vel[0] = 2;
          vel[1] = 0;
        },
        Access::write(Sym<REAL>("VELOCITY")),
        Access::write(Sym<INT>("INTERNAL_STATE")), Access::read(Sym<INT>("ID")))
        ->execute();

    particle_loop(
        "set_vels_ids_b", B,
        [=](auto vel, auto sp_id, auto id) {
          sp_id[0] = 1;
          vel[0] = 4;
          vel[1] = 0;
        },
        Access::write(Sym<REAL>("VELOCITY")),
        Access::write(Sym<INT>("INTERNAL_STATE")), Access::read(Sym<INT>("ID")))
        ->execute();

    A->add_particles_local(B);
    auto particle_subgroup = particle_sub_group(A);

    int cell_count = A->domain->mesh->get_cell_count();
    std::vector<int> subdivision_order(cell_count, 1);

    auto coll_cell_h = make_coll_cell_hierarchy<CartesianCollCellH>(
        A->sycl_target,
        std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
        subdivision_order);

    auto cc_manager = std::make_shared<CollisionCellManager>(
        A->sycl_target, coll_cell_h, std::vector<INT>{0, 1});

    auto lambda_remove = [](auto target) {
      target->get_particle_group()->remove_particles(target);
    };

    auto lambda_marker = [](auto w) { return w[0] < 1e-12; };
    auto accessor = Access::read(Sym<REAL>("WEIGHT"));
    auto test_removal_wrapper = std::make_shared<TransformationWrapper>(
        std::vector<std::shared_ptr<MarkingStrategy>>{
            make_direct_marking_strategy("small", lambda_marker, accessor)},
        make_lambda_transformation_strategy("remove", lambda_remove));

    std::mt19937 rng_state(52234234);
    std::uniform_real_distribution<> rng_dist(0.0, 1.0);
    auto rng_lambda_sampler = [&]() -> REAL { return rng_dist(rng_state); };

    auto rng_function =
        std::make_shared<HostRNGGenerationFunction<REAL>>(rng_lambda_sampler);
    // Mocking the species to test the correct reduced mass effects
    auto species_1 = Species("ION", 1.2, 0.0, 0);
    auto species_2 = Species("ION2", 2.0, 0.0, 1);

    auto rng_lambda = [&]() -> REAL { return rng_val; };

    auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 1);
    auto spec = std::make_shared<SWPMDSMCSpecification>(rng_kernel);
    auto controller = SWPMReactionController(
        species_1.get_id(), species_2.get_id(), spec, rng_function, cc_manager,
        std::vector<std::shared_ptr<TransformationWrapper>>{
            test_removal_wrapper},
        std::vector<std::shared_ptr<TransformationWrapper>>{
            test_removal_wrapper});

    auto cs_data =
        CSPairData<2, ConstantCrossSection>(ConstantCrossSection(0.0005, 1.0));

    auto rng_lambda_hs = [&]() -> REAL { return 0.6; };

    auto rng_kernel_hs = host_atomic_block_kernel_rng<REAL>(rng_lambda_hs, 2);
    auto hs_scattering_data =
        HSScatteringData<2>(species_1, species_2, rng_kernel_hs);
    auto pair_data_calculator = PairDataCalculator(hs_scattering_data);

    auto scattering_kernels = PairScatteringKernels<2>();

    auto swpm_reaction = std::make_shared<
        SWPMReaction<2, decltype(cs_data), decltype(scattering_kernels),
                     decltype(pair_data_calculator)>>(
        A->sycl_target,
        std::array<int, 2>{static_cast<int>(species_1.get_id()),
                           static_cast<int>(species_2.get_id())},
        std::array<int, 2>{2, 3}, cs_data, scattering_kernels,
        pair_data_calculator);

    controller.add_reaction(swpm_reaction);

    controller.apply(A, dt);

    return A;
  };

  auto A = test_rng_lambda(0.1, 0.1);
  int cell_count = A->domain->mesh->get_cell_count();
  // Expected velocities:
  // Centre-of-mass velocity: (1.2*2+2*4)/3.2 = 3.25
  // Relative velocity: 2
  // Random components: sample 0.6 in both directions, normalized to
  // 0.6/*sqrt(2*0.6^2) = a
  // Final velocities: [2.75 + 2*a, 2*a], and with -a for particle B

  REAL vel_com = 3.25;
  REAL expected_vel_random = 2 * 0.6 / std::sqrt(2 * 0.6 * 0.6);
  for (int i = 0; i < cell_count; i++) {

    auto weight = A->get_cell(Sym<REAL>("WEIGHT"), i);
    auto velocity = A->get_cell(Sym<REAL>("VELOCITY"), i);
    auto species_id = A->get_cell(Sym<INT>("INTERNAL_STATE"), i);

    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 1.0);
      if (species_id->at(rowx, 0) == 2) {
        EXPECT_DOUBLE_EQ(velocity->at(rowx, 0), vel_com + expected_vel_random);
        EXPECT_DOUBLE_EQ(velocity->at(rowx, 1), expected_vel_random);
      } else if (species_id->at(rowx, 0) == 3) {
        EXPECT_DOUBLE_EQ(velocity->at(rowx, 0), vel_com - expected_vel_random);
        EXPECT_DOUBLE_EQ(velocity->at(rowx, 1), -expected_vel_random);
      }
    }
  }
  auto child_subgroup = particle_sub_group(
      A, [](auto IS) { return IS[0] == 2; },
      Access::read(Sym<INT>("INTERNAL_STATE")));
  // We expect 6-7 pairs per cell
  // 50 * 50 (particles) * 4 * pi * 0.00005 (sigma_v) * 1.0 (q_hat) / 0.25
  // (volume)
  ASSERT_GE(child_subgroup->get_npart_local(), 6 * cell_count);
  ASSERT_LE(child_subgroup->get_npart_local(), 7 * cell_count);

  // Intentionally fail every rejection
  A = test_rng_lambda(0.1, 3.0);
  child_subgroup = particle_sub_group(
      A, [](auto IS) { return IS[0] == 2; },
      Access::read(Sym<INT>("INTERNAL_STATE")));
  ASSERT_EQ(child_subgroup->get_npart_local(), 0);

  // Exhaust all parents
  A = test_rng_lambda(100.0, 0.1);
  child_subgroup = particle_sub_group(
      A, [](auto IS) { return IS[0] == 2; },
      Access::read(Sym<INT>("INTERNAL_STATE")));
  ASSERT_EQ(child_subgroup->get_npart_local(), 50 * cell_count);

  auto parent_subgroup = particle_sub_group(
      A, [](auto IS) { return IS[0] == 0; },
      Access::read(Sym<INT>("INTERNAL_STATE")));
  ASSERT_EQ(parent_subgroup->get_npart_local(), 0);
}

TEST(SWPMReactionController, multi_reaction) {

  const int N_total = 800;

  auto test_rng_lambda = [=](auto dt, auto rng_val) {
    auto A = create_test_particle_group(N_total);
    auto B = create_test_particle_group(N_total);

    particle_loop(
        "set_vels_ids_a", A,
        [=](auto vel, auto sp_id, auto id) {
          vel[0] = 2;
          vel[1] = 0;
        },
        Access::write(Sym<REAL>("VELOCITY")),
        Access::write(Sym<INT>("INTERNAL_STATE")), Access::read(Sym<INT>("ID")))
        ->execute();

    particle_loop(
        "set_vels_ids_b", B,
        [=](auto vel, auto sp_id, auto id) {
          sp_id[0] = 1;
          vel[0] = 4;
          vel[1] = 0;
        },
        Access::write(Sym<REAL>("VELOCITY")),
        Access::write(Sym<INT>("INTERNAL_STATE")), Access::read(Sym<INT>("ID")))
        ->execute();

    A->add_particles_local(B);
    auto particle_subgroup = particle_sub_group(A);

    int cell_count = A->domain->mesh->get_cell_count();
    std::vector<int> subdivision_order(cell_count, 1);

    auto coll_cell_h = make_coll_cell_hierarchy<CartesianCollCellH>(
        A->sycl_target,
        std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
        subdivision_order);

    auto cc_manager = std::make_shared<CollisionCellManager>(
        A->sycl_target, coll_cell_h, std::vector<INT>{0, 1});

    auto lambda_remove = [](auto target) {
      target->get_particle_group()->remove_particles(target);
    };

    auto lambda_marker = [](auto w) { return w[0] < 1e-12; };
    auto accessor = Access::read(Sym<REAL>("WEIGHT"));
    auto test_removal_wrapper = std::make_shared<TransformationWrapper>(
        std::vector<std::shared_ptr<MarkingStrategy>>{
            make_direct_marking_strategy("small", lambda_marker, accessor)},
        make_lambda_transformation_strategy("remove", lambda_remove));

    std::mt19937 rng_state(52234234);
    std::uniform_real_distribution<> rng_dist(0.0, 1.0);
    auto rng_lambda_sampler = [&]() -> REAL { return rng_dist(rng_state); };

    auto rng_function =
        std::make_shared<HostRNGGenerationFunction<REAL>>(rng_lambda_sampler);
    // Mocking the species to test the correct reduced mass effects
    auto species_1 = Species("ION", 1.2, 0.0, 0);
    auto species_2 = Species("ION2", 2.0, 0.0, 1);

    auto rng_lambda = [&]() -> REAL { return rng_val; };

    auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 1);
    auto spec = std::make_shared<SWPMDSMCSpecification>(rng_kernel);
    auto controller = SWPMReactionController(
        species_1.get_id(), species_2.get_id(), spec, rng_function, cc_manager,
        std::vector<std::shared_ptr<TransformationWrapper>>{
            test_removal_wrapper},
        std::vector<std::shared_ptr<TransformationWrapper>>{
            test_removal_wrapper});

    auto cs_data =
        CSPairData<2, ConstantCrossSection>(ConstantCrossSection(0.0005, 1.0));

    auto rng_lambda_hs = [&]() -> REAL { return 0.6; };

    auto rng_kernel_hs = host_atomic_block_kernel_rng<REAL>(rng_lambda_hs, 2);
    auto hs_scattering_data =
        HSScatteringData<2>(species_1, species_2, rng_kernel_hs);
    auto pair_data_calculator = PairDataCalculator(hs_scattering_data);

    auto scattering_kernels = PairScatteringKernels<2>();

    auto swpm_reaction = std::make_shared<
        SWPMReaction<2, decltype(cs_data), decltype(scattering_kernels),
                     decltype(pair_data_calculator)>>(
        A->sycl_target,
        std::array<int, 2>{static_cast<int>(species_1.get_id()),
                           static_cast<int>(species_2.get_id())},
        std::array<int, 2>{2, 3}, cs_data, scattering_kernels,
        pair_data_calculator);

    auto swpm_reaction_2 = std::make_shared<
        SWPMReaction<2, decltype(cs_data), decltype(scattering_kernels),
                     decltype(pair_data_calculator)>>(
        A->sycl_target,
        std::array<int, 2>{static_cast<int>(species_1.get_id()),
                           static_cast<int>(species_2.get_id())},
        std::array<int, 2>{4, 5}, cs_data, scattering_kernels,
        pair_data_calculator);

    controller.add_reaction(swpm_reaction);
    controller.add_reaction(swpm_reaction_2);

    controller.apply(A, dt);

    return A;
  };

  auto A = test_rng_lambda(0.1, 0.1);
  int cell_count = A->domain->mesh->get_cell_count();
  for (int i = 0; i < cell_count; i++) {

    auto weight = A->get_cell(Sym<REAL>("WEIGHT"), i);
    auto velocity = A->get_cell(Sym<REAL>("VELOCITY"), i);
    auto species_id = A->get_cell(Sym<INT>("INTERNAL_STATE"), i);

    const int nrow = weight->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      if (species_id->at(rowx, 0) > 1) {
        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 0.5);
      } else {
        EXPECT_DOUBLE_EQ(weight->at(rowx, 0), 1.0);
      }
    }
  }
  auto child_subgroup = particle_sub_group(
      A, [](auto IS) { return IS[0] == 2; },
      Access::read(Sym<INT>("INTERNAL_STATE")));
  // We expect 12-13 pairs per cell
  // 50 * 50 (particles) * 4 * pi * 0.0001 (sigma_v sum) * 1.0 (q_hat) / 0.25
  // (volume)
  ASSERT_GE(child_subgroup->get_npart_local(), 12 * cell_count);
  ASSERT_LE(child_subgroup->get_npart_local(), 13 * cell_count);
}

TEST(SWPMReactionController, double_step) {

  const int N_total = 800;

  auto test_rng_lambda = [=](auto dt, auto rng_val) {
    auto A = create_test_particle_group(N_total);
    auto B = create_test_particle_group(N_total);

    particle_loop(
        "set_vels_ids_a", A,
        [=](auto vel, auto sp_id, auto id) {
          vel[0] = 2;
          vel[1] = 0;
        },
        Access::write(Sym<REAL>("VELOCITY")),
        Access::write(Sym<INT>("INTERNAL_STATE")), Access::read(Sym<INT>("ID")))
        ->execute();

    particle_loop(
        "set_vels_ids_b", B,
        [=](auto vel, auto sp_id, auto id) {
          sp_id[0] = 1;
          vel[0] = 4;
          vel[1] = 0;
        },
        Access::write(Sym<REAL>("VELOCITY")),
        Access::write(Sym<INT>("INTERNAL_STATE")), Access::read(Sym<INT>("ID")))
        ->execute();

    A->add_particles_local(B);
    auto particle_subgroup = particle_sub_group(A);

    int cell_count = A->domain->mesh->get_cell_count();
    std::vector<int> subdivision_order(cell_count, 1);

    auto coll_cell_h = make_coll_cell_hierarchy<CartesianCollCellH>(
        A->sycl_target,
        std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
        subdivision_order);

    auto cc_manager = std::make_shared<CollisionCellManager>(
        A->sycl_target, coll_cell_h, std::vector<INT>{0, 1});

    auto lambda_remove = [](auto target) {
      target->get_particle_group()->remove_particles(target);
    };

    auto lambda_marker = [](auto w) { return w[0] < 1e-12; };
    auto accessor = Access::read(Sym<REAL>("WEIGHT"));
    auto test_removal_wrapper = std::make_shared<TransformationWrapper>(
        std::vector<std::shared_ptr<MarkingStrategy>>{
            make_direct_marking_strategy("small", lambda_marker, accessor)},
        make_lambda_transformation_strategy("remove", lambda_remove));

    std::mt19937 rng_state(52234234);
    std::uniform_real_distribution<> rng_dist(0.0, 1.0);
    auto rng_lambda_sampler = [&]() -> REAL { return rng_dist(rng_state); };

    auto rng_function =
        std::make_shared<HostRNGGenerationFunction<REAL>>(rng_lambda_sampler);
    // Mocking the species to test the correct reduced mass effects
    auto species_1 = Species("ION", 1.2, 0.0, 0);
    auto species_2 = Species("ION2", 2.0, 0.0, 1);

    auto rng_lambda = [&]() -> REAL { return rng_val; };

    auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 1);
    auto spec = std::make_shared<SWPMDSMCSpecification>(rng_kernel);
    auto controller = SWPMReactionController(
        species_1.get_id(), species_2.get_id(), spec, rng_function, cc_manager,
        std::vector<std::shared_ptr<TransformationWrapper>>{
            test_removal_wrapper},
        std::vector<std::shared_ptr<TransformationWrapper>>{
            test_removal_wrapper});

    auto cs_data =
        CSPairData<2, ConstantCrossSection>(ConstantCrossSection(0.0005, 1.0));

    auto rng_lambda_hs = [&]() -> REAL { return 0.6; };

    auto rng_kernel_hs = host_atomic_block_kernel_rng<REAL>(rng_lambda_hs, 2);
    auto hs_scattering_data =
        HSScatteringData<2>(species_1, species_2, rng_kernel_hs);
    auto pair_data_calculator = PairDataCalculator(hs_scattering_data);

    auto scattering_kernels = PairScatteringKernels<2>();

    auto swpm_reaction = std::make_shared<
        SWPMReaction<2, decltype(cs_data), decltype(scattering_kernels),
                     decltype(pair_data_calculator)>>(
        A->sycl_target,
        std::array<int, 2>{static_cast<int>(species_1.get_id()),
                           static_cast<int>(species_2.get_id())},
        std::array<int, 2>{2, 3}, cs_data, scattering_kernels,
        pair_data_calculator);

    controller.add_reaction(swpm_reaction);

    controller.apply(A, dt);
    controller.apply(A, dt);

    return A;
  };

  auto A = test_rng_lambda(0.1, 0.1);
  int cell_count = A->domain->mesh->get_cell_count();
  auto child_subgroup = particle_sub_group(
      A, [](auto IS) { return IS[0] == 2; },
      Access::read(Sym<INT>("INTERNAL_STATE")));
  // We expect 6-7 pairs per cell in the first step
  // 50 * 50 (particles) * 4 * pi * 0.00005 (sigma_v) * 1.0 (q_hat) / 0.25
  // (volume)
  // We expect 9-10 pairs per cell in the second step
  // (44 or 43) * (44 or 43) (particles) * 4 * pi * 0.0001 (sigma_v now updated)
  // * 1.0 (q_hat) / 0.25 (volume)
  ASSERT_GE(child_subgroup->get_npart_local(), 15 * cell_count);
  ASSERT_LE(child_subgroup->get_npart_local(), 17 * cell_count);
}
