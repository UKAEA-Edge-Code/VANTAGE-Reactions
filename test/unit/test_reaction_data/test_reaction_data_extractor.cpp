#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include "../include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(ReactionData, ExtractorData_DefaultOffset) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group<2>(N_total);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto weight_extract = ExtractorData<1>(NP::Sym<REAL>("WEIGHT"));
  auto pos_extract = ExtractorData<2>(NP::Sym<REAL>("POSITION"));

  auto data_calc_weight = DataCalculator(weight_extract);
  auto data_calc_pos = DataCalculator(pos_extract);

  auto pre_req_weight = std::make_shared<NP::NDLocalArray<REAL, 2>>(
      particle_group->sycl_target, 0, data_calc_weight.get_data_size());
  auto pre_req_pos = std::make_shared<NP::NDLocalArray<REAL, 2>>(
      particle_group->sycl_target, 0, data_calc_pos.get_data_size());

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    pre_req_weight = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 1);
    pre_req_pos = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 2);

    data_calc_weight.fill_buffer(pre_req_weight, particle_sub_group, i, i + 1);
    data_calc_pos.fill_buffer(pre_req_pos, particle_sub_group, i, i + 1);

    auto weight_buf = pre_req_weight->get();
    auto pos_buf = pre_req_pos->get();

    auto pos_cell = particle_group->get_cell(NP::Sym<REAL>("POSITION"), i);
    auto weight_cell = particle_group->get_cell(NP::Sym<REAL>("WEIGHT"), i);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      EXPECT_DOUBLE_EQ(weight_buf[ipart], weight_cell->at(ipart, 0));
      EXPECT_DOUBLE_EQ(pos_buf[ipart * 2 + 0], pos_cell->at(ipart, 0));
      EXPECT_DOUBLE_EQ(pos_buf[ipart * 2 + 1], pos_cell->at(ipart, 1));
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ReactionData, ExtractorData_Specific1DComponent) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group<3>(N_total);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  // FLUID_FLOW_SPEED has values: comp 0 -> 1.0, comp 1 -> 3.0, comp 2 -> 5.0
  auto comp0_extract = ExtractorData<1>(NP::Sym<REAL>("FLUID_FLOW_SPEED"), 0);
  auto comp1_extract = ExtractorData<1>(NP::Sym<REAL>("FLUID_FLOW_SPEED"), 1);
  auto comp2_extract = ExtractorData<1>(NP::Sym<REAL>("FLUID_FLOW_SPEED"), 2);

  // Using extract helper with default n_comp = 1
  auto comp0_quick = extract("FLUID_FLOW_SPEED", 0);
  auto comp1_quick = extract("FLUID_FLOW_SPEED", 1);
  auto comp2_quick = extract("FLUID_FLOW_SPEED", 2);

  auto data_calc0 = DataCalculator(comp0_extract);
  auto data_calc1 = DataCalculator(comp1_extract);
  auto data_calc2 = DataCalculator(comp2_extract);

  auto data_calc0_q = DataCalculator(comp0_quick);
  auto data_calc1_q = DataCalculator(comp1_quick);
  auto data_calc2_q = DataCalculator(comp2_quick);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto n_part_cell = particle_sub_group->get_npart_cell(i);

    auto buf0 = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 1);
    auto buf1 = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 1);
    auto buf2 = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 1);
    auto buf0_q = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 1);
    auto buf1_q = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 1);
    auto buf2_q = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 1);

    data_calc0.fill_buffer(buf0, particle_sub_group, i, i + 1);
    data_calc1.fill_buffer(buf1, particle_sub_group, i, i + 1);
    data_calc2.fill_buffer(buf2, particle_sub_group, i, i + 1);
    data_calc0_q.fill_buffer(buf0_q, particle_sub_group, i, i + 1);
    data_calc1_q.fill_buffer(buf1_q, particle_sub_group, i, i + 1);
    data_calc2_q.fill_buffer(buf2_q, particle_sub_group, i, i + 1);

    auto d0 = buf0->get();
    auto d1 = buf1->get();
    auto d2 = buf2->get();
    auto d0_q = buf0_q->get();
    auto d1_q = buf1_q->get();
    auto d2_q = buf2_q->get();

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      EXPECT_DOUBLE_EQ(d0[ipart], 1.0);
      EXPECT_DOUBLE_EQ(d1[ipart], 3.0);
      EXPECT_DOUBLE_EQ(d2[ipart], 5.0);

      EXPECT_DOUBLE_EQ(d0_q[ipart], 1.0);
      EXPECT_DOUBLE_EQ(d1_q[ipart], 3.0);
      EXPECT_DOUBLE_EQ(d2_q[ipart], 5.0);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ReactionData, ExtractorData_MultiComponentOffset) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group<3>(N_total);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  // Extract 2 components starting at offset 0 (Vx, Vy) and offset 1 (Vy, Vz)
  auto vel_xy = ExtractorData<2>(NP::Sym<REAL>("VELOCITY"), 0);
  auto vel_yz = ExtractorData<2>(NP::Sym<REAL>("VELOCITY"), 1);

  auto flow_yz = extract<2>("FLUID_FLOW_SPEED", 1);

  auto data_calc_xy = DataCalculator(vel_xy);
  auto data_calc_yz = DataCalculator(vel_yz);
  auto data_calc_flow_yz = DataCalculator(flow_yz);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto n_part_cell = particle_sub_group->get_npart_cell(i);

    auto buf_xy = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 2);
    auto buf_yz = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 2);
    auto buf_flow = std::make_shared<NP::NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, n_part_cell, 2);

    data_calc_xy.fill_buffer(buf_xy, particle_sub_group, i, i + 1);
    data_calc_yz.fill_buffer(buf_yz, particle_sub_group, i, i + 1);
    data_calc_flow_yz.fill_buffer(buf_flow, particle_sub_group, i, i + 1);

    auto d_xy = buf_xy->get();
    auto d_yz = buf_yz->get();
    auto d_flow = buf_flow->get();

    auto vel_cell = particle_group->get_cell(NP::Sym<REAL>("VELOCITY"), i);

    for (int ipart = 0; ipart < n_part_cell; ipart++) {
      EXPECT_DOUBLE_EQ(d_xy[ipart * 2 + 0], vel_cell->at(ipart, 0));
      EXPECT_DOUBLE_EQ(d_xy[ipart * 2 + 1], vel_cell->at(ipart, 1));

      EXPECT_DOUBLE_EQ(d_yz[ipart * 2 + 0], vel_cell->at(ipart, 1));
      EXPECT_DOUBLE_EQ(d_yz[ipart * 2 + 1], vel_cell->at(ipart, 2));

      EXPECT_DOUBLE_EQ(d_flow[ipart * 2 + 0], 3.0);
      EXPECT_DOUBLE_EQ(d_flow[ipart * 2 + 1], 5.0);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

TEST(ReactionData, ExtractorData_ReactionApplication) {
  const int N_total = 100;

  auto particle_group = create_test_particle_group<2>(N_total);
  auto particle_sub_group =
      std::make_shared<NP::ParticleSubGroup>(particle_group);

  // Extract y-component of POSITION (offset 1, 1 comp) and 1D weight (offset 0)
  auto pos_y = extract("POSITION", 1);
  auto weight = extract<1>("WEIGHT", 0);
  auto concat = ConcatenatorData(pos_y, weight);

  auto test_reaction =
      LinearReactionBase<0, TestReactionData, TestReactionDataCalcKernels<0>,
                         DataCalculator<decltype(concat)>>(
          particle_group->sycl_target, 0, std::array<int, 0>{},
          TestReactionData(2.0), TestReactionDataCalcKernels<0>(),
          DataCalculator(concat));

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<NP::ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.calculate_rates(particle_sub_group, i, i + 1);
    test_reaction.apply(particle_sub_group, i, i + 1, 0.1,
                        descendant_particles);

    auto position = particle_group->get_cell(NP::Sym<REAL>("POSITION"), i);
    auto source_density =
        particle_group->get_cell(NP::Sym<REAL>("ELECTRON_SOURCE_DENSITY"), i);
    auto source_energy =
        particle_group->get_cell(NP::Sym<REAL>("ELECTRON_SOURCE_ENERGY"), i);

    const int nrow = position->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      // source_density received concat[0] = position_y
      EXPECT_DOUBLE_EQ(source_density->at(rowx, 0), position->at(rowx, 1));
      // source_energy received concat[1] = weight = 1.0
      EXPECT_DOUBLE_EQ(source_energy->at(rowx, 0), 1.0);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
