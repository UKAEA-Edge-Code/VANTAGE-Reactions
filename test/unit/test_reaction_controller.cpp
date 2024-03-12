#include "common_markers.hpp"
#include "common_transformations.hpp"
#include "containers/cell_dat_const.hpp"
#include "merge_transformation.hpp"
#include "particle_sub_group.hpp"
#include "transformation_wrapper.hpp"
#include "typedefs.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <reaction_base.hpp>
#include <reaction_controller.hpp>
#include <reaction_data.hpp>

using namespace NESO::Particles;
using namespace Reactions;

template <INT num_products_per_parent>
struct TestReaction
    : public LinearReactionBase<TestReaction<num_products_per_parent>,
                                num_products_per_parent> {

  friend struct LinearReactionBase<TestReaction, num_products_per_parent>;

  TestReaction() = default;

  TestReaction(SYCLTargetSharedPtr sycl_target_, Sym<REAL> total_reaction_rate_,
               REAL rate_, std::vector<int> in_states_, std::vector<int> out_states_)
      : LinearReactionBase<TestReaction<num_products_per_parent>,
                           num_products_per_parent>(
            sycl_target_, total_reaction_rate_,
            std::vector<Sym<REAL>>{Sym<REAL>("V"),
                                   Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<REAL>>{Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<INT>>{Sym<INT>("INTERNAL_STATE")},
            std::vector<Sym<INT>>(), in_states_, out_states_),
        test_reaction_data(TestReactionData(rate_)) {
        }

public:
  void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::SymVector::Write<INT> &write_req_ints,
                       Access::SymVector::Write<REAL> &write_req_reals,
                       Access::LocalArray::Read<REAL> &pre_req_data,
                       double dt) const {
    auto k_W = write_req_reals.at(0, index, 0);
    write_req_reals.at(0, index, 0) += (k_W * modified_weight);
  }

private:
  struct TestReactionData : public ReactionDataBase<TestReactionData> {
    TestReactionData() = default;

    TestReactionData(REAL rate_) : rate(rate_){};

    const REAL &get_n_to_SI() const { return n_to_SI; }

    REAL calc_rate(Access::LoopIndex::Read &index,
                   Access::SymVector::Read<REAL> &vars) const {

      return this->rate;
    }

  private:
    REAL rate;
    const REAL n_to_SI = 3.0e18;
  };

  TestReactionData test_reaction_data;

protected:
  const TestReactionData &get_reaction_data() const {
    return test_reaction_data;
  }
};

struct IoniseReaction : public LinearReactionBase<IoniseReaction, 0> {

  friend struct LinearReactionBase<IoniseReaction, 0>;

  IoniseReaction() = default;

  IoniseReaction(SYCLTargetSharedPtr sycl_target_,
                 Sym<REAL> total_reaction_rate_, std::vector<int> in_states_)
      : LinearReactionBase<IoniseReaction, 0>(
            sycl_target_, total_reaction_rate_,
            std::vector<Sym<REAL>>{
                Sym<REAL>("V"), Sym<REAL>("ELECTRON_TEMPERATURE"),
                Sym<REAL>("ELECTRON_DENSITY"), Sym<REAL>("SOURCE_ENERGY"),
                Sym<REAL>("SOURCE_MOMENTUM"), Sym<REAL>("SOURCE_DENSITY"),
                Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<REAL>>{
                Sym<REAL>("V"), Sym<REAL>("ELECTRON_TEMPERATURE"),
                Sym<REAL>("ELECTRON_DENSITY"), Sym<REAL>("SOURCE_ENERGY"),
                Sym<REAL>("SOURCE_MOMENTUM"), Sym<REAL>("SOURCE_DENSITY"),
                Sym<REAL>("COMPUTATIONAL_WEIGHT")},
            std::vector<Sym<INT>>(), std::vector<Sym<INT>>(), in_states_, std::vector<int>{}),
        test_reaction_data(IoniseReactionData()) {}

public:
  void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::SymVector::Write<INT> &write_req_ints,
                       Access::SymVector::Write<REAL> &write_req_reals,
                       Access::LocalArray::Read<REAL> &pre_req_data,
                       double dt) const {
    auto k_V_0 = write_req_reals.at(0, index, 0);
    auto k_V_1 = write_req_reals.at(0, index, 1);
    const REAL vsquared = (k_V_0 * k_V_0) + (k_V_1 * k_V_1);

    REAL k_n_scale = 1.0; // / test_reaction_data.get_n_to_SI();
    REAL inv_k_dt = 1.0 / dt;

    auto nE = write_req_reals.at(2, index, 0);

    // Set SOURCE_DENSITY
    write_req_reals.at(5, index, 0) =
        -nE * modified_weight * k_n_scale * inv_k_dt;

    // Get SOURCE_DENSITY for SOURCE_MOMENTUM calc
    auto k_SD = write_req_reals.at(5, index, 0);
    write_req_reals.at(4, index, 0) = k_SD * k_V_0;
    write_req_reals.at(4, index, 1) = k_SD * k_V_1;

    // Set SOURCE_ENERGY
    write_req_reals.at(3, index, 0) = k_SD * vsquared * 0.5;

    // Get then set COMPUTATIONAL_WEIGHT
    auto k_W = write_req_reals.at(6, index, 0);
    write_req_reals.at(6, index, 0) -= (k_W * modified_weight);
  }

private:
  struct IoniseReactionData : public ReactionDataBase<IoniseReactionData> {
    IoniseReactionData() = default;

    IoniseReactionData(REAL &dt_) : ReactionDataBase<IoniseReactionData>(dt_) {}

    const REAL &get_n_to_SI() const { return n_to_SI; }

    REAL calc_rate(Access::LoopIndex::Read &index,
                   Access::SymVector::Read<REAL> &vars) const {

      return 1.0;
    }

  private:
    const REAL n_to_SI = 3.0e18;
  };

  mutable IoniseReactionData test_reaction_data;

protected:
  IoniseReactionData &get_reaction_data() const { return test_reaction_data; }
};

inline auto create_test_particle_group(int N_total)
    -> shared_ptr<ParticleGroup> {

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

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<REAL>("TOT_REACTION_RATE"), 1),
                             ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
                             ParticleProp(Sym<INT>("INTERNAL_STATE"), 1),
                             ParticleProp(Sym<REAL>("ELECTRON_TEMPERATURE"), 1),
                             ParticleProp(Sym<REAL>("ELECTRON_DENSITY"), 1),
                             ParticleProp(Sym<REAL>("SOURCE_ENERGY"), 1),
                             ParticleProp(Sym<REAL>("SOURCE_MOMENTUM"), ndim),
                             ParticleProp(Sym<REAL>("SOURCE_DENSITY"), 1)};
  auto particle_group =
      std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

  const int cell_count = domain->mesh->get_cell_count();
  const int N = npart_per_cell * cell_count;

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells,
                                 rng_pos);

  auto velocities =
      NESO::Particles::normal_distribution(N, ndim, 0.0, 0.5, rng_vel);
  // std::uniform_int_distribution<int> uniform_dist(
  //     0, size - 1);
  ParticleSet initial_distribution(N, particle_group->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("V")][px][dimx] =
          velocities.at(dimx).at(px);
      initial_distribution[Sym<REAL>("SOURCE_MOMENTUM")][px][dimx] = 0.0;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    initial_distribution[Sym<REAL>("TOT_REACTION_RATE")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("COMPUTATIONAL_WEIGHT")][px][0] = 1.0;
    initial_distribution[Sym<INT>("INTERNAL_STATE")][px][0] = 0;
    initial_distribution[Sym<REAL>("ELECTRON_TEMPERATURE")][px][0] = 2.0;
    initial_distribution[Sym<REAL>("ELECTRON_DENSITY")][px][0] = 3.0e18;
    initial_distribution[Sym<REAL>("SOURCE_ENERGY")][px][0] = 0.0;
    initial_distribution[Sym<REAL>("SOURCE_DENSITY")][px][0] = 0.0;
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

TEST(ReactionController, single_reaction_multi_apply) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>(
          Sym<REAL>("P"), Sym<REAL>("COMPUTATIONAL_WEIGHT"), Sym<REAL>("V"));

  auto reaction_controller =
      ReactionController(child_transform, Sym<INT>("INTERNAL_STATE"));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 1;

  auto test_reaction = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate,
      std::vector<int>{0}, std::vector<int>{1});

  test_reaction.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  reaction_controller.add_reaction(std::make_shared<TestReaction<num_products_per_parent>>(test_reaction));

  reaction_controller.apply_reactions(particle_group, 0.01);

  auto original_group_marking = make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(Sym<INT>("INTERNAL_STATE"), 0);
  auto original_group = original_group_marking->make_marker_subgroup(
    std::make_shared<ParticleSubGroup>(particle_group)
  );

  auto tot_weight = make_shared<CellDatConst<REAL>>(particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
    original_group,
    [=](auto W, auto TotW) {
      TotW.fetch_add(0, 0, W[0]);
    },
    Access::read(Sym<REAL>("COMPUTATIONAL_WEIGHT")),
    Access::add(tot_weight)
  )->execute(0);

  auto initial_weight_per_cell = tot_weight->get_cell(0)->at(0, 0);

  auto merged_group_marking = make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(Sym<INT>("INTERNAL_STATE"), 1);
  auto merged_group = merged_group_marking->make_marker_subgroup(
    std::make_shared<ParticleSubGroup>(particle_group)
  );

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 2);

    std::vector<INT> cells = {icell, icell};
    std::vector<INT> layers = {0, 1};

    auto merged_particles = merged_group->get_particles(cells, layers);
    for (int i = 0; i < 2; i ++) {
      EXPECT_NEAR(merged_particles->at(Sym<REAL>("COMPUTATIONAL_WEIGHT"), i, 0), initial_weight_per_cell / 2, 1e-12);
    }
  }

  reaction_controller.apply_reactions(particle_group, 0.01);

  merged_group = merged_group_marking->make_marker_subgroup(
    std::make_shared<ParticleSubGroup>(particle_group)
  );

  // TODO: Find a way to check the weights of the descendant
  // particles against the weight of the parent group after
  // one time-step and after two time-steps.
  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 4);    
  }

  particle_group->domain->mesh->free(); // Explicit free? Yuck
}

TEST(ReactionController, multi_reaction_multi_apply) {
  const int N_total = 1600;

  auto particle_group = create_test_particle_group(N_total);

  auto particle_group_2 = create_test_particle_group(N_total);
  auto loop2 = particle_loop(
      "set_internal_state2", particle_group_2,
      [=](auto internal_state) { internal_state[0] = 2; },
      Access::write(Sym<INT>("INTERNAL_STATE")));

  loop2->execute();

  particle_group->add_particles_local(particle_group_2);

  auto cell_count = particle_group->domain->mesh->get_cell_count();

  auto child_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>(
          Sym<REAL>("P"), Sym<REAL>("COMPUTATIONAL_WEIGHT"), Sym<REAL>("V"));

  auto reaction_controller =
      ReactionController(child_transform, Sym<INT>("INTERNAL_STATE"));

  REAL test_rate = 5.0; // example rate

  const INT num_products_per_parent = 1;

  auto test_reaction1 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate,
      std::vector<int>{0}, std::vector<int>{1});

  test_rate = 5.0; // example rate

  auto test_reaction2 = TestReaction<num_products_per_parent>(
      particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate,
      std::vector<int>{2}, std::vector<int>{3});

  test_reaction1.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  test_reaction2.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  reaction_controller.add_reaction(std::make_shared<TestReaction<num_products_per_parent>>(test_reaction1));

  reaction_controller.add_reaction(std::make_shared<TestReaction<num_products_per_parent>>(test_reaction2));

  reaction_controller.apply_reactions(particle_group, 0.1);

  auto original_group_marking = make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(Sym<INT>("INTERNAL_STATE"), 0);
  auto original_group = original_group_marking->make_marker_subgroup(
    std::make_shared<ParticleSubGroup>(particle_group)
  );

  auto original_group_marking2 = make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(Sym<INT>("INTERNAL_STATE"), 2);
  auto original_group2 = original_group_marking->make_marker_subgroup(
    std::make_shared<ParticleSubGroup>(particle_group)
  );

  auto tot_weight = make_shared<CellDatConst<REAL>>(particle_group->sycl_target, cell_count, 1, 1);
  auto tot_weight2 = make_shared<CellDatConst<REAL>>(particle_group->sycl_target, cell_count, 1, 1);

  particle_loop(
    original_group,
    [=](auto W, auto TotW) {
      TotW.fetch_add(0, 0, W[0]);
    },
    Access::read(Sym<REAL>("COMPUTATIONAL_WEIGHT")),
    Access::add(tot_weight)
  )->execute(0);

  particle_loop(
    original_group2,
    [=](auto W, auto TotW) {
      TotW.fetch_add(0, 0, W[0]);
    },
    Access::read(Sym<REAL>("COMPUTATIONAL_WEIGHT")),
    Access::add(tot_weight2)
  )->execute(0);

  auto initial_weight_per_cell = tot_weight->get_cell(0)->at(0, 0);
  auto initial_weight_per_cell2 = tot_weight2->get_cell(0)->at(0, 0);

  auto merged_group_marking = make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(Sym<INT>("INTERNAL_STATE"), 1);
  auto merged_group = merged_group_marking->make_marker_subgroup(
    std::make_shared<ParticleSubGroup>(particle_group)
  );

  auto merged_group_marking2 = make_marking_strategy<ComparisonMarkerSingle<EqualsComp<INT>, INT>>(Sym<INT>("INTERNAL_STATE"), 3);
  auto merged_group2 = merged_group_marking2->make_marker_subgroup(
    std::make_shared<ParticleSubGroup>(particle_group)
  );

  for (int icell = 0; icell < cell_count; icell++) {
    EXPECT_EQ(merged_group->get_npart_cell(icell), 2);
    EXPECT_EQ(merged_group2->get_npart_cell(icell), 2);

    std::vector<INT> cells = {icell, icell};
    std::vector<INT> layers = {0, 1};

    auto merged_particles = merged_group->get_particles(cells, layers);
    auto merged_particles2 = merged_group2->get_particles(cells, layers);
    for (int i = 0; i < 2; i ++) {
      EXPECT_NEAR(merged_particles->at(Sym<REAL>("COMPUTATIONAL_WEIGHT"), i, 0), initial_weight_per_cell / 2, 1e-12);
      EXPECT_NEAR(merged_particles2->at(Sym<REAL>("COMPUTATIONAL_WEIGHT"), i, 0), initial_weight_per_cell2 / 2, 1e-12);
    }
  }

  particle_group->domain->mesh->free();
  particle_group_2->domain->mesh->free();
}

TEST(ReactionController, ionisation_reaction) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  auto reaction_controller = ReactionController(Sym<INT>("INTERNAL_STATE"));

  auto ionise_reaction =
      IoniseReaction(particle_group->sycl_target,
                     Sym<REAL>("TOT_REACTION_RATE"), std::vector<int>{0});

  ionise_reaction.flush_buffer(
      static_cast<size_t>(particle_group->get_npart_local()));

  reaction_controller.add_reaction(std::make_shared<IoniseReaction>(ionise_reaction));

  reaction_controller.apply_reactions(particle_group, 1.5);

  auto test_removal_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{make_marking_strategy<
          ComparisonMarkerSingle<EqualsComp<REAL>, REAL>>(
          Sym<REAL>("COMPUTATIONAL_WEIGHT"), 0.0)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto W = particle_group->get_cell(Sym<REAL>("COMPUTATIONAL_WEIGHT"), cellx);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(W->at(rowx, 0), 0.0);
    };
  };

  test_removal_wrapper.transform(particle_group);

  auto final_particle_num = particle_group->get_npart_local();

  EXPECT_EQ(final_particle_num, 0);

  particle_group->domain->mesh->free();
}
