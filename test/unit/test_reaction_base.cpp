#include "compute_target.hpp"
#include "containers/local_array.hpp"
#include "containers/sym_vector.hpp"
#include "loop/particle_loop_index.hpp"
#include "packing_unpacking.hpp"
#include "particle_group.hpp"
#include "particle_sub_group.hpp"
#include "reaction_base.hpp"
#include "reaction_data.hpp"
#include "typedefs.hpp"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <iostream>

using namespace NESO::Particles;

template <INT in_state_id>
struct TestReaction: public LinearReactionBase<TestReaction<in_state_id>, in_state_id> {

    friend struct LinearReactionBase<TestReaction, in_state_id>;

    TestReaction() = default;

    TestReaction(
        SYCLTargetSharedPtr sycl_target_,
        Sym<REAL> total_reaction_rate_,
        REAL rate_
    ) :
        LinearReactionBase<TestReaction<in_state_id>, in_state_id>(
            sycl_target_,
            total_reaction_rate_,
            std::vector<Sym<REAL>> {
                Sym<REAL>("V"),
                Sym<REAL>("COMPUTATIONAL_WEIGHT")
            },
            std::vector<Sym<REAL>>(),
            std::vector<Sym<INT>> {
                Sym<INT>("INTERNAL_STATE")
            },
            std::vector<Sym<INT>>()
        ),
        test_reaction_data(TestReactionData(rate_))
    {}

    private:
        struct TestReactionData: public ReactionDataBase<TestReactionData> {
            TestReactionData() = default;

            TestReactionData(
                REAL rate_
            ) : rate(rate_) {};

            REAL calc_rate(Access::LoopIndex::Read& index,Access::SymVector::Read<REAL>& vars) const {

                return this->rate;
            }

            private:
                REAL rate;
        };

        TestReactionData test_reaction_data;

    protected:
        const TestReactionData& get_reaction_data() const {
            return test_reaction_data;
        }

};

template <INT in_state_id>
struct TestReactionVarRate: public LinearReactionBase<TestReactionVarRate<in_state_id>, in_state_id> {

    friend struct LinearReactionBase<TestReactionVarRate, in_state_id>;

    TestReactionVarRate() = default;

    TestReactionVarRate(
        SYCLTargetSharedPtr sycl_target_,
        Sym<REAL> total_reaction_rate_,
        Sym<REAL> read_var
    ) :
        LinearReactionBase<TestReactionVarRate<in_state_id>, in_state_id>(
            sycl_target_,
            total_reaction_rate_,
            std::vector<Sym<REAL>>{read_var},
            std::vector<Sym<REAL>>(),
            std::vector<Sym<INT>>(),
            std::vector<Sym<INT>>()
        )
    {
        this->set_reaction_data(TestReactionVarData());
    }

    private:
        struct TestReactionVarData: public ReactionDataBase<TestReactionVarData> {
            TestReactionVarData() = default;

            REAL calc_rate(Access::LoopIndex::Read& index,Access::SymVector::Read<REAL>& vars) const {

                return vars.at(0,index,0);
            }    
        };        

        mutable TestReactionVarData test_reaction_data;
    
    protected:
        TestReactionVarData& get_reaction_data() const {
            return test_reaction_data;
        }

        void set_reaction_data(const TestReactionVarData& reaction_data_) const {
            test_reaction_data = reaction_data_;
        }

};

// TODO: Generalise and clean this up
auto create_test_particle_group(int N_total) -> shared_ptr<ParticleGroup> {

    const int ndim = 2;
    std::vector<int> dims(ndim);
    dims[0] = 2;
    dims[1] = 2;

    const double cell_extent = 1.0;
    const int subdivision_order = 1;
    const int stencil_width = 1;
    
    const int global_cell_count = dims[0] * dims[1] * std::pow(std::pow(2, subdivision_order), ndim);
    const int npart_per_cell = std::round((double) N_total / (double) global_cell_count);

    auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                        subdivision_order, stencil_width);

    auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

    auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

    auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<REAL>("V"), ndim),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<INT>("ID"), 1),
                               ParticleProp(Sym<REAL>("TOT_REACTION_RATE"), 1),
                               ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
                               ParticleProp(Sym<INT>("INTERNAL_STATE"), 1)};    
    auto particle_group = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

    const int rank = sycl_target->comm_pair.rank_parent;
    const int size = sycl_target->comm_pair.size_parent;

    std::mt19937 rng_pos(52234234 + rank);
    std::mt19937 rng_vel(52234231 + rank);
    std::mt19937 rng_rank(18241);

    const int cell_count = domain->mesh->get_cell_count();
    const int N = npart_per_cell * cell_count;

    std::vector<std::vector<double>> positions;
    std::vector<int> cells;
    uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells, rng_pos);

    auto velocities = NESO::Particles::normal_distribution(
        N, ndim, 0.0, 0.5, rng_vel);
    // std::uniform_int_distribution<int> uniform_dist(
    //     0, size - 1);
    ParticleSet initial_distribution(N, particle_group->get_particle_spec());
    for (int px = 0; px < N; px++) {
        for (int dimx = 0; dimx < ndim; dimx++) {
            initial_distribution[Sym<REAL>("P")][px][dimx] = positions.at(dimx).at(px);
            initial_distribution[Sym<REAL>("V")][px][dimx] = velocities.at(dimx).at(px);
        }
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
        initial_distribution[Sym<INT>("ID")][px][0] = px;
        initial_distribution[Sym<REAL>("TOT_REACTION_RATE")][px][0] = 0.0;
        initial_distribution[Sym<REAL>("COMPUTATIONAL_WEIGHT")][px][0] = 1.0;
        initial_distribution[Sym<INT>("INTERNAL_STATE")][px][0] = 0;
    }
    particle_group->add_particles_local(initial_distribution);

    auto pbc = std::make_shared<CartesianPeriodic>(sycl_target, mesh, particle_group->position_dat);
    auto ccb = std::make_shared<CartesianCellBin>(sycl_target, mesh, particle_group->position_dat, particle_group->cell_id_dat);

    pbc->execute();
    particle_group->hybrid_move();
    ccb->execute();
    particle_group->cell_move();

    MPI_Barrier(sycl_target->comm_pair.comm_parent);

    return particle_group;

}

TEST(LinearReactionBase, calc_rate) {
    const int N_total = 1000;

    auto particle_group = create_test_particle_group(N_total);
    auto particle_sub_group = std::make_shared<ParticleSubGroup>(
        particle_group,
        [=](auto ISTATE) {
            return (ISTATE[0] == 0);
        },
        Access::read(Sym<INT>("INTERNAL_STATE"))
    );

    REAL test_rate = 5.0;  //example rate

    auto test_reaction = TestReaction<0>(
        particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), test_rate
    );

    test_reaction.flush_buffer(static_cast<size_t>(particle_group->get_npart_local()));

    int cell_count = particle_group->domain->mesh->get_cell_count();

    const INT num_products_per_parent = 1;

    for (int i=0; i < cell_count;i++){

        test_reaction.run_rate_loop(particle_sub_group, i);
        test_reaction.descendant_product_loop(particle_sub_group, i, num_products_per_parent);
        test_reaction.run_rate_loop(particle_sub_group, i);
        test_reaction.descendant_product_loop(particle_sub_group, i, num_products_per_parent);

        auto descendant_particle_sub_group = std::make_shared<ParticleSubGroup>(
            particle_group,
            [=](auto ISTATE) {
                return (ISTATE[0] == 1);
            },
            Access::read(Sym<INT>("INTERNAL_STATE"))
        );

        auto parent_particles = std::make_shared<ParticleGroup>(
            particle_group->domain, particle_group->get_particle_spec(), particle_group->sycl_target
        );

        auto descendant_particles = std::make_shared<ParticleGroup>(
            particle_group->domain, particle_group->get_particle_spec(), particle_group->sycl_target
        );

        parent_particles->add_particles_local(particle_sub_group);
        descendant_particles->add_particles_local(descendant_particle_sub_group);

        auto position = parent_particles->get_cell(Sym<REAL>("P"), i);
        auto tot_reaction_rate = parent_particles->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
        
        // auto velocity = descendant_particles->get_cell(Sym<REAL>("V"), i);
        // auto weight = descendant_particles->get_cell(Sym<REAL>("COMPUTATIONAL_WEIGHT"), i);

        const int nrow = position->nrow;

        for (int rowx = 0; rowx < nrow; rowx++) {
            EXPECT_EQ(tot_reaction_rate->at(rowx, 0), 2*test_rate) << "calc_rate did not set TOT_REACTION_RATE correctly...";
        }

    }

    particle_group->domain->mesh->free(); // Explicit free? Yuck
}

TEST(LinearReactionBase, calc_var_rate) {
    const int N_total = 1000;

    auto particle_group = create_test_particle_group(N_total);
    auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

    auto test_reaction = TestReactionVarRate<0>(
        particle_group->sycl_target, Sym<REAL>("TOT_REACTION_RATE"), Sym<REAL>("P")
    );

    test_reaction.flush_buffer(static_cast<size_t>(particle_group->get_npart_local()));

    int cell_count = particle_group->domain->mesh->get_cell_count();

    for (int i=0; i < cell_count;i++){

        test_reaction.run_rate_loop(particle_sub_group, i);
        test_reaction.run_rate_loop(particle_sub_group, i);

        auto position = particle_group->get_cell(Sym<REAL>("P"), i);
        auto tot_reaction_rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
        const int nrow = position->nrow;

        for (int rowx = 0; rowx < nrow; rowx++) {
            EXPECT_EQ(tot_reaction_rate->at(rowx, 0), 2*position->at(rowx, 0)) << "calc_rate dP not set TOT_REACTION_RATE correctly...";
        }
    }

    particle_group->domain->mesh->free(); // Explicit free? Yuck

}