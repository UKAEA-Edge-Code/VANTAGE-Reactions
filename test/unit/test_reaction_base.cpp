#include "containers/local_array.hpp"
#include "containers/sym_vector.hpp"
#include "loop/particle_loop_index.hpp"
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

struct TestReactionData: public ReactionDataBase<TestReactionData> {
    
    TestReactionData() = default;

    TestReactionData(
        REAL rate_
    ) : rate(rate_) {};

    REAL calc_rate() const {

        return this->rate;
    }

    private:
        REAL rate;
};

struct TestReactionVarData: public ReactionDataBase<TestReactionVarData> {
    TestReactionVarData() = default;

    REAL calc_rate(Access::LoopIndex::Read& index,Access::SymVector::Read<REAL>& vars) const {

        return vars.at(0,index,0);
    }    
};

template <typename ReactionDataType, int in_state_id>
struct TestReaction: public LinearReactionBase<TestReaction<ReactionDataType, in_state_id>, ReactionDataType, in_state_id> {

    TestReaction() = default;

    TestReaction(
        Sym<REAL> total_reaction_rate_,
        ReactionDataType reaction_data_
    ) :
        LinearReactionBase<TestReaction<ReactionDataType, in_state_id>, ReactionDataType, in_state_id>(
            total_reaction_rate_,
            std::vector<Sym<REAL>>(),
            std::vector<Sym<REAL>>(),
            std::vector<Sym<INT>>(),
            std::vector<Sym<INT>>(),
            reaction_data_
        )
    {}

    void run_rate_loop(ParticleGroupSharedPtr particle_group, INT cell_idx) {
        auto reaction_data_buffer = this->get_reaction_data();

        auto device_rate_buffer = std::make_shared<LocalArray<REAL>>(
            particle_group->sycl_target,
            this->get_rate_buffer().size(),
            0
        );

        auto total_reaction_rate_buffer = this->get_total_reaction_rate();

        auto req_dats_real = sym_vector<REAL>(
            particle_group, this->get_read_req_dats_real()
        );

        auto loop = particle_loop(
            "calc_rate_loop",
            particle_group,
            [=](auto particle_index, auto req_reals, auto tot_rate, auto buffer){
                INT current_count = particle_index.get_loop_linear_index();
                REAL rate = reaction_data_buffer.calc_rate();
                buffer[current_count] = rate;
                tot_rate[0] += rate;
            },
            Access::read(ParticleLoopIndex{}),
            Access::read(req_dats_real),
            Access::write(total_reaction_rate_buffer),
            Access::write(device_rate_buffer)
        );

        loop->execute(cell_idx);

        this->set_rate_buffer(device_rate_buffer->get());

        this->set_reaction_data(reaction_data_buffer);

        return;
    }
};

template <typename ReactionDataType, int in_state_id>
struct TestReactionVarRate: public LinearReactionBase<TestReactionVarRate<ReactionDataType, in_state_id>, ReactionDataType, in_state_id> {

    TestReactionVarRate() = default;

    TestReactionVarRate(
        Sym<REAL> total_reaction_rate_,
        Sym<REAL> read_var,
        ReactionDataType reaction_data_
    ) :
        LinearReactionBase<TestReactionVarRate<ReactionDataType, in_state_id>, ReactionDataType, in_state_id>(
            total_reaction_rate_,
            std::vector<Sym<REAL>>{read_var},
            std::vector<Sym<REAL>>(),
            std::vector<Sym<INT>>(),
            std::vector<Sym<INT>>(),
            reaction_data_
        )
    {}

    void run_rate_loop(ParticleGroupSharedPtr particle_group, INT cell_idx) {
        auto reaction_data_buffer = this->get_reaction_data();

        auto device_rate_buffer = std::make_shared<LocalArray<REAL>>(
            particle_group->sycl_target,
            this->get_rate_buffer().size(),
            0
        );

        auto total_reaction_rate_buffer = this->get_total_reaction_rate();

        auto req_dats_real = sym_vector<REAL>(
            particle_group, this->get_read_req_dats_real()
        );

        auto loop = particle_loop(
            "calc_rate_loop",
            particle_group,
            [=](auto particle_index, auto req_reals, auto tot_rate, auto buffer){
                INT current_count = particle_index.get_loop_linear_index();
                REAL rate = reaction_data_buffer.calc_rate(particle_index, req_reals);
                buffer[current_count] = rate;
                tot_rate[0] += rate;
            },
            Access::read(ParticleLoopIndex{}),
            Access::read(req_dats_real),
            Access::write(total_reaction_rate_buffer),
            Access::write(device_rate_buffer)
        );

        loop->execute(cell_idx);

        this->set_rate_buffer(device_rate_buffer->get());

        this->set_reaction_data(reaction_data_buffer);

        return;
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

    auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

    auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

    auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<REAL>("V"), ndim),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<INT>("ID"), 1),
                               ParticleProp(Sym<REAL>("TOT_REACTION_RATE"), 1)};

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

    REAL test_rate = 5.0;  //example rate

    auto test_reaction_data = TestReactionData(test_rate);

    auto test_reaction = TestReaction<TestReactionData, 0>(Sym<REAL>("TOT_REACTION_RATE"), test_reaction_data);

    test_reaction.flush_buffer(static_cast<size_t>(particle_group->get_npart_local()));

    int cell_count = particle_group->domain->mesh->get_cell_count();

    for (int i=0; i < cell_count;i++){

        test_reaction.run_rate_loop(particle_group, i);
        test_reaction.run_rate_loop(particle_group, i);

        auto loop = particle_loop(
                "Verify calc_rate execution",
                particle_group,
                [=](auto T){
                    EXPECT_EQ(T.at(0), 2*test_rate) << "calc_rate did not set TOT_REACTION_RATE correctly...";
                },
                Access::read(Sym<REAL>("TOT_REACTION_RATE"))
        );

        loop->execute(i);

    }

    particle_group->domain->mesh->free(); // Explicit free? Yuck

}

TEST(LinearReactionBase, calc_var_rate) {
    const int N_total = 1000;

    auto particle_group = create_test_particle_group(N_total);

    auto test_reaction_data = TestReactionVarData();

    auto test_reaction = TestReactionVarRate<TestReactionVarData, 0>(
        Sym<REAL>("TOT_REACTION_RATE"), Sym<REAL>("P"), test_reaction_data
    );

    test_reaction.flush_buffer(static_cast<size_t>(particle_group->get_npart_local()));

    int cell_count = particle_group->domain->mesh->get_cell_count();

    for (int i=0; i < cell_count;i++){

        test_reaction.run_rate_loop(particle_group, i);
        test_reaction.run_rate_loop(particle_group, i);

        auto loop = particle_loop(
                "Verify calc_rate execution",
                particle_group,
                [=](auto T,auto P){
                    EXPECT_EQ(T.at(0), 2*P.at(0)) << "calc_rate dP not set TOT_REACTION_RATE correctly...";
                },
                Access::read(Sym<REAL>("TOT_REACTION_RATE")),
                Access::read(Sym<REAL>("P"))
        );

        loop->execute(i);
    }

    particle_group->domain->mesh->free(); // Explicit free? Yuck

}