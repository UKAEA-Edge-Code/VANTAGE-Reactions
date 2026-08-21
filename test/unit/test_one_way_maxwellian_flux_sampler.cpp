#include "include/mock_particle_group.hpp"
#include "include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(OneWayMaxwellianFluxSampler, SamplesExpectedVelocityWithDeterministicRNG) {

  const int N_total = 100;
  auto particle_group = create_test_particle_group<3>(N_total);

  particle_group->add_particle_dat(NP::Sym<REAL>("SURFACE_BASIS_E1"), 3);
  particle_group->add_particle_dat(NP::Sym<REAL>("SURFACE_BASIS_E2"), 3);
  particle_group->add_particle_dat(NP::Sym<REAL>("SURFACE_BASIS_PI"), 3);
  particle_group->add_particle_dat(NP::Sym<REAL>("SAMPLED_VELOCITY"), 3);

  std::array<REAL, 3> flow_speed{};
  std::array<REAL, 3> basis_e1{};
  std::array<REAL, 3> basis_e2{};
  std::array<REAL, 3> basis_pi{};
  flow_speed[0] = -std::sqrt(2.0);
  flow_speed[1] = std::sqrt(3.0);
  flow_speed[2] = std::sqrt(5.0);
  basis_e1[0] = 2.0 / std::sqrt(38.0);
  basis_e1[1] = 3.0 / std::sqrt(38.0);
  basis_e1[2] = 5.0 / std::sqrt(38.0);
  basis_e2[0] = 5.0 / std::sqrt(195.0);
  basis_e2[1] = 7.0 / std::sqrt(195.0);
  basis_e2[2] = 11.0 / std::sqrt(195.0);
  basis_pi[0] = 3.0 / std::sqrt(83.0);
  basis_pi[1] = 5.0 / std::sqrt(83.0);
  basis_pi[2] = 7.0 / std::sqrt(83.0);

  particle_loop(
      "one_way_maxwellian_flux_init_loop", particle_group,
      [=](auto particle_index, auto flow_speed_dat, auto basis_e1_dat,
          auto basis_e2_dat, auto basis_pi_dat) {
        for (int i = 0; i < 3; ++i) {
          flow_speed_dat[i] = flow_speed[i];
          basis_e1_dat[i] = basis_e1[i];
          basis_e2_dat[i] = basis_e2[i];
          basis_pi_dat[i] = basis_pi[i];
        }
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::Sym<REAL>("FLUID_FLOW_SPEED")),
      NP::Access::write(NP::Sym<REAL>("SURFACE_BASIS_E1")),
      NP::Access::write(NP::Sym<REAL>("SURFACE_BASIS_E2")),
      NP::Access::write(NP::Sym<REAL>("SURFACE_BASIS_PI")))
      ->execute();

  const REAL norm_ratio = 1.0;
  const REAL sampled_e1 =
      (-2.0 * std::sqrt(2.0) + 3.0 * std::sqrt(3.0) + 5.0 * std::sqrt(5.0)) /
      std::sqrt(38.0);
  const REAL sampled_e2 =
      (-5.0 * std::sqrt(2.0) + 7.0 * std::sqrt(3.0) + 11.0 * std::sqrt(5.0)) /
          std::sqrt(195.0) -
      std::sqrt(2.0 * norm_ratio) * std::sqrt(-2 * std::log(0.75));
  const REAL sampled_pi = 3.0 * std::sqrt(2.0 * norm_ratio);

  std::array<REAL, 3> expected_vals;

  for (int i = 0; i < 3; i++) {
    expected_vals[i] = sampled_e1 * basis_e1[i] + sampled_e2 * basis_e2[i] +
                       sampled_pi * basis_pi[i];
  }

  auto rng_lambda = [&]() -> REAL { return 0.75; };
  auto rng_kernel = NP::host_atomic_block_kernel_rng<REAL>(rng_lambda, 1000);

  OneWayMaxwellianFluxSampler sampler(norm_ratio, rng_kernel,
                                      get_default_map());
  auto sampler_on_device = sampler.get_on_device_obj();

  auto req_int_props_ = sampler.get_required_int_sym_vector();
  auto req_real_props_ = sampler.get_required_real_sym_vector();

  particle_loop(
      "one_way_maxwellian_flux_sample_loop", particle_group,
      [=](auto particle_index, auto req_int_props, auto req_real_props,
          auto sampled_velocity, auto kernel) {
        auto sampled = sampler_on_device.calc_data(
            particle_index, req_int_props, req_real_props, kernel);
        sampled_velocity[0] = sampled[0];
        sampled_velocity[1] = sampled[1];
        sampled_velocity[2] = sampled[2];
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::sym_vector<INT>(particle_group, req_int_props_)),
      NP::Access::read(NP::sym_vector<REAL>(particle_group, req_real_props_)),
      NP::Access::write(NP::Sym<REAL>("SAMPLED_VELOCITY")),
      NP::Access::read(sampler.get_rng_kernel()))
      ->execute();

  const int cell_count = particle_group->domain->mesh->get_cell_count();
  for (int i = 0; i < cell_count; i++) {
    auto sampled_cell =
        particle_group->get_cell(NP::Sym<REAL>("SAMPLED_VELOCITY"), i);

    for (int row = 0; row < sampled_cell->nrow; row++) {
      EXPECT_NEAR(sampled_cell->at(row, 0), expected_vals[0], 1e-14);
      EXPECT_NEAR(sampled_cell->at(row, 1), expected_vals[1], 1e-14);
      EXPECT_NEAR(sampled_cell->at(row, 2), expected_vals[2], 1e-14);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}
