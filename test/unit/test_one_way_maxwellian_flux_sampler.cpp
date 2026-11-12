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
  const REAL drift_pi =
      (-3.0 * std::sqrt(2.0) + 5.0 * std::sqrt(3.0) + 7.0 * std::sqrt(5.0)) /
      std::sqrt(83.0);
  const REAL sampled_pi = drift_pi + 2.0 * norm_ratio / drift_pi;

  std::array<REAL, 3> expected_vals;

  for (int i = 0; i < 3; i++) {
    expected_vals[i] = sampled_e1 * basis_e1[i] + sampled_e2 * basis_e2[i] +
                       sampled_pi * basis_pi[i];
  }

  auto rng_lambda = [&]() -> REAL { return 0.75; };
  auto rng_kernel = NP::host_atomic_block_kernel_rng<REAL>(rng_lambda, 1000);

  OneWayMaxwellianFluxSampler sampler(norm_ratio, rng_kernel, PropertiesMap());
  auto sampler_on_device = sampler.get_on_device_obj();

  auto req_int_props_ = sampler.get_arg_pack().int_sym_vector();
  auto req_real_props_ = sampler.get_arg_pack().real_sym_vector();

  particle_loop(
      "one_way_maxwellian_flux_sample_loop", particle_group,
      [=](auto particle_index, auto req_int_props, auto req_real_props,
          auto sampled_velocity, auto kernel) {
        const auto accessors = SingleReactionDataAccessors{
            particle_index, req_int_props, req_real_props};
        auto sampled = sampler_on_device.calc_data(accessors, kernel);
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
  // DataCalculator must be able to invoke the sampler without an adapter.
  auto subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);
  auto buffer = std::make_shared<NP::NDLocalArray<REAL, 2>>(
      particle_group->sycl_target, particle_group->get_npart_local(), 3);
  DataCalculator<OneWayMaxwellianFluxSampler> calculator(sampler);
  calculator.fill_buffer(buffer, subgroup, 0, cell_count);
  const auto calculated = buffer->get();
  ASSERT_EQ(calculated.size(), 3 * particle_group->get_npart_local());
  for (size_t row = 0; row < calculated.size() / 3; row++) {
    for (size_t dim = 0; dim < 3; dim++) {
      EXPECT_NEAR(calculated[row * 3 + dim], expected_vals[dim], 1e-14);
    }
  }
  EXPECT_TRUE(rng_kernel->valid_internal_state());
  for (int i = 0; i < cell_count; i++) {
    auto sampled_cell =
        particle_group->get_cell(NP::Sym<REAL>("SAMPLED_VELOCITY"), i);

    auto panic_cell = particle_group->get_cell(
        NP::Sym<INT>(PropertiesMap().at(default_properties.panic)), i);
    for (int row = 0; row < sampled_cell->nrow; row++) {
      EXPECT_EQ(panic_cell->at(row, 0), 0);
      EXPECT_NEAR(sampled_cell->at(row, 0), expected_vals[0], 1e-14);
      EXPECT_NEAR(sampled_cell->at(row, 1), expected_vals[1], 1e-14);
      EXPECT_NEAR(sampled_cell->at(row, 2), expected_vals[2], 1e-14);
    }
  }

  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

namespace {

template <bool EnableOneWayMaxEdgeCase>
void check_normal_sampling(const REAL drift,
                           const std::vector<REAL> &random_values,
                           const REAL expected_normal,
                           const bool expect_panic = false) {
  auto particle_group = create_test_particle_group<3>(100);
  for (const auto &name : {"SURFACE_BASIS_E1", "SURFACE_BASIS_E2",
                           "SURFACE_BASIS_PI", "SAMPLED_VELOCITY"}) {
    particle_group->add_particle_dat(NP::Sym<REAL>(name), 3);
  }
  auto subgroup = NP::particle_sub_group(
      particle_group, [](auto id) { return id[0] == 0; },
      NP::Access::read(NP::Sym<INT>("ID")));
  ASSERT_EQ(subgroup->get_npart_local(), 1);

  NP::particle_loop(
      subgroup,
      [=](auto flow, auto temperature, auto e1, auto e2, auto pi) {
        temperature[0] = 1.0;
        for (int dim = 0; dim < 3; dim++) {
          flow[dim] = dim == 2 ? drift : 0.0;
          e1[dim] = dim == 0 ? 1.0 : 0.0;
          e2[dim] = dim == 1 ? 1.0 : 0.0;
          pi[dim] = dim == 2 ? 1.0 : 0.0;
        }
      },
      NP::Access::write(NP::Sym<REAL>("FLUID_FLOW_SPEED")),
      NP::Access::write(NP::Sym<REAL>("FLUID_TEMPERATURE")),
      NP::Access::write(NP::Sym<REAL>("SURFACE_BASIS_E1")),
      NP::Access::write(NP::Sym<REAL>("SURFACE_BASIS_E2")),
      NP::Access::write(NP::Sym<REAL>("SURFACE_BASIS_PI")))
      ->execute();

  std::size_t counter = 0;
  auto rng_kernel = NP::host_atomic_block_kernel_rng<REAL>(
      [&]() -> REAL {
        return random_values.at(counter++ % random_values.size());
      },
      16);
  rng_kernel->num_random_numbers_override = random_values.size();
  rng_kernel->suppress_warnings = true;
  OneWayMaxwellianFluxSampler sampler(1.0, rng_kernel);
  const auto on_device = sampler.get_on_device_obj();
  const auto req_int = sampler.get_arg_pack().int_sym_vector();
  const auto req_real = sampler.get_arg_pack().real_sym_vector();
  NP::particle_loop(
      subgroup,
      [=](auto index, auto int_props, auto real_props, auto output,
          auto kernel) {
        const auto accessors =
            SingleReactionDataAccessors{index, int_props, real_props};
        const auto sample =
            on_device.template calc_data<EnableOneWayMaxEdgeCase>(accessors,
                                                                  kernel);
        for (int dim = 0; dim < 3; dim++) {
          output[dim] = sample[dim];
        }
      },
      NP::Access::read(NP::ParticleLoopIndex{}),
      NP::Access::write(NP::sym_vector<INT>(particle_group, req_int)),
      NP::Access::read(NP::sym_vector<REAL>(particle_group, req_real)),
      NP::Access::write(NP::Sym<REAL>("SAMPLED_VELOCITY")),
      NP::Access::read(rng_kernel))
      ->execute();

  EXPECT_EQ(rng_kernel->valid_internal_state(), !expect_panic);
  for (int cell = 0; cell < particle_group->domain->mesh->get_cell_count();
       cell++) {
    const auto ids = particle_group->get_cell(NP::Sym<INT>("ID"), cell);
    const auto sampled =
        particle_group->get_cell(NP::Sym<REAL>("SAMPLED_VELOCITY"), cell);
    const auto panic =
        particle_group->get_cell(NP::Sym<INT>("REACTIONS_PANIC_FLAG"), cell);
    for (int row = 0; row < ids->nrow; row++) {
      if (ids->at(row, 0) == 0) {
        EXPECT_EQ(panic->at(row, 0), expect_panic ? 1 : 0);
        EXPECT_NEAR(sampled->at(row, 0), 0.0, 1e-14);
        EXPECT_NEAR(sampled->at(row, 1),
                    expect_panic ? 0.0 : -std::sqrt(-2.0 * std::log(0.75)),
                    1e-14);
        EXPECT_NEAR(sampled->at(row, 2), expected_normal, 1e-14);
        if (!expect_panic) {
          EXPECT_GT(sampled->at(row, 2), 0.0);
        }
      }
    }
  }
  particle_group->sycl_target->free();
  particle_group->domain->mesh->free();
}

} // namespace

TEST(OneWayMaxwellianFluxSampler, LargeDriftRejectsNonpositiveSamples) {
  const REAL expected =
      4.25 + std::sqrt(-2.0 * std::log(0.8)) * std::cos(2.0 * M_PI * 0.01);
  // The shifted proposal has mean 4.25; reject -1 and 0 before acceptance.
  check_normal_sampling<true>(4.0,
                              {0.75, 0.75, std::exp(-0.5 * 5.25 * 5.25), 0.5,
                               std::exp(-0.5 * 4.25 * 4.25), 0.5, 0.8, 0.01,
                               0.01},
                              expected);
}

TEST(OneWayMaxwellianFluxSampler, UsesOriginalSamplerWhenEdgeCaseDoesNotApply) {
  check_normal_sampling<false>(3.0, {0.75, 0.75, 0.8, 0.01}, 4.0);
  check_normal_sampling<true>(1.0, {0.75, 0.75, 0.8, 0.01}, 4.0);
}

TEST(OneWayMaxwellianFluxSampler, LargeDriftReportsExhaustedRNG) {
  // Exhaust the buffer during a retry; calc_data must set panic and return
  // zero.
  check_normal_sampling<true>(3.0, {0.75, 0.75, std::exp(-8.0), 0.5, 0.8}, 0.0,
                              true);
}
