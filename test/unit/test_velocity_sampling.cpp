#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(VelocitySampling, FilteredMaxwellianFailure) {
  struct AlwaysFailCrossSection : public AbstractCrossSection {
    AlwaysFailCrossSection() = default;

    REAL get_value_at(const REAL &relative_vel) const { return 0.0; }

    REAL get_max_rate_val() const { return 0.0; }

    bool accept_reject(REAL relative_vel = 0.0, REAL uniform_rand = 0.0,
                       REAL value_at = 0.0, REAL max_rate_val = 0.0) const {
      return false;
    }
  };

  auto cs = AlwaysFailCrossSection();

  std::mt19937 rng = std::mt19937(52234126);
  std::uniform_real_distribution<REAL> uniform_dist(0.0, 1.0);
  auto rng_lambda = [&]() -> REAL {
    REAL rng_sample;
    do {
      rng_sample = uniform_dist(rng);
    } while (rng_sample == 0.0);
    return rng_sample;
  };

  auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 4);

  const int ndim = 2;
  auto sampler =
      FilteredMaxwellianSampler<ndim, decltype(cs)>(1.0, cs, rng_kernel);

  auto sampler_on_device = sampler.get_on_device_obj();

  auto particle_group = create_test_particle_group(1000);

  auto req_int_props_ = sampler.get_required_int_sym_vector();

  auto req_real_props_ = sampler.get_required_real_sym_vector();

  particle_loop(
      "vel_sampling_fail_loop", particle_group,
      [=](auto particle_index, auto req_int_props, auto req_real_props,
          auto kernel) {
        auto test = sampler_on_device.calc_data(particle_index, req_int_props,
                                                req_real_props, kernel);
      },
      Access::read(ParticleLoopIndex{}),
      Access::write(sym_vector<INT>(particle_group, req_int_props_)),
      Access::read(sym_vector<REAL>(particle_group, req_real_props_)),
      Access::read(sampler.get_rng_kernel()))
      ->execute();

  EXPECT_TRUE(panicked(particle_sub_group(particle_group)));

  particle_group->domain->mesh->free();
}
