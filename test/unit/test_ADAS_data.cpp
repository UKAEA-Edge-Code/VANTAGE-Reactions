#include "include/mock_particle_group.hpp"
#include <array>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <reactions/reactions.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

struct coefficient_values {
private:
  static const int ndens_dim = 8;
  static const int ntemp_dim = 10;

  static constexpr std::array<REAL, ndens_dim> dens_range = {
      1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18, 5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  static constexpr std::array<REAL, ntemp_dim> temp_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};

  std::array<std::array<REAL, ndens_dim>, ntemp_dim> coeffs;

  std::vector<REAL> coeffs_vec;
  std::vector<std::vector<REAL>> ranges;

public:
  coefficient_values() {
    REAL temp_i = 0.0;
    for (int itemp = 0; itemp < ntemp_dim; itemp++) {
      temp_i = this->temp_range[itemp];
      for (int idens = 0; idens < ndens_dim; idens++) {
        this->coeffs[itemp][idens] = temp_i * dens_range[idens];
        this->coeffs_vec.push_back(this->coeffs[itemp][idens]);
      }
    }

    std::vector<REAL> dens_range_vec(this->dens_range.begin(),
                                     this->dens_range.end());
    std::vector<REAL> temp_range_vec(this->temp_range.begin(),
                                     this->temp_range.end());

    this->ranges.push_back(temp_range_vec);
    this->ranges.push_back(dens_range_vec);
  };

  const std::array<REAL, ndens_dim> &get_dens_range() {
    return this->dens_range;
  }

  const std::array<REAL, ntemp_dim> &get_temp_range() {
    return this->temp_range;
  }

  const std::array<std::array<REAL, ndens_dim>, ntemp_dim> &get_coeffs() {
    return this->coeffs;
  }

  const std::vector<std::vector<REAL>> &get_ranges_vec() {
    return this->ranges;
  }

  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }
};

TEST(ADASData, calc_data) {
  // Interpolation points
  REAL fluid_density_interp = 6.5e18;
  REAL fluid_temp_interp = 3.0e3;

  // Initialize a particle group with a single particle with the fluid density
  // and fluid temperature set to the interpolation values.
  auto particle_group = create_test_particle_group(1e5);

  auto npart = particle_group->get_npart_local();

  particle_loop(
      particle_group,
      [=](auto fdens, auto ftemp) {
        fdens.at(0) = fluid_density_interp;
        ftemp.at(0) = fluid_temp_interp;
      },
      Access::write(Sym<REAL>("FLUID_DENSITY")),
      Access::write(Sym<REAL>("FLUID_TEMPERATURE")))
      ->execute();

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values();
  auto ranges_vec = ADAS_values.get_ranges_vec();
  auto coeffs_vec = ADAS_values.get_coeffs_vec();

  // Construct the ADASData object and extract the on-device object.
  auto test_adas_data =
      ADASData(coeffs_vec, ranges_vec, particle_group->sycl_target);
  auto test_adas_data_on_device = test_adas_data.get_on_device_obj();

  auto calculate_rates_int_syms = test_adas_data.get_required_int_sym_vector();
  auto calculate_rates_real_syms =
      test_adas_data.get_required_real_sym_vector();

  // For storing the calculation result.
  LocalArraySharedPtr<REAL> rate_buffer = std::make_shared<LocalArray<REAL>>(
      particle_group->sycl_target, npart, 0.0);

  auto rate_data_loop = particle_loop(
      "rate_data_loop", particle_group,
      [=](auto particle_index, auto req_int_props, auto req_real_props,
          auto kernel, auto buffer) {
        INT current_count = particle_index.get_loop_linear_index();

        std::array<REAL, 1> rate = test_adas_data_on_device.calc_data(
            particle_index, req_int_props, req_real_props, kernel);

        buffer[current_count] = rate[0];
        // printf("current_count = %d\n", int(current_count));
        // printf("rate = %f\n", rate[0]);
      },
      Access::read(ParticleLoopIndex{}),
      Access::write(sym_vector<INT>(particle_group, calculate_rates_int_syms)),
      Access::read(sym_vector<REAL>(particle_group, calculate_rates_real_syms)),
      Access::read(test_adas_data.get_rng_kernel()),
      Access::write(rate_buffer));

  rate_data_loop->execute();

  auto interpolated_rate_data = rate_buffer->get();

  EXPECT_DOUBLE_EQ(interpolated_rate_data[0], 1.95e22);

  particle_group->domain->mesh->free();
}