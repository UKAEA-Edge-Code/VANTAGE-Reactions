#include <array>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <neso_particles/typedefs.hpp>
#include <reactions/reactions.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

inline auto single_particle(const REAL &fluid_density_interp,
                            const REAL &fluid_temp_interp)
    -> std::shared_ptr<ParticleGroup> {
  auto dims = std::vector<int>(1, 1);

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  const int stencil_width = 1;

  const int pre_subdivision_cells =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  const int global_cell_count =
      pre_subdivision_cells * std::pow(std::pow(2, subdivision_order), 1);
  const int npart_per_cell = std::round((double)1 / (double)global_cell_count);

  auto mesh = std::make_shared<CartesianHMesh>(
      MPI_COMM_WORLD, 1, dims, cell_extent, subdivision_order, stencil_width);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  auto particle_spec_builder = ParticleSpecBuilder(1);

  auto properties_map = get_default_map();

  particle_spec_builder.add_particle_prop(Properties<REAL>(std::vector<int>{
      default_properties.fluid_density, default_properties.fluid_temperature}));

  auto particle_spec = particle_spec_builder.get_particle_spec();

  auto particle_group =
      std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  ParticleSet initial_distribution(1, particle_group->particle_spec);
  initial_distribution[Sym<REAL>("WEIGHT")][0][0] = 1.0;
  initial_distribution[Sym<REAL>("FLUID_DENSITY")][0][0] = fluid_density_interp;
  initial_distribution[Sym<REAL>("FLUID_TEMPERATURE")][0][0] =
      fluid_temp_interp;

  particle_group->add_particles_local(initial_distribution);
  return particle_group;
}

struct coefficient_values {
private:
  static const int ndens_dim = 8;
  static const int ntemp_dim = 10;

  constexpr static std::array<REAL, ndens_dim> dens_range = {
      1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18, 5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  constexpr static std::array<REAL, ntemp_dim> temp_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};

  std::array<std::array<REAL, ndens_dim>, ntemp_dim> coeffs;

public:
  coefficient_values() {
    REAL temp_i = 0.0;
    for (int itemp = 0; itemp < ntemp_dim; itemp++) {
      temp_i = this->temp_range[itemp];
      for (int idens = 0; idens < ndens_dim; idens++) {
        this->coeffs[itemp][idens] = temp_i * dens_range[idens];
      }
    }
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
};

TEST(ADASData, calc_data) {
  // Interpolation points
  REAL fluid_density_interp = 6.5e18;
  REAL fluid_temp_interp = 3.0e3;

  // Initialize a particle group with a single particle with the fluid density
  // and fluid temperature set to the interpolation values.
  auto particle_group =
      single_particle(fluid_density_interp, fluid_temp_interp);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values();
  auto temp_range = ADAS_values.get_temp_range();
  auto dens_range = ADAS_values.get_dens_range();
  auto coeffs = ADAS_values.get_coeffs();

  // Construct the ADASData object and extract the on-device object.
  auto test_adas_data = ADASData<temp_range.size(), dens_range.size()>(
      coeffs, temp_range, dens_range);
  auto test_adas_data_on_device = test_adas_data.get_on_device_obj();

  auto calculate_rates_int_syms = test_adas_data.get_required_int_sym_vector();
  auto calculate_rates_real_syms =
      test_adas_data.get_required_real_sym_vector();

  // For storing the calculation result.
  LocalArraySharedPtr<REAL> rate_buffer =
      std::make_shared<LocalArray<REAL>>(particle_group->sycl_target, 1, 0.0);

  auto rate_data_loop = particle_loop(
      "rate_data_loop", particle_group,
      [=](auto particle_index, auto req_int_props, auto req_real_props,
          auto kernel, auto buffer) {
        INT current_count = particle_index.get_loop_linear_index();

        std::array<REAL, 1> rate = test_adas_data_on_device.calc_data(
            particle_index, req_int_props, req_real_props, kernel);

        buffer[current_count] = rate[0];
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