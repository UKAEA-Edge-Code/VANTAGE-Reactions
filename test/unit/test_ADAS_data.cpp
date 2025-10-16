#include <reactions/reactions.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

inline auto single_particle() -> std::shared_ptr<ParticleGroup> {
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
  initial_distribution[Sym<REAL>("FLUID_DENSITY")][0][0] = 6.5e18;
  initial_distribution[Sym<REAL>("FLUID_TEMPERATURE")][0][0] = 3.0e3;

  particle_group->add_particles_local(initial_distribution);
  return particle_group;
}

TEST(ADASData, calc_data) {
  auto particle_group = single_particle();

  static const int ndens_dim = 8;
  static const int ntemp_dim = 10;

  std::array<REAL, ndens_dim> dens_range;
  std::array<REAL, ntemp_dim> temp_range;

  std::array<std::array<REAL, ndens_dim>, ntemp_dim> coeffs;

  temp_range = {1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
                5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
                3.59381366e+04, 1.00000000e+05};

  dens_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};

  REAL temp_i = 0.0;
  for (int itemp = 0; itemp < ntemp_dim; itemp++) {
    temp_i = temp_range[itemp];
    for (int idens = 0; idens < ndens_dim; idens++) {
      coeffs[itemp][idens] = temp_i * dens_range[idens];
    }
  }

  auto test_adas_data =
      ADASData<ntemp_dim, ndens_dim>(coeffs, temp_range, dens_range);
  auto test_adas_data_on_device = test_adas_data.get_on_device_obj();

  auto calculate_rates_int_syms = test_adas_data.get_required_int_sym_vector();
  auto calculate_rates_real_syms =
      test_adas_data.get_required_real_sym_vector();

  LocalArraySharedPtr<REAL> rate_buffer = std::make_shared<LocalArray<REAL>>(particle_group->sycl_target, 1, 0.0);

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

  printf("Interpolated rate: %e\n", interpolated_rate_data[0]);

  particle_group->domain->mesh->free();
}