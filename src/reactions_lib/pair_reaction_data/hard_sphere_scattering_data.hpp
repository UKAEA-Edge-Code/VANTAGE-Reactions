#ifndef REACTIONS_HARD_SPHERE_SCATTERING_DATA_H
#define REACTIONS_HARD_SPHERE_SCATTERING_DATA_H
#include "../pair_reaction_data.hpp"
#include "../particle_properties_map.hpp"
#include <iostream>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device:Reaction data class computing post-collision velocities from
 * hard-sphere collisions, i.e. with isotropic scattering in the COM frame
 *
 * Assumes that the RNG kernel provides 3 normally distributed random numbers
 *
 * @tparam vel_ndim The velocity space dimensionality
 */
template <size_t vel_ndim>
struct HSScatteringDataOnDevice
    : public PairReactionDataBaseOnDevice<2 * vel_ndim,
                                          HostAtomicBlockKernelRNG<REAL>> {

  HSScatteringDataOnDevice() = default;

  std::array<REAL, 2 * vel_ndim> calc_data(
      const PairReactionDataAccessors &accessor_pack,
      typename HostAtomicBlockKernelRNG<REAL>::KernelType &rng_kernel) const {

    REAL rel_speed = 0;
    std::array<REAL, vel_ndim> random_dir;
    std::array<REAL, vel_ndim - 1> random_nums;
    std::array<REAL, vel_ndim> com_vel;
    std::array<REAL, vel_ndim> vel_a;
    std::array<REAL, vel_ndim> vel_b;
    std::array<REAL, 2 * vel_ndim> output;
    REAL rel_vel;
    bool is_kernel_valid = true;
    for (int i = 0; i < vel_ndim; i++) {
      vel_a[i] = accessor_pack.req_real_props_a.at(this->velocity_ind_a, i);
      vel_b[i] = accessor_pack.req_real_props_b.at(this->velocity_ind_b, i);
    }

    constexpr REAL two_pi = 2 * M_PI;
    if constexpr (vel_ndim == 2) {

      random_nums[0] = rng_kernel.at(accessor_pack.index, 0, &is_kernel_valid);

      REAL valuecos;
      const REAL valuesin = Kernel::sincos(two_pi * random_nums[0], &valuecos);
      random_dir[0] = valuecos;
      random_dir[1] = valuesin;

    } else {
      random_nums[0] = rng_kernel.at(accessor_pack.index, 0, &is_kernel_valid);
      random_nums[1] = rng_kernel.at(accessor_pack.index, 1, &is_kernel_valid);

      REAL valuecos;
      const REAL valuesin = Kernel::sincos(two_pi * random_nums[0], &valuecos);
      REAL valuecos_theta;
      const REAL valuesin_theta =
          Kernel::sincos(M_PI * random_nums[1], &valuecos_theta);
      random_dir[0] = valuecos * valuesin_theta;
      random_dir[1] = valuesin * valuesin_theta;
      random_dir[2] = valuecos_theta;
    }
    for (int i = 0; i < vel_ndim; i++) {
      rel_vel = vel_a[i] - vel_b[i];
      rel_speed += rel_vel * rel_vel;

      com_vel[i] = vel_a[i] * this->mass_a + vel_b[i] * this->mass_b;
    }
    rel_speed = Kernel::sqrt(rel_speed);
    if (!is_kernel_valid) {
      accessor_pack.req_int_props_a.at(this->panic_ind_a, 0) += 1;
      accessor_pack.req_int_props_b.at(this->panic_ind_b, 0) += 1;
    }

    for (int i = 0; i < vel_ndim; i++) {
      output[i] = com_vel[i] * this->tot_mass_inv +
                  this->mass_b * this->tot_mass_inv * rel_speed * random_dir[i];
      output[vel_ndim + i] =
          com_vel[i] * this->tot_mass_inv -
          this->mass_a * this->tot_mass_inv * rel_speed * random_dir[i];
    }
    return output;
  }

public:
  int velocity_ind_a, velocity_ind_b, panic_ind_a, panic_ind_b;
  REAL mass_a, mass_b, tot_mass_inv;
};

/**
 * @brief Reaction class computing post-collision velocities for a hard-sphere
 * DSMC collision
 *
 * @tparam vel_ndim The velocity space dimensionality
 */
template <size_t vel_ndim>
struct HSScatteringData
    : public PairReactionDataBase<HSScatteringDataOnDevice<vel_ndim>,
                                  2 * vel_ndim,
                                  HostAtomicBlockKernelRNG<REAL>> {

  constexpr static auto props = default_properties;

  constexpr static auto required_simple_real_props =
      std::array<int, 1>{props.velocity};
  constexpr static auto required_simple_int_props =
      std::array<int, 1>{props.panic};

  /**
   * @brief Constructor for HSScatteringData.
   *
   * @param species_a Species of the first scattering particle (needed for
   * species mass)
   * @param species_b Species of the first scattering particle (needed for
   * species mass)
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  HSScatteringData(
      const Species &species_a, const Species &species_b,
      std::shared_ptr<HostAtomicBlockKernelRNG<REAL>> rng_kernel,
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase<HSScatteringDataOnDevice<vel_ndim>, 2 * vel_ndim,
                             HostAtomicBlockKernelRNG<REAL>>(
            Properties<INT>(required_simple_int_props),
            Properties<REAL>(required_simple_real_props), properties_map) {

    static_assert(vel_ndim == 2 || vel_ndim == 3,
                  "Only 2 or 3 dimensional velocity allowed for hard sphere "
                  "scattering data.");
    this->on_device_obj = HSScatteringDataOnDevice<vel_ndim>();

    this->on_device_obj->mass_a = species_a.get_mass();
    this->on_device_obj->mass_b = species_b.get_mass();
    this->on_device_obj->tot_mass_inv =
        1 / (species_a.get_mass() + species_b.get_mass());

    this->set_rng_kernel(rng_kernel);
    this->index_on_device_object();
  }

  /**
   * @brief Index the particle velocity and panic index
   */
  void index_on_device_object() {

    auto arg_pack = this->get_arg_pack();
    this->on_device_obj->velocity_ind_a =
        arg_pack.required_real_props_a.find_index(
            this->properties_map.at(props.velocity));

    this->on_device_obj->velocity_ind_b =
        arg_pack.required_real_props_b.find_index(
            this->properties_map.at(props.velocity));

    this->on_device_obj->panic_ind_a = arg_pack.required_int_props_a.find_index(
        this->properties_map.at(props.panic));

    this->on_device_obj->panic_ind_b = arg_pack.required_int_props_b.find_index(
        this->properties_map.at(props.panic));
  };
};
}; // namespace VANTAGE::Reactions
#endif
