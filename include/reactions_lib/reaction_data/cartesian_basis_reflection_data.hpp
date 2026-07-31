#ifndef REACTIONS_CARTESIAN_BASIS_REFLECTION_DATA_H
#define REACTIONS_CARTESIAN_BASIS_REFLECTION_DATA_H
#include "../particle_properties_map.hpp"
#include "../utils.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: ReactionData calculating a reflected velocity vector in a
 * cartesian basis determined by the ingoing velocity and the normal.
 *
 * The expected inputs are a size 3 array with entries for the velocities.
 * The first two components are the components parallel to the surface (with the
 * first component in the direction determined by the projection of the ingoing
 * particle velocity). The final component is in the direction of the surface
 * normal (directed back into the domain).
 *
 * Works only for 3D
 */
struct CartesianBasisReflectionDataOnDevice
    : public ReactionDataBaseOnDevice<3, DEFAULT_RNG_KERNEL, 3> {

  CartesianBasisReflectionDataOnDevice() = default;

  /**
   * @brief Function to calculate the reflected velocities
   *
   * @param input The v_x, v_y, v_z components of the reflected vector (see
   * class docstring)
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A REAL-valued array of size e that contains the calculated
   * reflected velocities.
   */
  std::array<REAL, 3>
  calc_data(const std::array<REAL, 3> input,
            const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename DEFAULT_RNG_KERNEL::KernelType &kernel) const {

    std::array<REAL, 3> surface_n;
    std::array<REAL, 3> vel;

    for (int vdim = 0; vdim < 3; vdim++) {
      surface_n[vdim] = req_real_props.at(normal_ind, index, vdim);
      vel[vdim] = req_real_props.at(vel_ind, index, vdim);
    }

    std::array<REAL, 9> basis = utils::get_normal_basis(vel, surface_n);
    std::array<REAL, 3> result;

    for (auto i = 0; i < 3; i++) {

      result[i] = input[0] * basis[i] + input[1] * basis[i + 3] +
                  input[2] * basis[i + 6];
    }
    return result;
  }

public:
  int normal_ind, vel_ind;
};

/**
 * @brief ReactionData calculating reflected velocity
 * from components in the cartesian coordinate system derived from the surface
 * normal and the velocity vector of the particle. The vectors of the local
 * basis are:
 *
 * x - in the direction along the projection of the velocity onto the surface
 * y - in the plane of the surface, perpendicular to x
 * z - along the surface normal pointing into the domain
 *
 * The input array is expected to be a size 3 array with entries for velocities
 * in the three directions.
 *
 */
struct CartesianBasisReflectionData
    : public ReactionDataBase<CartesianBasisReflectionDataOnDevice, 3,
                              DEFAULT_RNG_KERNEL, 3> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.velocity, props.boundary_intersection_normal};
  /**
   * @brief Constructor for CartesianBasisReflectionData.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  CartesianBasisReflectionData(
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<CartesianBasisReflectionDataOnDevice, 3,
                         DEFAULT_RNG_KERNEL, 3>(
            Properties<REAL>(required_simple_real_props), properties_map) {

    this->on_device_obj = CartesianBasisReflectionDataOnDevice();
    this->index_on_device_object();
  }

  /**
   * @brief Index the particle velocity and surface normal properties on the
   * on-device object
   */
  void index_on_device_object() {

    this->on_device_obj->normal_ind = this->required_real_props.find_index(
        this->properties_map.at(props.boundary_intersection_normal));

    this->on_device_obj->vel_ind = this->required_real_props.find_index(
        this->properties_map.at(props.velocity));
  };
};
}; // namespace VANTAGE::Reactions
#endif
