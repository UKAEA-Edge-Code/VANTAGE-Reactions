#ifndef REACTIONS_SPECULAR_REFLECTION_DATA_H
#define REACTIONS_SPECULAR_REFLECTION_DATA_H
#include "../particle_properties_map.hpp"
#include "../utils.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: ReactionData calculating specularly reflected velocity
 * given ingoing velocity and surface normal
 *
 * @tparam ndim The velocity space dimensionality
 */
template <size_t ndim>
struct SpecularReflectionDataOnDevice
    : public ReactionDataBaseOnDevice<ndim, DEFAULT_RNG_KERNEL, ndim> {

  SpecularReflectionDataOnDevice() = default;

  /**
   * @brief Function to calculate the specularly reflected velocities
   *
   * @param accessors Bundled accessors for the ParticleLoop.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A REAL-valued array of size ndim that contains the calculated
   * reflected velocities.
   */
  std::array<REAL, ndim>
  calc_data(const std::array<REAL, ndim> input,
            const SingleReactionDataAccessors &accessors,
            typename ReactionDataBaseOnDevice<ndim>::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    std::array<REAL, ndim> surface_n;

    // Calculate 2 * v_in dot n
    for (int vdim = 0; vdim < ndim; vdim++) {
      surface_n[vdim] =
          accessors.req_real_props.at(normal_ind, accessors.index, vdim);
    }

    return utils::reflect_vector(input, surface_n);
  }

public:
  int normal_ind;
};

/**
 * @brief ReactionData calculating specularly reflected velocity
 * given ingoing velocity and surface normal
 *
 * @tparam ndim The velocity space dimensionality
 */
template <size_t ndim>
struct SpecularReflectionData
    : public ReactionDataBase<SpecularReflectionDataOnDevice<ndim>, ndim,
                              DEFAULT_RNG_KERNEL, ndim> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.boundary_intersection_normal};
  /**
   * @brief Constructor for SpecularReflectionData.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  SpecularReflectionData(
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<SpecularReflectionDataOnDevice<ndim>, ndim,
                         DEFAULT_RNG_KERNEL, ndim>(
            Properties<REAL>(required_simple_real_props), properties_map) {

    this->on_device_obj = SpecularReflectionDataOnDevice<ndim>();
    this->index_on_device_object();
  }

  /**
   * @brief Index the surface normal properties on the
   * on-device object
   */
  void index_on_device_object() {

    this->on_device_obj->normal_ind =
        this->argument_pack.required_real_props.find_index(
            this->properties_map.at(props.boundary_intersection_normal));
  };
};
}; // namespace VANTAGE::Reactions
#endif
