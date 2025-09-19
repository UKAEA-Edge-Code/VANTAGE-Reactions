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
struct SpecularReflectionDataOnDevice : public ReactionDataBaseOnDevice<ndim> {

  /**
   * @brief Function to calculate the specularly reflected velocities
   *
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
   * @return A REAL-valued array of size ndim that contains the calculated
   * reflected velocities.
   */
  std::array<REAL, ndim>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice<ndim>::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    std::array<REAL, ndim> k_V;
    std::array<REAL, ndim> surface_n;
    REAL proj_factor = 0.0;

    // Calculate 2 * v_in dot n
    for (int vdim = 0; vdim < ndim; vdim++) {
      k_V[vdim] = req_real_props.at(velocity_ind, index, vdim);
      surface_n[vdim] = req_real_props.at(normal_ind, index, vdim);
    }

    return utils::reflect_vector(k_V, surface_n);
  }

public:
  int velocity_ind, normal_ind;
};

/**
 * @brief ReactionData calculating specularly reflected velocity
 * given ingoing velocity and surface normal
 *
 * @tparam ndim The velocity space dimensionality
 */
template <size_t ndim>
struct SpecularReflectionData : public ReactionDataBase<ndim> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.velocity, props.boundary_intersection_normal};
  /**
   * @brief Constructor for SpecularReflectionData.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  SpecularReflectionData(
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<ndim>(Properties<REAL>(required_simple_real_props),
                               properties_map) {

    this->device_obj.velocity_ind = this->required_real_props.simple_prop_index(
        props.velocity, this->properties_map);

    this->device_obj.normal_ind = this->required_real_props.simple_prop_index(
        props.boundary_intersection_normal);
  }

private:
  SpecularReflectionDataOnDevice<ndim> device_obj;

public:
  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */

  SpecularReflectionDataOnDevice<ndim> get_on_device_obj() {
    return this->device_obj;
  }
};
}; // namespace VANTAGE::Reactions
#endif
