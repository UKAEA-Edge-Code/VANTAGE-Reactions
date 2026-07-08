#ifndef REACTIONS_CS_PAIR_REACTION_DATA_H
#define REACTIONS_CS_PAIR_REACTION_DATA_H
#include "../cross_sections/constant_rate_cs.hpp"
#include "../pair_reaction_data.hpp"
#include "../particle_properties_map.hpp"
#include <iostream>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device:Reaction class computing the sigma * v_r value for each
 * particle pair
 *
 * @tparam vel_ndim The velocity space dimensionality
 * @tparam CROSS_SECTION The typename corresponding to the cross-section class
 * used
 */
template <size_t vel_ndim, typename CROSS_SECTION>
struct CSPairDataOnDevice : public PairReactionDataBaseOnDevice<> {

  CSPairDataOnDevice() = default;
  /**
   * @brief Constructor for CSPairDataOnDevice.
   *
   * @param cross_section Cross section object to be used in the rejection
   * method sampling
   */
  CSPairDataOnDevice(CROSS_SECTION cross_section)
      : cross_section(cross_section) {};

  std::array<REAL, 1>
  calc_data(const PairReactionDataAccessors &accessor_pack,
            typename PairReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &rng_kernel) const {

    REAL rel_speed = 0;
    REAL rel_vel;
    for (int i = 0; i < vel_ndim; i++) {
      rel_vel = accessor_pack.req_real_props_a.at(this->velocity_ind_a, i) -
                accessor_pack.req_real_props_b.at(this->velocity_ind_b, i);
      rel_speed += rel_vel * rel_vel;
    }
    rel_speed = Kernel::sqrt(rel_speed);

    return std::array<REAL, 1>{this->cross_section.get_value_at(rel_speed) *
                               rel_speed};
  }

  /**
   * @brief Return the maximum rate value from the contained cross section up to
   * some maximum relative velocity
   *
   * @param max_rel_vel
   * @return REAL-valued maximum rate
   */
  REAL get_cs_max_rate_val(REAL max_rel_vel) {
    return this->cross_section.get_max_rate_val(max_rel_vel);
  }

public:
  int velocity_ind_a, velocity_ind_b;
  CROSS_SECTION cross_section;
};

/**
 * @brief Reaction class computing the sigma * v_r value for each particle pair
 *
 * @tparam vel_ndim The velocity space dimensionality
 * @tparam CROSS_SECTION The typename corresponding to the cross-section class
 * used
 */
template <size_t vel_ndim, typename CROSS_SECTION = ConstantRateCrossSection>
struct CSPairData
    : public PairReactionDataBase<CSPairDataOnDevice<vel_ndim, CROSS_SECTION>> {

  constexpr static auto props = default_properties;

  constexpr static auto required_simple_real_props =
      std::array<int, 1>{props.velocity};

  /**
   * @brief Constructor for CSPairData.
   *
   * @param cross_section Cross section object to be used in the calculation
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  CSPairData(CROSS_SECTION cross_section,
             std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase<CSPairDataOnDevice<vel_ndim, CROSS_SECTION>>(
            Properties<REAL>(required_simple_real_props), properties_map) {
    this->on_device_obj =
        CSPairDataOnDevice<vel_ndim, CROSS_SECTION>(cross_section);

    static_assert(std::is_base_of_v<AbstractCrossSection, CROSS_SECTION>,
                  "Template parameter CROSS_SECTION is not derived from "
                  "AbstractCrossSection...");

    this->index_on_device_object();
  }

  /**
   * \overload
   * @brief Constructor which sets default values for the
   * cross_section and properties_map.
   *
   */
  CSPairData() : CSPairData(ConstantRateCrossSection(0.0)) {}

  /**
   * @brief Index the particle velocity temperature
   */
  void index_on_device_object() {

    auto arg_pack = this->get_arg_pack();
    this->on_device_obj->velocity_ind_a =
        arg_pack.required_real_props_a.find_index(
            this->properties_map.at(props.velocity));

    this->on_device_obj->velocity_ind_b =
        arg_pack.required_real_props_b.find_index(
            this->properties_map.at(props.velocity));
  };
};
}; // namespace VANTAGE::Reactions
#endif
