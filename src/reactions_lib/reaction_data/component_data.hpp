#ifndef REACTIONS_COMPONENT_DATA_H
#define REACTIONS_COMPONENT_DATA_H
#include "../reaction_data.hpp"
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction data that just extracts values of a component of a
 * real particle dat
 */

struct ComponentDataOnDevice : public ReactionDataBaseOnDevice<1> {

  ComponentDataOnDevice() = default;

  /**
   * @brief Function to extract particle dat values into an array
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
   * @return A REAL-valued array of containing the extracted data
   */
  std::array<REAL, 1>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice<1>::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    std::array<REAL, 1> result;

    result[0] = req_real_props.at(this->prop_ind, index, this->comp_ind);

    return result;
  }

public:
  int prop_ind;
  int comp_ind;
};

/**
 * @brief Reaction data used to extract real valued ParticleDat
 */
struct ComponentData : public ReactionDataBase<ComponentDataOnDevice, 1> {

  /**
   * @brief Constructor for ComponentData.
   *
   * @param extracted_sym The Sym<REAL> corresponding to the ParticleDat whose
   * components should be extracted
   * @param comp The component of the ParticleDat to be extracted
   */
  ComponentData(const Sym<REAL> &extracted_sym, const int comp)
      : ReactionDataBase<ComponentDataOnDevice, 1>(),
        extracted_sym(extracted_sym) {

    this->required_real_props.add(extracted_sym.name);
    this->on_device_obj = ComponentDataOnDevice();

    this->on_device_obj->comp_ind = comp;
    this->index_on_device_object();
  }

  /**
   * @brief Index the particle weight on the on-device object
   */
  void index_on_device_object() {

    this->on_device_obj->prop_ind =
        this->required_real_props.find_index(this->extracted_sym.name);
  };

private:
  Sym<REAL> extracted_sym;
};

auto inline component(const std::string &name, const int comp) {

  return ComponentData(Sym<REAL>(name), comp);
}
}; // namespace VANTAGE::Reactions
#endif
