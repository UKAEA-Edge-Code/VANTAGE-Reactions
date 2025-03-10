#ifndef COMMON_MARKERS_H
#define COMMON_MARKERS_H
#include "transformation_wrapper.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace Reactions {
// TODO: improve documentation

/**
 * @brief Wrapper for more than comparison to be used by the the comparison
 * marking strategy
 *
 * @tparam T REAL or INT
 */
template <typename T> struct MoreThanComp {

  bool operator()(const T &var, const T &comp_val) const {

    return var > comp_val;
  }
};

/**
 * @brief Wrapper for less than comparison to be used by the the comparison
 * marking strategy
 *
 * @tparam T REAL or INT
 */
template <typename T> struct LessThanComp {

  bool operator()(const T &var, const T &comp_val) const {

    return var < comp_val;
  }
};

/**
 * @brief Wrapper for equals comparison to be used by the the comparison marking
 * strategy
 *
 * @tparam T REAL or INT
 */
template <typename T> struct EqualsComp {

  bool operator()(const T &var, const T &comp_val) const {

    return var == comp_val;
  }
};
/**
 * @brief Marking strategy that performs a comparison of a given INT or REAL
 * valued ParticleDat
 *
 * @tparam T Comparison wrapper class that has an overloaded operator() taking
 * in two INTs or two REALs
 * @tparam U REAL or INT, same as the templating on T
 */
template <typename U, template <typename V> typename T>
struct ComparisonMarkerSingle
    : MarkingStrategyBase<ComparisonMarkerSingle<U, T>> {

  friend struct MarkingStrategyBase<ComparisonMarkerSingle<U, T>>;

private:
  /**
   * @brief Device type wrapping the comparison call
   *
   */
  struct ComparisonMarkerSingleDevice
      : MarkingFunctionWrapperBase<ComparisonMarkerSingleDevice> {

    ComparisonMarkerSingleDevice()
        : comparison_val(0), comparison_component(0),
          comparison_wrapper(T<U>()) {}

    ComparisonMarkerSingleDevice(U comparison_value, INT comparison_component)
        : comparison_val(comparison_value),
          comparison_component(comparison_component),
          comparison_wrapper(T<U>()) {}

    bool marking_condition(Access::SymVector::Read<REAL> &real_vars,
                           Access::SymVector::Read<INT> &int_vars) const {

      static_assert(std::is_same<REAL, U>::value || std::is_same<INT, U>::value,
                    "Only REAL or INT templating allowed for U on "
                    "ComparisonMarkerSingle");

      if constexpr (std::is_same<REAL, U>::value)

        return this->comparison_wrapper(
            real_vars.at(0, this->comparison_component), this->comparison_val);

      else

        return this->comparison_wrapper(
            int_vars.at(0, this->comparison_component), this->comparison_val);
    }

  private:
    U comparison_val; //!< Value to compare the real-valued ParticleDat passed
                      //!< to marking_condition against using <
    INT comparison_component; //!< Component of the passed real-valued
                              //!< ParticleDat to compare
    T<U> comparison_wrapper;  //!< Comparison function wrapper (see
                              //!< ComparisonMarkerSingle description for
                              //!< requirements on this)
  };

public:
  ComparisonMarkerSingle() = delete;

  ComparisonMarkerSingle(const Sym<REAL> comparison_var)
      : MarkingStrategyBase<ComparisonMarkerSingle<U, T>>(
            std::vector<Sym<REAL>>{comparison_var}, std::vector<Sym<INT>>()),
        device_wrapper(ComparisonMarkerSingleDevice()) {}

  ComparisonMarkerSingle(const Sym<INT> comparison_var)
      : MarkingStrategyBase<ComparisonMarkerSingle<U, T>>(
            std::vector<Sym<REAL>>(), std::vector<Sym<INT>>{comparison_var}),
        device_wrapper(ComparisonMarkerSingleDevice()) {}

  ComparisonMarkerSingle(const Sym<REAL> comparison_var,
                         const REAL comparison_value)
      : MarkingStrategyBase<ComparisonMarkerSingle<U, T>>(
            std::vector<Sym<REAL>>{comparison_var}, std::vector<Sym<INT>>()),
        device_wrapper(ComparisonMarkerSingleDevice(comparison_value, 0)) {}

  ComparisonMarkerSingle(const Sym<INT> comparison_var,
                         const INT comparison_value)
      : MarkingStrategyBase<ComparisonMarkerSingle<U, T>>(
            std::vector<Sym<REAL>>(), std::vector<Sym<INT>>{comparison_var}),
        device_wrapper(ComparisonMarkerSingleDevice(comparison_value, 0)) {}

  ComparisonMarkerSingle(const Sym<REAL> comparison_var,
                         const REAL comparison_value,
                         const INT comparison_component)
      : MarkingStrategyBase<ComparisonMarkerSingle<U, T>>(
            std::vector<Sym<REAL>>{comparison_var}, std::vector<Sym<INT>>()),
        device_wrapper(ComparisonMarkerSingleDevice(comparison_value,
                                                    comparison_component)) {}

  ComparisonMarkerSingle(const Sym<INT> comparison_var,
                         const INT comparison_value,
                         const INT comparison_component)
      : MarkingStrategyBase<ComparisonMarkerSingle<U, T>>(
            std::vector<Sym<REAL>>(), std::vector<Sym<INT>>{comparison_var}),
        device_wrapper(ComparisonMarkerSingleDevice(comparison_value,
                                                    comparison_component)) {}

protected:
  ComparisonMarkerSingleDevice get_device_data() const {
    return this->device_wrapper;
  }

private:
  ComparisonMarkerSingleDevice
      device_wrapper; //!< Device copyable wrapper for the comparison function
};

/**
 * @brief Marking strategy that selects only those particles in cells containing
 * some minimum number of particles
 *
 */
struct MinimumNPartInCellMarker : MarkingStrategy {

public:
  MinimumNPartInCellMarker() = delete;

  MinimumNPartInCellMarker(INT min_npart) : min_npart(min_npart){};

  ParticleSubGroupSharedPtr
  make_marker_subgroup(ParticleSubGroupSharedPtr particle_group) {

    auto min_npart = this->min_npart;
    auto marker_subgroup = std::make_shared<ParticleSubGroup>(
        particle_group,
        [=](auto cell_info_npart) {
          return cell_info_npart.get() >= min_npart;
        },
        Access::read(CellInfoNPart{}));
    return marker_subgroup;
  };

private:
  INT min_npart;
};

} // namespace Reactions
#endif
