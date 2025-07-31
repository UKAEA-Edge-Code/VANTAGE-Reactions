#ifndef REACTIONS_COMMON_MARKERS_H
#define REACTIONS_COMMON_MARKERS_H
#include "particle_properties_map.hpp"
#include "transformation_wrapper.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {
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

    /**
     * @brief Constructor for ComparisonMarkerSingleDevice.
     *
     * @param comparison_value Value to compare the ParticleDat value with
     * @param comparison_component Component of the ParticleDat to compare
     */
    ComparisonMarkerSingleDevice(U comparison_value, INT comparison_component)
        : comparison_val(comparison_value),
          comparison_component(comparison_component),
          comparison_wrapper(T<U>()) {}

    /**
     * \overload
     * @brief Constructor for ComparisonMarkerSingleDevice that sets both
     * arguments to 0.
     */
    ComparisonMarkerSingleDevice() : 
      ComparisonMarkerSingleDevice(0, 0) {}

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
    U comparison_val; //!< Value to compare the ParticleDat passed
                      //!< to marking_condition against using <
    INT comparison_component; //!< Component of the passed
                              //!< ParticleDat to compare
    T<U> comparison_wrapper;  //!< Comparison function wrapper (see
                              //!< ComparisonMarkerSingle description for
                              //!< requirements on this)
  };

public:
  ComparisonMarkerSingle() = delete;

  /**
   * @brief Constructor for ComparisonMarkerSingle (INT-version).
   *
   * @param comparison_var Sym<INT> specifying the comparison property
   * @param comparison_value Value to compare the INT-valued ParticleDat
   * @param comparison_component Component of the INT-valued ParticleDat to compare
   */
  ComparisonMarkerSingle(
    const Sym<INT> comparison_var,
    const INT comparison_value,
    const INT comparison_component
  ) : MarkingStrategyBase<ComparisonMarkerSingle<INT, T>>(
      std::vector<Sym<REAL>>(), std::vector<Sym<INT>>{comparison_var}
    )
  {
    if (std::isnan(comparison_value)) {
      this->device_wrapper = ComparisonMarkerSingleDevice();
    }
    else {
      if (std::isnan(comparison_component)) {
        this->device_wrapper = ComparisonMarkerSingleDevice(comparison_value, 0);
      }
      else {
        this->device_wrapper = ComparisonMarkerSingleDevice(comparison_value, comparison_component);
      }
    }

  };

  /**
   * \overload
   * @brief Constructor for ComparisonMarkerSingle (REAL-version).
   *
   * @param comparison_var Sym<REAL> specifying the comparison property.
   * @param comparison_value Value to compare the REAL-valued ParticleDat.
   * @param comparison_component Component of the REAL-valued ParticleDat to compare.
   */
  ComparisonMarkerSingle(
    const Sym<REAL> comparison_var,
    const REAL comparison_value,
    const INT comparison_component
  ) : MarkingStrategyBase<ComparisonMarkerSingle<REAL, T>>(
      std::vector<Sym<REAL>>{comparison_var}, std::vector<Sym<INT>>()
    )
  {
    if (std::isnan(comparison_value)) {
      this->device_wrapper = ComparisonMarkerSingleDevice();
    }
    else {
      if (std::isnan(comparison_component)) {
        this->device_wrapper = ComparisonMarkerSingleDevice(comparison_value, 0);
      }
      else {
        this->device_wrapper = ComparisonMarkerSingleDevice(comparison_value, comparison_component);
      }
    }

  };
  
  /**
   * \overload
   * @brief Constructor for ComparisonMarkerSingle (REAL-version) that sets comparison_value and comparison_component to default values (NaN).
   *
   * @param comparison_var Sym<REAL> specifying the comparison property.
   */
  ComparisonMarkerSingle(const Sym<REAL> comparison_var) : 
    ComparisonMarkerSingle(comparison_var, std::numeric_limits<REAL>().quiet_NaN(), std::numeric_limits<INT>().quiet_NaN()) {}

  /**
   * \overload
   * @brief Constructor for ComparisonMarkerSingle (INT-version) that sets comparison_value and comparison_component to default values (NaN).
   *
   * @param comparison_var Sym<INT> specifying the comparison property.
   */
  ComparisonMarkerSingle(const Sym<INT> comparison_var) : 
    ComparisonMarkerSingle(comparison_var, std::numeric_limits<INT>().quiet_NaN(), std::numeric_limits<INT>().quiet_NaN()) {}

  /**
   * \overload
   * @brief Constructor for ComparisonMarkerSingle (REAL-version) that sets comparison_component to a default value (NaN).
   *
   * @param comparison_var Sym<REAL> specifying the comparison property.
   * @param comparison_value Value to compare the REAL-valued ParticleDat.
   */
  ComparisonMarkerSingle(const Sym<REAL> comparison_var, const REAL comparison_value) : 
    ComparisonMarkerSingle(comparison_var, comparison_value, std::numeric_limits<INT>().quiet_NaN()) {}

  /**
   * \overload
   * @brief Constructor for ComparisonMarkerSingle (INT-version) that sets comparison_component to a default value (NaN).
   *
   * @param comparison_var Sym<INT> specifying the comparison property.
   * @param comparison_value Value to compare the INT-valued ParticleDat.
   */
  ComparisonMarkerSingle(const Sym<INT> comparison_var, const INT comparison_value) :
    ComparisonMarkerSingle(comparison_var, comparison_value, std::numeric_limits<INT>().quiet_NaN()) {}

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

  /**
   * @brief Constructor for MinimumNPartInCellMarker.
   *
   * @param min_npart Minimum number of particles in a cell.
   */
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

/**
 * @brief Marking strategy that selects only those particles with a panic flag >
 * 0
 *
 */
struct PanickedParticleMarker : MarkingStrategy {

public:
  PanickedParticleMarker() = delete;

  /**
   * @brief Constructor for PanickedParticleMarker.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be used to remap the Sym for the Panic property.
   */
  PanickedParticleMarker(
      const std::map<int, std::string> &properties_map = get_default_map())
        {
          NESOWARN(
            map_subset_check(properties_map),
            "The provided properties_map does not include all the keys from the default_map (and therefore is not an extension of that map). \
            There may be inconsitencies with indexing of properties."
          );

          this->panic_sym = Sym<INT>(properties_map.at(default_properties.panic));
        };

  ParticleSubGroupSharedPtr
  make_marker_subgroup(ParticleSubGroupSharedPtr particle_group) {

    auto marker_subgroup = std::make_shared<ParticleSubGroup>(
        particle_group, [=](auto panic) { return panic[0] > 0; },
        Access::read(this->panic_sym));
    return marker_subgroup;
  };

private:
  Sym<INT> panic_sym;
};

inline bool panicked(
    ParticleSubGroupSharedPtr particle_group,
    const std::map<int, std::string> &properties_map = get_default_map()) {

  auto marker = PanickedParticleMarker(properties_map);

  return marker.make_marker_subgroup(particle_group)->get_npart_local();
}

} // namespace VANTAGE::Reactions
#endif
