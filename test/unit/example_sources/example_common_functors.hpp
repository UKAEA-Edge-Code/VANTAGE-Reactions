#ifndef EXAMPLE_COMMON_FUNCTORS_HPP
#define EXAMPLE_COMMON_FUNCTORS_HPP

#include "neso_particles/typedefs.hpp"

/**
 * @brief Functor for marking particles with low weight.
 */
struct LowWeightMark {
  template <typename T> bool operator()(const T &w) const {
    return w[0] < 1e-6;
  }
};

/**
 * @brief Functor for marking particles with a given internal state ID.
 */
struct IdEqualsMark {
  int id;
  IdEqualsMark(int id) : id(id) {}
  template <typename T> bool operator()(const T &x) const { return x[0] == id; }
};

/**
 * @brief Functor for removing all particles in a subgroup.
 */
struct RemoveSubgroupTransform {
  template <typename T> void operator()(T target) const {
    target->get_particle_group()->remove_particles(target);
  }
};

/**
 * @brief Functor for setting the first component of a particle ID to zero.
 */
struct SetIdToZeroKernel {
  template <typename T> void operator()(T id) const { id.at(0) = 0; }
};

/**
 * @brief Constant RNG functor for use in examples.
 */
struct ConstantRng {
  REAL value;
  ConstantRng(REAL v) : value(v) {}
  REAL operator()() const { return value; }
};

#endif // EXAMPLE_COMMON_FUNCTORS_HPP
