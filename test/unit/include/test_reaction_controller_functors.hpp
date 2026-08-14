#ifndef TEST_REACTION_CONTROLLER_FUNCTORS_HPP
#define TEST_REACTION_CONTROLLER_FUNCTORS_HPP

#include "test_common.hpp"

/**
 * @brief Functor for reducing particle weights into a cellwise accumulator.
 */
struct WeightReducer {
  template <typename W, typename GA> void operator()(const W &w, GA &ga) const {
    ga.fetch_add(0, 0, w[0]);
  }
};

/**
 * @brief Functor for matching internal state values.
 */
struct InternalStateEquals {
  NP::INT value;
  InternalStateEquals(NP::INT value) : value(value) {}
  template <typename T> bool operator()(const T &x) const {
    return x[0] == value;
  }
};

/**
 * @brief Functor for marking particles with very small weight.
 */
struct SmallWeightMarker {
  template <typename T> bool operator()(const T &w) const {
    return w[0] < 1e-12;
  }
};

/**
 * @brief Functor for removing all particles in a subgroup.
 */
struct RemoveSubgroupTransform {
  template <typename T> void operator()(T target) const {
    target->get_particle_group()->remove_particles(target);
  }
};

#endif
