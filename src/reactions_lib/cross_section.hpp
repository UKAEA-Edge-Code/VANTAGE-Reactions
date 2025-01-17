#pragma once
#include <neso_particles.hpp>

// TODO: Update docs
using namespace NESO::Particles;

struct AbstractCrossSection {

    virtual REAL get_value_at(REAL relative_vel){};

    virtual REAL get_max_rate_val(){};
};

