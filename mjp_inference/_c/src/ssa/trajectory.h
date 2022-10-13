#pragma once

#include "../../types.h"
#include "../util/conversion.h"

struct Trajectory {

    vec initial;
    vec tspan;
    std::vector<double> time;
    std::vector<int> events;
    mat_rm states;

    // constructors
    Trajectory() = default;
    Trajectory(pybind11::dict trajectory);

    // helper functions
    pybind11::dict to_dict();
    inline void clear() {
        *this = Trajectory();
    }

};
